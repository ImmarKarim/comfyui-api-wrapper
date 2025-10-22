# postprocess_worker
import asyncio
import logging
import os  # Still needed for symlink and remove operations
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
import json

import aiobotocore.session
import aiofiles
import aiofiles.os
import aiohttp

from config import OUTPUT_DIR, S3_CONFIG, S3_ENABLED, WEBHOOK_CONFIG, WEBHOOK_ENABLED

logger = logging.getLogger(__name__)


class PostprocessWorker:
    """
    Upload generated assets and fire webhook response
    """
    def __init__(self, worker_id, kwargs):
        self.worker_id = worker_id
        self.preprocess_queue = kwargs["preprocess_queue"]
        self.generation_queue = kwargs["generation_queue"]
        self.postprocess_queue = kwargs["postprocess_queue"]
        self.request_store = kwargs["request_store"]
        self.response_store = kwargs["response_store"]
        
        # Configuration
        self.output_dir = Path(OUTPUT_DIR)

    async def work(self):
        logger.info(f"PostprocessWorker {self.worker_id}: waiting for jobs")
        while True:
            # Get a task from the job queue
            request_id = await self.postprocess_queue.get()
            if request_id is None:
                # None is a signal that there are no more tasks
                break

            # Process the job
            logger.info(f"PostprocessWorker {self.worker_id} processing job: {request_id}")
            
            try:
                # Get request and result from stores
                request = await self.request_store.get(request_id)
                result = await self.response_store.get(request_id)
                
                if not request:
                    raise Exception(f"Request {request_id} not found in store")
                if not result:
                    raise Exception(f"Result {request_id} not found in store")
                
                # Only process if we have ComfyUI output (successful generation)
                if hasattr(result, 'comfyui_response') and result.comfyui_response:
                    logger.info(f"Processing outputs for {request_id}")
                    logger.debug(f"ComfyUI response structure: {json.dumps(result.comfyui_response, indent=2)[:1000]}")
                    
                    # Move generated assets to organized directory
                    await self.move_assets(request_id, result)
                    
                    # Handle S3 upload - check payload first, then environment variables
                    s3_config = await self.get_s3_config(request.input)
                    if s3_config:
                        await self.upload_assets(request_id, s3_config, result)
                    else:
                        logger.info(f"No S3 configuration found for {request_id}, skipping upload")
                else:
                    logger.info(f"No ComfyUI output for {request_id}, likely a failed job")
                    if hasattr(result, 'comfyui_response'):
                        logger.debug(f"ComfyUI response was: {result.comfyui_response}")

                # Update final status only if not already failed
                if result.status != "failed":
                    result.status = "completed"
                    result.message = "Processing complete."
                else:
                    logger.info(f"Job {request_id} already marked as failed, keeping failure status")
                
                await self.response_store.set(request_id, result)
                logger.info(f"PostprocessWorker {self.worker_id} completed job: {request_id}")
                
            except Exception as e:
                logger.error(f"PostprocessWorker {self.worker_id} failed job {request_id}: {e}")
                
                try:
                    # Update result to show failure
                    result = await self.response_store.get(request_id)
                    if result:
                        result.status = "failed"
                        result.message = f"Post-processing failed: {str(e)}"
                        await self.response_store.set(request_id, result)
                    
                except Exception as store_error:
                    logger.error(f"Failed to update result store for {request_id}: {store_error}")
            
            finally:
                # Handle webhook - check payload first, then environment variables
                webhook_config = await self.get_webhook_config(request.input)
                if webhook_config:
                    try:
                        await self.send_webhook(webhook_config['url'], result, webhook_config.get('extra_params', {}))
                    except Exception as webhook_error:
                        # Will not mark a 'completed' job job as failed
                        logger.error(f"Failed to run webhook for {request_id}: {webhook_error}")
                else:
                    logger.info(f"No webhook configuration found for {request_id}")
                # Mark the job as complete
                self.postprocess_queue.task_done()
            
        logger.info(f"PostprocessWorker {self.worker_id} finished")
    
    async def move_assets(self, request_id: str, result) -> None:
        """Move generated assets to organized directory structure"""
        try:
            # Create job-specific output directory
            job_output_dir = self.output_dir / request_id
            await aiofiles.os.makedirs(str(job_output_dir), exist_ok=True)
            
            if not hasattr(result, 'output'):
                result.output = []

            # Parse ComfyUI history response structure
            # The response from history API typically looks like:
            # {
            #   "prompt_id": {
            #     "prompt": [...],
            #     "outputs": {
            #       "node_id": {
            #         "images": [{"filename": "...", "subfolder": "...", "type": "output"}],
            #         "gifs": [...],
            #         "videos": [...]
            #       }
            #     }
            #   }
            # }
            
            # Find the outputs in the response
            outputs = None
            
            # First, check if the response is wrapped with the prompt_id
            if isinstance(result.comfyui_response, dict):
                # Get the first (and usually only) key which is the prompt_id
                for prompt_id, prompt_data in result.comfyui_response.items():
                    if isinstance(prompt_data, dict) and 'outputs' in prompt_data:
                        outputs = prompt_data['outputs']
                        logger.debug(f"Found outputs under prompt_id {prompt_id}")
                        break
                    elif isinstance(prompt_data, dict):
                        # Sometimes outputs might be directly in the prompt_data
                        logger.debug(f"Checking if prompt_data contains output nodes directly")
                        outputs = prompt_data
                        break
            
            if not outputs:
                logger.warning(f"No outputs found in ComfyUI response for {request_id}")
                logger.debug(f"Full response structure: {json.dumps(result.comfyui_response, indent=2)[:2000]}")
                return
            
            # Process each node's outputs
            processed_files = []
            for node_id, node_outputs in outputs.items():
                if not isinstance(node_outputs, dict):
                    logger.debug(f"Skipping non-dict node output: {node_id}")
                    continue
                
                logger.debug(f"Processing node {node_id} outputs: {list(node_outputs.keys())}")
                
                # Prefer 'files' over 'images' if both exist
                # 'files' contains only actual outputs from this job
                # 'images' may contain history from previous jobs
                output_types_to_process = []
                if 'files' in node_outputs:
                    logger.info(f"Node {node_id}: Using 'files' array (actual job outputs)")
                    output_types_to_process.append('files')
                    # Process other types except 'images' to avoid duplicates
                    for key in node_outputs.keys():
                        if key not in ['images', 'files']:
                            output_types_to_process.append(key)
                else:
                    # No 'files' array, process all output types including 'images'
                    output_types_to_process = list(node_outputs.keys())
                
                # Look for different output types (images, gifs, videos, files, etc.)
                for output_type in output_types_to_process:
                    output_list = node_outputs.get(output_type)
                    if not isinstance(output_list, list):
                        logger.debug(f"Skipping non-list output type {output_type} in node {node_id}")
                        continue
                    
                    for item in output_list:
                        # Handle both dict format and string format (files can be paths)
                        if isinstance(item, str):
                            # Convert string path to dict format
                            item = {"filename": Path(item).name, "subfolder": "", "type": "output"}
                        
                        if isinstance(item, dict) and 'filename' in item:
                            # Skip preview/temp files
                            file_type = item.get('type', '')
                            if file_type in ['temp', 'preview']:
                                logger.debug(f"Skipping {file_type} file: {item.get('filename')}")
                                continue
                            
                            # Process this output file
                            processed = await self._process_output_file(
                                item, 
                                job_output_dir, 
                                request_id,
                                node_id,
                                output_type
                            )
                            if processed:
                                processed_files.append(processed)
            
            # Add all processed files to the result
            result.output = processed_files
            logger.info(f"Processed {len(processed_files)} output files for {request_id}")
            
        except Exception as e:
            logger.error(f"Error moving assets for {request_id}: {e}", exc_info=True)
            raise

    async def _process_output_file(self, item: Dict, job_output_dir: Path, request_id: str, node_id: str, output_type: str) -> Optional[Dict]:
        """Process a single output file - copy to job directory and create symlink"""
        try:
            filename = item.get('filename', '')
            subfolder = item.get('subfolder', '')
            file_type = item.get('type', 'output')
            
            if not filename:
                logger.warning(f"No filename in output item: {item}")
                return None
            
            # Construct the original file path
            # ComfyUI typically saves files in OUTPUT_DIR/subfolder/filename
            if subfolder:
                original_path = self.output_dir / subfolder / filename
            else:
                original_path = self.output_dir / filename
            
            # Check if the file exists
            if not original_path.exists():
                logger.warning(f"Original file not found: {original_path}")
                # Try without subfolder as fallback
                if subfolder:
                    fallback_path = self.output_dir / filename
                    if fallback_path.exists():
                        logger.info(f"Found file at fallback location: {fallback_path}")
                        original_path = fallback_path
                    else:
                        return None
                else:
                    return None
            
            # Destination path in job directory
            dest_path = job_output_dir / filename
            
            # Get the real path (in case original_path is a symlink from a cached result)
            real_original_path = original_path.resolve()

            # Defensive guard: only allow copying if source is either a top-level file
            # directly under OUTPUT_DIR, or within a subdirectory whose first segment
            # matches the full request_id (which may contain slashes).
            try:
                rel = real_original_path.relative_to(self.output_dir)
                rel_parts = rel.parts
                # Allow top-level files (e.g., YY00001_0084.png)
                if len(rel_parts) == 1:
                    pass
                # Allow when the path starts with request_id directory
                elif rel_parts[0] == str(request_id):
                    pass
                else:
                    logger.debug(f"Skipping source from different request directory: {real_original_path}")
                    return None
            except Exception:
                # If not under OUTPUT_DIR, proceed (should not happen normally)
                pass
            
            
            logger.info(f"Copying {real_original_path} to {dest_path}")
            
            # Copy the file (using real path to handle symlinks)
            await self._copy_file_async(real_original_path, dest_path)
            
            # Remove original file/symlink and create new symlink pointing to our copy
            if original_path.exists() or original_path.is_symlink():
                await self._remove_file_async(original_path)
            
            # Create symlink from original location to our copy
            await self._create_symlink_async(dest_path, original_path)
            
            logger.debug(f"Created symlink: {original_path} -> {dest_path}")
            
            # Return file info for result
            return {
                "filename": filename,
                "local_path": str(dest_path),
                "type": file_type,
                "subfolder": subfolder,
                "node_id": node_id,
                "output_type": output_type
            }
            
        except Exception as e:
            logger.error(f"Error processing output file {item}: {e}", exc_info=True)
            return None

    async def _copy_file_async(self, src: Path, dst: Path) -> None:
        """Async file copy"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, shutil.copy2, str(src), str(dst))

    async def _remove_file_async(self, path: Path) -> None:
        """Async file/symlink removal"""
        loop = asyncio.get_running_loop()
        if path.is_symlink():
            await loop.run_in_executor(None, path.unlink)
        else:
            await loop.run_in_executor(None, os.remove, str(path))

    async def _create_symlink_async(self, target: Path, link: Path) -> None:
        """Async symlink creation - link points to target"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, os.symlink, str(target), str(link))

    async def upload_assets(self, request_id: str, s3_config: Dict, result) -> None:
        """Upload assets to S3 storage"""
        if not hasattr(result, 'output') or not result.output:
            logger.info(f"No assets to upload for {request_id}")
            return
            
        try:
            session = aiobotocore.session.get_session()
            
            # Build S3 client config
            client_config = {
                'aws_access_key_id': s3_config.get("access_key_id"),
                'aws_secret_access_key': s3_config.get("secret_access_key"),
            }
            
            if s3_config.get("endpoint_url"):
                client_config['endpoint_url'] = s3_config["endpoint_url"]
            
            if s3_config.get("region"):
                client_config['region_name'] = s3_config["region"]
                
            # Configure timeouts and retries
            aio_config = aiobotocore.config.AioConfig(
                connect_timeout=int(s3_config.get("connect_timeout", 60)),
                retries={"max_attempts": int(s3_config.get("connect_attempts", 3))}
            )
            client_config['config'] = aio_config
            
            async with session.create_client('s3', **client_config) as s3_client:
                bucket_name = s3_config.get("bucket_name")
                if not bucket_name:
                    raise ValueError("S3 bucket_name is required")
                
                # Upload all files concurrently
                tasks = []
                for obj in result.output:
                    local_path = obj.get("local_path")
                    if local_path and Path(local_path).exists():
                        task = asyncio.create_task(
                            self.upload_file_and_get_url(
                                request_id, s3_client, bucket_name, local_path
                            )
                        )
                        tasks.append(task)
                    else:
                        logger.warning(f"Local file not found: {local_path}")
                        tasks.append(asyncio.create_task(self._return_none()))
                
                # Wait for all uploads
                if tasks:
                    upload_urls = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Update result objects with URLs (may be permanent direct URLs or 7-day presigned URLs)
                    for obj, url_result in zip(result.output, upload_urls):
                        if isinstance(url_result, Exception):
                            logger.error(f"Upload failed for {obj.get('local_path')}: {url_result}")
                            obj["upload_error"] = str(url_result)
                        elif url_result:
                            obj["url"] = url_result
                            
                    logger.info(f"Uploaded {len([u for u in upload_urls if u and not isinstance(u, Exception)])} assets for {request_id}")
                    
        except Exception as e:
            logger.error(f"Error uploading assets for {request_id}: {e}")
            raise

    async def _return_none(self):
        """Helper for asyncio.gather with missing files"""
        return None

    async def upload_file_and_get_url(self, request_id: str, s3_client, bucket_name: str, local_path: str) -> Optional[str]:
        """Upload a single file. Prefer returning a public direct URL (never expires)
        by attempting ACL='public-read' first. Uses presigned URL generation for 
        provider-agnostic URL formatting, then strips query params for public objects.
        Falls back to full presigned URL (7-day expiry) if public access not available.
        """
        try:
            file_path = Path(local_path)
            s3_key = f"{request_id}/{file_path.name}"
            
            logger.debug(f"Uploading {s3_key} to bucket {bucket_name}")

            # Read file content once
            async with aiofiles.open(local_path, 'rb') as file:
                file_content = await file.read()

            # 1) Try public upload using ACL='public-read' (may fail if bucket blocks public ACLs)
            uploaded_public = False
            try:
                await s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=file_content,
                    ACL='public-read'
                )
                uploaded_public = True
                logger.debug(f"Uploaded {s3_key} with public-read ACL")
            except Exception as acl_error:
                logger.info(f"Public ACL upload not permitted, falling back to private object: {acl_error}")
                # 2) Upload privately without ACL
                await s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=file_content
                )

            # 3) Generate presigned URL (S3 client handles all provider-specific URL formatting)
            presigned_url = await s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_key},
                ExpiresIn=604800  # 7 days
            )

            # 4) If uploaded as public, strip query params to get permanent direct URL
            if uploaded_public:
                parsed = urlparse(presigned_url)
                # Direct URL = base URL without query parameters (never expires)
                direct_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                
                # Verify the direct URL is actually accessible
                if await self._url_is_public(direct_url):
                    logger.info(f"Returning permanent public URL for {s3_key}")
                    return direct_url
                else:
                    logger.info(f"Direct URL not accessible (ACL may not have applied), using presigned URL")

            # 5) Fallback: Return full presigned URL with expiration
            logger.debug(f"Returning presigned URL (7-day expiry) for {s3_key}")
            return presigned_url
            
        except Exception as e:
            logger.error(f"Error uploading {local_path}: {e}")
            raise

    async def _url_is_public(self, url: str) -> bool:
        """Check if a URL is publicly accessible via a lightweight HEAD request."""
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(url, allow_redirects=True) as resp:
                    return 200 <= resp.status < 300
        except Exception:
            return False

    async def send_webhook(self, webhook_url: str, result, extra_params: Dict = None) -> None:
        """Send webhook notification with result"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            
            # Prepare webhook payload
            webhook_data = {
                "id": result.id,
                "status": result.status,
                "message": result.message,
                "output": getattr(result, 'output', [])
            }
            
            # Add extra parameters if provided
            if extra_params:
                webhook_data.update(extra_params)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    webhook_url,
                    json=webhook_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.warning(f"Webhook failed (status {response.status}): {error_text}")
                    else:
                        logger.info(f"Webhook sent successfully to {webhook_url}")
                        
        except Exception as e:
            logger.error(f"Error sending webhook to {webhook_url}: {e}")
            # Don't raise - webhook failures shouldn't fail the whole job

    async def get_s3_config(self, input_data) -> Optional[Dict]:
        """Get S3 configuration from payload or centralized config (from environment)"""
        try:
            # Check if S3 config provided in payload
            if hasattr(input_data, 's3') and input_data.s3:
                if input_data.s3.is_configured():
                    logger.info("Using S3 config from payload")
                    return input_data.s3.get_config()
            
            # Fall back to centralized config (which reads from environment)
            if S3_ENABLED:
                logger.info("Using S3 config from environment variables")
                return S3_CONFIG.copy()  # Return a copy to avoid mutation
            
            # No valid config found
            logger.debug("No S3 configuration available")
            return None
            
        except Exception as e:
            logger.error(f"Error getting S3 config: {e}")
            return None

    async def get_webhook_config(self, input_data) -> Optional[Dict]:
        """Get webhook configuration from payload or centralized config (from environment)"""
        try:
            # Check if webhook config provided in payload
            if hasattr(input_data, 'webhook') and input_data.webhook:
                if input_data.webhook.has_valid_url():
                    logger.info("Using webhook config from payload")
                    return {
                        'url': input_data.webhook.url,
                        'extra_params': input_data.webhook.extra_params,
                        'timeout': input_data.webhook.timeout
                    }
            
            # Fall back to centralized config (which reads from environment)
            if WEBHOOK_ENABLED:
                logger.info("Using webhook config from environment variables")
                return {
                    'url': WEBHOOK_CONFIG['url'],
                    'extra_params': {},
                    'timeout': WEBHOOK_CONFIG['timeout']
                }
            
            # No valid config found
            logger.debug("No webhook configuration available")
            return None
            
        except Exception as e:
            logger.error(f"Error getting webhook config: {e}")
            return None