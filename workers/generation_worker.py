# generation_worker
import asyncio
import aiohttp
import json
import logging
import random
from typing import Optional, Dict, Any
from datetime import datetime

from config import COMFYUI_API_PROMPT, COMFYUI_API_HISTORY, COMFYUI_API_INTERRUPT, COMFYUI_API_WEBSOCKET
from metrics import record_generation_outcome, mark_gpu_unrecoverable

logger = logging.getLogger(__name__)


class GenerationWorker:
    """
    Send payload to ComfyUI and await completion using WebSocket
    """
    def __init__(self, worker_id, kwargs):
        self.worker_id = worker_id
        self.preprocess_queue = kwargs["preprocess_queue"]
        self.generation_queue = kwargs["generation_queue"]
        self.postprocess_queue = kwargs["postprocess_queue"]
        self.request_store = kwargs["request_store"]
        self.response_store = kwargs["response_store"]
        
        # Configuration
        self.max_wait_time = 3600  # 1 hour maximum wait
        self.ws_url = COMFYUI_API_WEBSOCKET
        self.client_id = f"worker_{worker_id}_{datetime.now().timestamp()}"
        self._comfy_ready = False

    async def work(self):
        logger.info(f"GenerationWorker {self.worker_id}: waiting for jobs")
        while True:
            # Get a task from the job queue
            request_id = await self.generation_queue.get()
            if request_id is None:
                # None is a signal that there are no more tasks
                break

            # Process the job
            logger.info(f"GenerationWorker {self.worker_id} processing job: {request_id}")
            
            try:
                # Get request and result from stores
                request = await self.request_store.get(request_id)
                result = await self.response_store.get(request_id)
                
                if not request:
                    raise Exception(f"Request {request_id} not found in store")
                if not result:
                    raise Exception(f"Result {request_id} not found in store")

                # Check for cancellation
                if result and getattr(result, 'status', '') == 'cancelled':
                    logger.info(f"PreprocessWorker {self.worker_id} skipping cancelled job: {request_id} - jumping to postprocess")
                    await self.postprocess_queue.put(request_id)
                    self.generation_queue.task_done()
                    continue

                # Ensure ComfyUI is up before posting the workflow (handles cold start)
                if not self._comfy_ready:
                    await self.wait_for_comfy_ready()
                    self._comfy_ready = True

                # Submit workflow to ComfyUI
                comfyui_job_id = await self.post_workflow(request)
                logger.info(f"Submitted job {request_id} to ComfyUI as {comfyui_job_id}")
                
                # Update status to show generation started
                result.status = "generating"
                result.message = f"Generation started (ComfyUI job: {comfyui_job_id})"
                await self.response_store.set(request_id, result)

                # Check if job is already complete (cached result)
                is_cached = await self.check_if_cached(comfyui_job_id)
                
                if is_cached:
                    logger.info(f"Job {comfyui_job_id} completed immediately (cached result)")
                    execution_result = {
                        "prompt_id": comfyui_job_id,
                        "nodes_executed": [],
                        "progress_updates": [],
                        "completed": True,
                        "cached": True,
                        "error": None
                    }
                else:
                    # Wait for completion using WebSocket
                    execution_result = await self.wait_for_completion_websocket(
                        comfyui_job_id, 
                        request_id
                    )
                
                # Get the final result from ComfyUI history
                comfyui_response = await self.get_result(comfyui_job_id)
                logger.info(f"Retrieved ComfyUI result for {request_id}")
                logger.debug(f"ComfyUI response structure: {json.dumps(comfyui_response, indent=2)[:500]}...")  # First 500 chars
                
                # Update result with success
                result.status = "generated"
                result.message = "Generation complete. Queued for post-processing."
                result.comfyui_response = comfyui_response
                # Store execution details in the comfyui_response if needed
                if execution_result:
                    # Merge execution details into the response
                    if isinstance(result.comfyui_response, dict):
                        result.comfyui_response["execution_details"] = execution_result
                await self.response_store.set(request_id, result)
                
                # Record success for health metrics
                record_generation_outcome(True)

                # Send for post-processing
                await self.postprocess_queue.put(request_id)
                logger.info(f"GenerationWorker {self.worker_id} completed job: {request_id}")
                
            except Exception as e:
                logger.error(f"GenerationWorker {self.worker_id} failed job {request_id}: {e}")
                # Record failure for health metrics
                record_generation_outcome(False)
                # Best-effort detection for unrecoverable CUDA errors even if not from websocket path
                try:
                    reason = _detect_cuda_unrecoverable_reason(str(e))
                    if reason:
                        mark_gpu_unrecoverable(reason)
                        logger.error(f"Marked GPU as unrecoverable due to error: {reason}")
                except Exception:
                    pass
                
                try:
                    # Update result to show failure
                    result = await self.response_store.get(request_id)
                    if result:
                        result.status = "failed"
                        result.message = f"Generation failed: {str(e)}"
                        await self.response_store.set(request_id, result)
                    
                    # Send job to postprocess for cleanup
                    await self.postprocess_queue.put(request_id)
                    
                except Exception as store_error:
                    logger.error(f"Failed to update result store for {request_id}: {store_error}")
            
            finally:
                # Mark the job as complete
                self.generation_queue.task_done()

        logger.info(f"GenerationWorker {self.worker_id} finished")

    async def wait_for_comfy_ready(self, max_wait_seconds: int = 120) -> None:
        """Wait until ComfyUI HTTP and WebSocket endpoints are reachable.

        This mitigates race conditions where the wrapper starts before ComfyUI.
        """
        start = asyncio.get_event_loop().time()
        attempt = 0
        http_ok = False
        ws_ok = False

        # Use small timeouts to keep loop responsive
        http_timeout = aiohttp.ClientTimeout(total=2.0)

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > max_wait_seconds:
                raise Exception(f"ComfyUI not ready after {max_wait_seconds}s")

            attempt += 1
            try:
                # Check HTTP readiness via history endpoint
                async with aiohttp.ClientSession(timeout=http_timeout) as session:
                    async with session.get(COMFYUI_API_HISTORY) as resp:
                        if resp.status in (200, 404):
                            http_ok = True
                        else:
                            http_ok = False
            except Exception:
                http_ok = False

            try:
                # Check WebSocket readiness with a lightweight probe
                async with aiohttp.ClientSession(timeout=http_timeout) as session:
                    async with session.ws_connect(self.ws_url, params={"clientId": "healthcheck"}) as ws:
                        ws_ok = True
            except Exception:
                ws_ok = False

            if http_ok and ws_ok:
                logger.info("ComfyUI is ready (HTTP and WebSocket reachable)")
                return

            # Exponential backoff with cap
            backoff = min(0.5 * (2 ** min(attempt, 5)), 3.0)
            logger.info(f"Waiting for ComfyUI to be ready (attempt {attempt}) - retrying in {backoff:.1f}s")
            await asyncio.sleep(backoff)

    async def post_workflow(self, request) -> str:
        """Submit workflow to ComfyUI API"""
        payload = {
            "prompt": request.input.workflow_json,
            "client_id": self.client_id  # Use our worker's client ID
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        
        max_attempts = 5
        base_delay_seconds = 1
        last_error: Optional[Exception] = None
        
        for attempt in range(1, max_attempts + 1):
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    logger.debug(f"Posting workflow to {COMFYUI_API_PROMPT} (attempt {attempt}/{max_attempts})")
                    logger.debug(f"Workflow keys: {list(request.input.workflow_json.keys()) if isinstance(request.input.workflow_json, dict) else 'not a dict'}")
                    
                    async with session.post(
                        COMFYUI_API_PROMPT, 
                        data=json.dumps(payload),
                        headers=headers
                    ) as response:
                        
                        response_text = await response.text()
                        logger.debug(f"ComfyUI API response status: {response.status}")
                        logger.debug(f"ComfyUI API response: {response_text[:500]}...")  # First 500 chars

                        # Detect unrecoverable CUDA errors even if HTTP succeeded
                        try:
                            reason = _detect_cuda_unrecoverable_reason(response_text)
                            if reason:
                                mark_gpu_unrecoverable(reason)
                                logger.error(f"Marked GPU as unrecoverable from post_workflow response: {reason}")
                        except Exception:
                            pass
                        
                        if response.status >= 400:
                            # Treat HTTP errors as non-retryable here; raise immediately
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"ComfyUI API error: {response_text}"
                            )
                        
                        response_data = json.loads(response_text)
                        
                        if "prompt_id" in response_data:
                            return response_data["prompt_id"]
                        elif "node_errors" in response_data:
                            error_details = json.dumps(response_data["node_errors"], indent=2)
                            # Detection from structured node_errors
                            try:
                                reason = _detect_cuda_unrecoverable_reason(error_details)
                                if reason:
                                    mark_gpu_unrecoverable(reason)
                                    logger.error(f"Marked GPU as unrecoverable from node_errors: {reason}")
                            except Exception:
                                pass
                            raise Exception(f"ComfyUI node errors: {error_details}")
                        elif "error" in response_data:
                            # Detection from top-level error field
                            try:
                                reason = _detect_cuda_unrecoverable_reason(str(response_data["error"]))
                                if reason:
                                    mark_gpu_unrecoverable(reason)
                                    logger.error(f"Marked GPU as unrecoverable from error field: {reason}")
                            except Exception:
                                pass
                            raise Exception(f"ComfyUI error: {response_data['error']}")
                        else:
                            raise Exception(f"Unexpected response from ComfyUI: {response_data}")
                except (asyncio.TimeoutError, aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError, aiohttp.ClientOSError) as e:
                    last_error = e
                    if attempt >= max_attempts:
                        raise Exception(f"Network error posting to ComfyUI: {e}")
                    # Exponential backoff with jitter
                    backoff_seconds = base_delay_seconds * (2 ** (attempt - 1))
                    jitter = random.uniform(0, base_delay_seconds)
                    sleep_for = backoff_seconds + jitter
                    logger.warning(f"post_workflow attempt {attempt} failed with network error: {e}. Retrying in {sleep_for:.2f}s")
                    await asyncio.sleep(sleep_for)
                except aiohttp.ClientResponseError as e:
                    # Non-retryable HTTP error
                    raise Exception(f"ComfyUI API error (HTTP {e.status}): {e.message}")
                except aiohttp.ClientError as e:
                    # Other aiohttp errors considered network-related; retry
                    last_error = e
                    if attempt >= max_attempts:
                        raise Exception(f"Network error posting to ComfyUI: {e}")
                    backoff_seconds = base_delay_seconds * (2 ** (attempt - 1))
                    jitter = random.uniform(0, base_delay_seconds)
                    sleep_for = backoff_seconds + jitter
                    logger.warning(f"post_workflow attempt {attempt} failed: {e}. Retrying in {sleep_for:.2f}s")
                    await asyncio.sleep(sleep_for)
                except json.JSONDecodeError as e:
                    # Parsing error is unlikely to be resolved by retrying; raise
                    raise Exception(f"Invalid JSON response from ComfyUI: {e}")

    async def check_if_cached(self, comfyui_job_id: str) -> bool:
        """Check if job is already complete (cached result)"""
        await asyncio.sleep(0.5)  # Give ComfyUI a moment to process
        
        timeout = aiohttp.ClientTimeout(total=5)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{COMFYUI_API_HISTORY}/{comfyui_job_id}"
                async with session.get(url) as response:
                    if response.status == 200:
                        history_data = await response.json()
                        # If we get non-empty data, the job is complete
                        if history_data and history_data != {}:
                            logger.info(f"Job {comfyui_job_id} found in history (cached)")
                            return True
            return False
        except Exception as e:
            logger.debug(f"Error checking cache status: {e}")
            return False
    
    async def wait_for_completion_websocket(self, comfyui_job_id: str, request_id: str) -> Dict[str, Any]:
        """
        Wait for ComfyUI job completion using WebSocket connection
        Returns execution result details
        """
        execution_result = {
            "prompt_id": comfyui_job_id,
            "nodes_executed": [],
            "progress_updates": [],
            "completed": False,
            "error": None
        }
        
        timeout = aiohttp.ClientTimeout(total=self.max_wait_time)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info(f"Connecting to ComfyUI WebSocket at {self.ws_url}")
                
                async with session.ws_connect(
                    self.ws_url,
                    params={"clientId": self.client_id}
                ) as ws:
                    logger.info(f"WebSocket connected for job {comfyui_job_id}")
                    
                    # Start listening for messages
                    start_time = asyncio.get_event_loop().time()
                    last_update_time = start_time
                    last_message_time = start_time
                    last_cancellation_check = start_time
                    
                    # Progressive timeout strategy
                    initial_timeout = 180.0  # 30 seconds to receive first message
                    message_timeout = 800.0  # 300 seconds between messages after first message received
                    max_no_message_retries = 3  # Number of times to retry when no messages received
                    no_message_retry_count = 0
                    
                    while True:
                        try:
                            # Set timeout based on whether we've received any messages
                            timeout_duration = initial_timeout if last_message_time == start_time else message_timeout
                            
                            msg = await asyncio.wait_for(
                                ws.receive(), 
                                timeout=timeout_duration
                            )
                            
                            last_message_time = asyncio.get_event_loop().time()
                            # Reset retry count since we received a message
                            no_message_retry_count = 0

                            current_time = asyncio.get_event_loop().time()
                            if current_time - last_cancellation_check > 5.0:  # Check every 5 seconds
                                if await self._check_if_cancelled(request_id):
                                    logger.info(f"Job {request_id} was cancelled during generation - aborting WebSocket")
                                    # Cancel the ComfyUI job
                                    await self.cancel_comfyui_job(comfyui_job_id)
                                    raise Exception(f"Job {request_id} was cancelled during generation")
                                last_cancellation_check = current_time
                            
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    message_type = data.get("type")
                                    
                                    logger.debug(f"WebSocket message type: {message_type}")
                                    
                                    # Check if this message is for our prompt
                                    if data.get("data", {}).get("prompt_id") == comfyui_job_id:
                                        
                                        if message_type == "execution_start":
                                            logger.info(f"Execution started for {comfyui_job_id}")
                                            await self._update_progress(
                                                request_id, 
                                                "Execution started..."
                                            )
                                        
                                        elif message_type == "execution_cached":
                                            nodes = data.get("data", {}).get("nodes", [])
                                            logger.info(f"Using cached results for nodes: {nodes}")
                                            execution_result["nodes_executed"].extend(nodes)
                                        
                                        elif message_type == "executing":
                                            node = data.get("data", {}).get("node")
                                            if node:
                                                logger.info(f"Executing node: {node}")
                                                execution_result["nodes_executed"].append(node)
                                                await self._update_progress(
                                                    request_id, 
                                                    f"Processing node: {node}"
                                                )
                                            elif data.get("data", {}).get("node") is None:
                                                # node = None means execution is complete
                                                logger.info(f"Execution complete for {comfyui_job_id}")
                                                execution_result["completed"] = True
                                                return execution_result
                                        
                                        elif message_type == "progress":
                                            progress_data = data.get("data", {})
                                            value = progress_data.get("value", 0)
                                            max_value = progress_data.get("max", 100)
                                            
                                            progress_pct = (value / max_value * 100) if max_value > 0 else 0
                                            progress_msg = f"Progress: {progress_pct:.1f}% ({value}/{max_value})"
                                            
                                            logger.info(f"Progress update: {progress_msg}")
                                            execution_result["progress_updates"].append({
                                                "time": asyncio.get_event_loop().time() - start_time,
                                                "value": value,
                                                "max": max_value,
                                                "percentage": progress_pct
                                            })
                                            
                                            # Update status every few seconds to avoid spam
                                            current_time = asyncio.get_event_loop().time()
                                            if current_time - last_update_time > 2:  # Update every 2 seconds
                                                await self._update_progress(request_id, progress_msg)
                                                last_update_time = current_time
                                        
                                        elif message_type == "execution_error":
                                            error_data = data.get("data", {})
                                            error_msg = f"Execution error: {error_data}"
                                            logger.error(error_msg)
                                            # Try to detect unrecoverable CUDA failures and mark sticky flag
                                            try:
                                                combined = json.dumps(error_data)
                                                reason = _detect_cuda_unrecoverable_reason(combined)
                                                if reason:
                                                    mark_gpu_unrecoverable(reason)
                                                    logger.error(f"Marked GPU as unrecoverable due to error: {reason}")
                                            except Exception:
                                                # Best-effort detection; ignore secondary errors
                                                pass
                                            execution_result["error"] = error_data
                                            raise Exception(error_msg)
                                        
                                        elif message_type == "executed":
                                            node = data.get("data", {}).get("node")
                                            output = data.get("data", {}).get("output")
                                            logger.info(f"Node {node} executed successfully")
                                            logger.debug(f"Node output: {json.dumps(output, indent=2)[:500]}...")
                                    
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse WebSocket message: {e}")
                                    logger.debug(f"Raw message: {msg.data}")
                        
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket error: {ws.exception()}")
                                raise Exception(f"WebSocket error: {ws.exception()}")
                            
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.warning("WebSocket connection closed")
                                break
                            
                        except asyncio.TimeoutError:
                            no_message_retry_count += 1
                            elapsed = asyncio.get_event_loop().time() - start_time
                            
                            # If we haven't received any messages, try to check job status before giving up
                            if last_message_time == start_time:
                                logger.warning(f"No WebSocket messages received for {comfyui_job_id} "
                                            f"(attempt {no_message_retry_count}/{max_no_message_retries}) "
                                            f"after {elapsed:.1f}s - checking job status")
                                
                                # Check if the job is complete/cached
                                try:
                                    if await self.check_if_cached(comfyui_job_id):
                                        logger.info(f"Job {comfyui_job_id} is complete (cached)")
                                        execution_result["completed"] = True
                                        execution_result["cached"] = True
                                        return execution_result
                                except Exception as check_error:
                                    logger.warning(f"Error checking job status: {check_error}")
                                
                                # If we've exhausted retries, give up
                                if no_message_retry_count >= max_no_message_retries:
                                    logger.error(f"No WebSocket messages received for {comfyui_job_id} "
                                            f"after {max_no_message_retries} attempts and {elapsed:.1f}s")
                                    raise Exception(f"No WebSocket messages received for job {comfyui_job_id} "
                                                f"after {max_no_message_retries} retry attempts")
                                
                                # Wait a bit before retrying (exponential backoff)
                                wait_time = min(5 * (2 ** (no_message_retry_count - 1)), 30)  # Cap at 30 seconds
                                logger.info(f"Waiting {wait_time}s before retry {no_message_retry_count + 1}")
                                await asyncio.sleep(wait_time)
                                
                            else:
                                # We were receiving messages but they stopped
                                logger.warning(f"WebSocket message timeout for job {comfyui_job_id} "
                                            f"(no message for {timeout_duration}s, elapsed: {elapsed:.1f}s)")
                                
                                # Try to check job status before giving up completely
                                try:
                                    if await self.check_if_cached(comfyui_job_id):
                                        logger.info(f"Job {comfyui_job_id} completed despite message timeout")
                                        execution_result["completed"] = True
                                        return execution_result
                                except Exception as check_error:
                                    logger.warning(f"Error checking job status after timeout: {check_error}")
                                
                                # If still no completion after timeout, raise error
                                raise Exception(f"WebSocket message timeout for job {comfyui_job_id} "
                                            f"after {timeout_duration} seconds without messages")
                        
                        # Check for overall timeout
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed > self.max_wait_time:
                            raise Exception(f"Timeout waiting for job {comfyui_job_id} after {elapsed:.1f} seconds")
                    
                    # If we exit the loop without completion, something went wrong
                    if not execution_result["completed"]:
                        # Final check before giving up
                        try:
                            if await self.check_if_cached(comfyui_job_id):
                                logger.info(f"Job {comfyui_job_id} completed (final check)")
                                execution_result["completed"] = True
                                return execution_result
                        except Exception as check_error:
                            logger.warning(f"Error in final job status check: {check_error}")
                        
                        raise Exception(f"WebSocket closed without completion for job {comfyui_job_id}")
                    
                    return execution_result
                    
        except asyncio.TimeoutError:
            logger.warning(f"WebSocket overall timeout for job {comfyui_job_id} - attempting to cancel")
            await self.cancel_comfyui_job(comfyui_job_id)
            raise Exception(f"WebSocket timeout for job {comfyui_job_id}")
        except aiohttp.ClientError as e:
            # Cancel the job since we can't monitor it anymore
            logger.warning(f"WebSocket connection error for job {comfyui_job_id} - attempting to cancel")
            await self.cancel_comfyui_job(comfyui_job_id)
            raise Exception(f"WebSocket connection error: {e}")
        except Exception as e:
            logger.error(f"WebSocket error for job {comfyui_job_id}: {e}")
            # Cancel on other errors to be safe
            await self.cancel_comfyui_job(comfyui_job_id)
            # Attempt detection from exception text too
            try:
                reason = _detect_cuda_unrecoverable_reason(str(e))
                if reason:
                    mark_gpu_unrecoverable(reason)
                    logger.error(f"Marked GPU as unrecoverable due to error: {reason}")
            except Exception:
                pass
            raise

    async def _update_progress(self, request_id: str, message: str):
        """Helper to update progress in the response store"""
        try:
            result = await self.response_store.get(request_id)
            if result:
                result.message = message
                await self.response_store.set(request_id, result)
        except Exception as e:
            logger.warning(f"Failed to update progress for {request_id}: {e}")

    async def get_result(self, comfyui_job_id: str) -> Optional[dict]:
        """Get the final result from ComfyUI history"""
        timeout = aiohttp.ClientTimeout(total=30)
        
        # Wait a moment for history to be updated
        await asyncio.sleep(0.5)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{COMFYUI_API_HISTORY}/{comfyui_job_id}"
                logger.debug(f"Fetching result from: {url}")
                
                async with session.get(url) as response:
                    response_text = await response.text()
                    logger.debug(f"History API status: {response.status}")
                    # Detect unrecoverable CUDA errors from history body
                    try:
                        reason = _detect_cuda_unrecoverable_reason(response_text)
                        if reason:
                            mark_gpu_unrecoverable(reason)
                            logger.error(f"Marked GPU as unrecoverable from history response: {reason}")
                    except Exception:
                        pass
                    
                    if response.status == 200:
                        history_data = json.loads(response_text)
                        
                        # Check if we got actual data
                        if not history_data or history_data == {}:
                            logger.warning(f"Empty history response for job {comfyui_job_id}")
                            # Try the general history endpoint
                            return await self._get_result_from_general_history(comfyui_job_id)
                        
                        logger.info(f"Retrieved ComfyUI history for job {comfyui_job_id}")
                        return history_data
                    else:
                        raise Exception(f"Failed to get result (status {response.status}): {response_text}")
                        
        except asyncio.TimeoutError:
            raise Exception(f"Timeout getting result for job {comfyui_job_id}")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error getting result: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in result: {e}")

    async def _get_result_from_general_history(self, comfyui_job_id: str) -> Optional[dict]:
        """Fallback: Get result from general history endpoint"""
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try the general history endpoint
                url = COMFYUI_API_HISTORY.rstrip(f"/{comfyui_job_id}")
                logger.debug(f"Trying general history endpoint: {url}")
                
                async with session.get(url) as response:
                    if response.status == 200:
                        all_history = await response.json()
                        
                        # Look for our job in the history
                        if comfyui_job_id in all_history:
                            logger.info(f"Found job {comfyui_job_id} in general history")
                            job_blob = {comfyui_job_id: all_history[comfyui_job_id]}
                            # Detect unrecoverable CUDA errors from general history
                            try:
                                reason = _detect_cuda_unrecoverable_reason(json.dumps(job_blob))
                                if reason:
                                    mark_gpu_unrecoverable(reason)
                                    logger.error(f"Marked GPU as unrecoverable from general history: {reason}")
                            except Exception:
                                pass
                            return job_blob
                        else:
                            logger.warning(f"Job {comfyui_job_id} not found in general history")
                            return {}
                    else:
                        return {}
                        
        except Exception as e:
            logger.error(f"Failed to get result from general history: {e}")
            return {}

    async def _check_if_cancelled(self, request_id: str) -> bool:
        """Check if the job has been cancelled"""
        try:
            result = await self.response_store.get(request_id)
            return result and getattr(result, 'status', '') == 'cancelled'
        except Exception as e:
            logger.warning(f"Error checking cancellation status for {request_id}: {e}")
            return False

    async def cancel_comfyui_job(self, comfyui_job_id: str):
        """Cancel a running job in ComfyUI"""
        try:       
            if not COMFYUI_API_INTERRUPT:
                logger.warning("COMFYUI_API_INTERRUPT not configured, cannot cancel job")
                return False
                
            payload = {
                "prompt_id": comfyui_job_id
            }
            
            headers = {
                'Content-Type': 'application/json'
            }
                
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                cancel_url = COMFYUI_API_INTERRUPT
                
                async with session.post(
                    cancel_url,
                    data=json.dumps(payload),
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        logger.info(f"Successfully cancelled ComfyUI job {comfyui_job_id}")
                        return True
                    else:
                        response_text = await response.text()
                        logger.warning(f"Failed to cancel ComfyUI job {comfyui_job_id}: HTTP {response.status} - {response_text}")
                        return False
                    
        except Exception as e:
            logger.error(f"Error cancelling ComfyUI job {comfyui_job_id}: {e}")
            return False
def _detect_cuda_unrecoverable_reason(text: str) -> str:
    """Detects CUDA launch failure or similar GPU fatal errors from text.

    Returns a non-empty reason string if an unrecoverable GPU issue is detected; otherwise empty string.
    """
    try:
        lowered = text.lower()
        # Common strings to detect unrecoverable GPU failures
        triggers = [
            "unspecified launch failure",
            "cuda error: unspecified launch failure",
            "cuda error",
            "cuda error: device-side assert triggered",
            "illegal memory access",
            "cuda error: an illegal memory access was encountered",
            "triton error [cuda]",
            "cuda runtime error",
            "triton error [cuda]: unspecified launch failure",
        ]
        for phrase in triggers:
            if phrase in lowered:
                return phrase
        return ""
    except Exception:
        return ""

