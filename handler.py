import asyncio
import logging
import time
import uuid

import runpod
from aiocache import SimpleMemoryCache

from requestmodels.models import Payload
from responses.result import Result
from workers.preprocess_worker import PreprocessWorker
from workers.generation_worker import GenerationWorker
from workers.postprocess_worker import PostprocessWorker


logger = logging.getLogger(__name__)


async def handler(event: dict):
    """Runpod async handler: execute the full generation pipeline.

    Runpod's serverless SDK natively supports async handlers, so we can
    directly define this as async and avoid asyncio.run() entirely.
    
    Expects event["input"] to match the API's Payload schema. If the provided
    input is already the inner object, we wrap it accordingly.
    """
    raw_input = event.get("input", event)
    if "input" in raw_input:
        payload_dict = raw_input
    else:
        payload_dict = {"input": raw_input}

    payload = Payload(**payload_dict)

    if not payload.input.request_id:
        payload.input.request_id = str(uuid.uuid4())
    request_id = payload.input.request_id

    # Per-request stores and queues
    request_store = SimpleMemoryCache(namespace="request_store")
    response_store = SimpleMemoryCache(namespace="response_store")

    preprocess_queue = asyncio.Queue()
    generation_queue = asyncio.Queue()
    postprocess_queue = asyncio.Queue()

    # Seed stores and queues
    result_pending = Result(id=request_id)
    await request_store.set(request_id, payload)
    await response_store.set(request_id, result_pending)
    await preprocess_queue.put(request_id)

    # Spin up one worker of each type for this request
    worker_config = {
        "preprocess_queue": preprocess_queue,
        "generation_queue": generation_queue,
        "postprocess_queue": postprocess_queue,
        "request_store": request_store,
        "response_store": response_store,
    }

    preprocess_worker = PreprocessWorker(1, worker_config)
    generation_worker = GenerationWorker(1, worker_config)
    postprocess_worker = PostprocessWorker(1, worker_config)

    tasks = [
        asyncio.create_task(preprocess_worker.work()),
        asyncio.create_task(generation_worker.work()),
        asyncio.create_task(postprocess_worker.work()),
    ]

    # Wait for terminal status with optional timeout
    terminal_statuses = {"completed", "failed", "timeout", "cancelled"}
    timeout_seconds = int(event.get("timeout", 3600))
    start_time = time.time()

    try:
        while True:
            result = await response_store.get(request_id)
            if result and getattr(result, "status", "") in terminal_statuses:
                break
            if time.time() - start_time > timeout_seconds:
                if result:
                    result.status = "timeout"
                    result.message = f"Timed out after {timeout_seconds} seconds"
                    await response_store.set(request_id, result)
                break
            await asyncio.sleep(0.5)

        # Ensure all queues finish processing before stopping workers
        await preprocess_queue.join()
        await generation_queue.join()
        await postprocess_queue.join()

    finally:
        # Signal workers to exit and wait for them
        await preprocess_queue.put(None)
        await generation_queue.put(None)
        await postprocess_queue.put(None)

        await asyncio.gather(*tasks, return_exceptions=True)

    final_result = await response_store.get(request_id)
    if isinstance(final_result, Result):
        return final_result.model_dump()
    # Fallback
    return {"id": request_id, "status": "failed", "message": "No result available"}


runpod.serverless.start({"handler": handler})


