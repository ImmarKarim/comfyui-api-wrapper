_consecutive_failures: int = 0
_gpu_unrecoverable_error: bool = False
_gpu_unrecoverable_reason: str = ""


def record_generation_outcome(success: bool) -> None:
    """Record outcome of a generation attempt (success or failure)."""
    global _consecutive_failures
    if success:
        _consecutive_failures = 0
    else:
        _consecutive_failures += 1


def get_consecutive_failures() -> int:
    return _consecutive_failures



def mark_gpu_unrecoverable(reason: str) -> None:
    """Mark the GPU as unrecoverable for this process lifetime.

    This should be called when we detect CUDA launch failures or similar
    fatal GPU runtime errors indicating the worker can no longer serve requests.
    """
    global _gpu_unrecoverable_error, _gpu_unrecoverable_reason
    _gpu_unrecoverable_error = True
    _gpu_unrecoverable_reason = reason


def clear_gpu_unrecoverable() -> None:
    """Clear the unrecoverable GPU flag (use with caution)."""
    global _gpu_unrecoverable_error, _gpu_unrecoverable_reason
    _gpu_unrecoverable_error = False
    _gpu_unrecoverable_reason = ""


def is_gpu_unrecoverable() -> bool:
    return _gpu_unrecoverable_error


def get_gpu_unrecoverable_reason() -> str:
    return _gpu_unrecoverable_reason

