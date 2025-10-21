_consecutive_failures: int = 0


def record_generation_outcome(success: bool) -> None:
    """Record outcome of a generation attempt (success or failure)."""
    global _consecutive_failures
    if success:
        _consecutive_failures = 0
    else:
        _consecutive_failures += 1


def get_consecutive_failures() -> int:
    return _consecutive_failures


