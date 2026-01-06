import asyncio
import functools
import random
from typing import Callable, Any, TypeVar, Tuple

T = TypeVar("T")

async def with_retry(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    retry_on_exceptions: Tuple[type, ...] = (Exception,)
) -> Any:
    """
    Executes an async function with exponential backoff retry logic.
    
    Args:
        func: The async function to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        retry_on_exceptions: Tuple of exception types that trigger a retry.
    """
    retries = 0
    while True:
        try:
            return await func()
        except retry_on_exceptions as e:
            retries += 1
            if retries > max_retries:
                print(f"Max retries ({max_retries}) reached for function {func.__name__}. Error: {e}")
                raise
            
            # Check for specific transient error indicators in the error message/code
            error_str = str(e).lower()
            is_transient = any(msg in error_str for msg in [
                "502", "503", "504", "bad gateway", "service unavailable", 
                "gateway timeout", "network connection lost", "connection reset"
            ])
            
            if not is_transient and retries == 1:
                # If it's not obviously transient, we still retry once just in case,
                # but we might want to log it differently.
                print(f"Non-transient error detected, retrying once: {e}")
            elif not is_transient:
                raise
            
            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** (retries - 1)) + random.uniform(0, 1), max_delay)
            print(f"Retry {retries}/{max_retries} for {func.__name__} after {delay:.2f}s due to error: {e}")
            await asyncio.sleep(delay)

def retry_db(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator version of with_retry"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await with_retry(
                functools.partial(func, *args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay
            )
        return wrapper
    return decorator
