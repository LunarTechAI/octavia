"""
Database utilities with retry logic and circuit breaker pattern.

This module provides robust Supabase operations with:
- Exponential backoff retry logic
- Circuit breaker pattern for preventing cascade failures
- Connection health monitoring
"""
import asyncio
import functools
import random
import time
import logging
from typing import Callable, Any, TypeVar, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject all requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening circuit
    success_threshold: int = 3          # Successes needed in half-open to close
    timeout_seconds: float = 30.0       # Time to wait before trying again
    half_open_max_calls: int = 3        # Max calls allowed in half-open state


@dataclass
class CircuitBreakerState:
    """Mutable state for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    next_attempt_time: float = 0.0


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascade failures.
    
    When a service is failing, the circuit breaker "opens" and immediately
    rejects requests rather than waiting for timeouts. After a timeout period,
    it enters "half-open" state to test if the service has recovered.
    
    Usage:
        breaker = CircuitBreaker(config)
        try:
            result = breaker.call(supabase_operation)
        except CircuitBreakerOpen:
            # Service is down, use fallback
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., Any]) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: Async function to execute
            
        Returns:
            Result of the function
            
        Raises:
            CircuitBreakerOpen: If circuit is open
            Original exception: If function fails
        """
        async with self._lock:
            await self._check_state()
            
            if self.state.state == CircuitState.OPEN:
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is OPEN. Service unavailable."
                )
        
        # Execute the function without the lock
        try:
            result = await func()
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _check_state(self):
        """Check and update circuit state based on timeouts."""
        if self.state.state == CircuitState.OPEN:
            if time.time() >= self.state.next_attempt_time:
                self.state.state = CircuitState.HALF_OPEN
                self.state.success_count = 0
                self.state.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
    
    async def _on_success(self):
        """Handle successful execution."""
        async with self._lock:
            if self.state.state == CircuitState.HALF_OPEN:
                self.state.success_count += 1
                if self.state.success_count >= self.config.success_threshold:
                    self.state.state = CircuitState.CLOSED
                    self.state.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' CLOSED (recovered)")
            else:
                # Reset failure count on success in closed state
                self.state.failure_count = 0
    
    async def _on_failure(self, error: Exception):
        """Handle failed execution."""
        async with self._lock:
            self.state.failure_count += 1
            self.state.last_failure_time = time.time()
            
            if self.state.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self.state.state = CircuitState.OPEN
                self.state.next_attempt_time = (
                    time.time() + self.config.timeout_seconds
                )
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN (failure in half-open): {error}"
                )
            elif self.state.failure_count >= self.config.failure_threshold:
                # Too many failures, open the circuit
                self.state.state = CircuitState.OPEN
                self.state.next_attempt_time = (
                    time.time() + self.config.timeout_seconds
                )
                logger.warning(
                    f"Circuit breaker '{self.name}' OPEN after {self.config.failure_threshold} failures"
                )
    
    def get_state(self) -> dict:
        """Get current state for monitoring."""
        return {
            "name": self.name,
            "state": self.state.state.value,
            "failure_count": self.state.failure_count,
            "success_count": self.state.success_count,
            "next_attempt_in_seconds": max(
                0, self.state.next_attempt_time - time.time()
            )
        }


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global circuit breaker for Supabase
_supabase_breaker: Optional[CircuitBreaker] = None


def get_supabase_breaker() -> CircuitBreaker:
    """Get or create the global Supabase circuit breaker."""
    global _supabase_breaker
    if _supabase_breaker is None:
        _supabase_breaker = CircuitBreaker(
            name="supabase",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout_seconds=30.0,
                half_open_max_calls=3
            )
        )
    return _supabase_breaker


async def with_retry(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    retry_on_exceptions: Tuple[type, ...] = (Exception,),
    use_circuit_breaker: bool = True
) -> Any:
    """
    Executes an async function with exponential backoff retry logic
    and optional circuit breaker protection.
    
    Args:
        func: The async function to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        retry_on_exceptions: Tuple of exception types that trigger a retry.
        use_circuit_breaker: Whether to wrap with circuit breaker.
        
    Returns:
        Result of the function.
    """
    # Wrap with circuit breaker if enabled
    if use_circuit_breaker:
        breaker = get_supabase_breaker()
        func = lambda: breaker.call(func)
    
    retries = 0
    last_error = None
    
    while True:
        try:
            return await func()
        except CircuitBreakerOpen:
            # Circuit is open, propagate immediately
            raise
        except retry_on_exceptions as e:
            retries += 1
            last_error = e
            
            if retries > max_retries:
                logger.error(
                    f"Max retries ({max_retries}) reached for function. "
                    f"Last error: {e}"
                )
                raise
            
            # Check for specific transient error indicators
            error_str = str(e).lower()
            is_transient = any(msg in error_str for msg in [
                "502", "503", "504", "bad gateway", "service unavailable",
                "gateway timeout", "network connection lost", "connection reset",
                "timeout", "temporary", "transient"
            ])
            
            if not is_transient:
                # For non-transient errors on first attempt, retry once only
                if retries == 1:
                    logger.warning(
                        f"Non-transient error, retrying once: {e}"
                    )
                    await asyncio.sleep(base_delay)
                    continue
                else:
                    # Second failure of non-transient error - give up
                    raise
            
            # Calculate exponential backoff with jitter
            delay = min(
                base_delay * (2 ** (retries - 1)) + random.uniform(0, 0.5),
                max_delay
            )
            
            logger.info(
                f"Retry {retries}/{max_retries} for function after {delay:.2f}s "
                f"due to transient error: {e}"
            )
            await asyncio.sleep(delay)


def retry_db(
    max_retries: int = 3,
    base_delay: float = 1.0,
    use_circuit_breaker: bool = True
):
    """
    Decorator version of with_retry for easy use with database operations.
    
    Usage:
        @retry_db(max_retries=3, base_delay=1.0)
        async def get_user(user_id: str):
            return supabase.table("users").select("*").eq("id", user_id).execute()
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await with_retry(
                functools.partial(func, *args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay,
                use_circuit_breaker=use_circuit_breaker
            )
        return wrapper
    return decorator


# Health check utilities
async def check_supabase_health() -> dict:
    """
    Check if Supabase connection is healthy.
    
    Returns:
        Dict with health status and latency.
    """
    start_time = time.time()
    try:
        # Simple query to check connectivity
        from shared_dependencies import supabase
        response = supabase.table("users").select("id").limit(1).execute()
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "healthy": True,
            "latency_ms": round(latency_ms, 2),
            "error": None
        }
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "healthy": False,
            "latency_ms": round(latency_ms, 2),
            "error": str(e)
        }


def get_circuit_breaker_status() -> dict:
    """Get the current status of all circuit breakers."""
    return get_supabase_breaker().get_state()
