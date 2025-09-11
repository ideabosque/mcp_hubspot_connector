"""
Rate Limiter Module for MCP HubSpot Connector

This module contains the RateLimiter class responsible for managing
API rate limiting to comply with HubSpot API restrictions.
"""

import asyncio
import time
from typing import Optional
import logging


class RateLimiter:
    """Rate limiter for HubSpot API calls"""

    def __init__(self, calls_per_second: int = 10, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the rate limiter
        
        Args:
            calls_per_second: Maximum number of API calls per second
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.calls_per_second = calls_per_second
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_call_time: Optional[float] = None
        self.call_count = 0
        self.window_start = time.time()
        self.logger = logging.getLogger(__name__)

    async def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        
        # Reset window if a second has passed
        if current_time - self.window_start >= 1.0:
            self.call_count = 0
            self.window_start = current_time
        
        # If we've hit the limit, wait until the next window
        if self.call_count >= self.calls_per_second:
            wait_time = 1.0 - (current_time - self.window_start)
            if wait_time > 0:
                self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                # Reset for new window
                self.call_count = 0
                self.window_start = time.time()
        
        # Increment call count
        self.call_count += 1
        self.last_call_time = time.time()

    async def execute_with_retry(self, operation, *args, **kwargs):
        """Execute an operation with automatic retries on rate limit errors"""
        
        for attempt in range(self.max_retries + 1):
            try:
                await self.wait_if_needed()
                return await operation(*args, **kwargs)
                
            except Exception as e:
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e) and attempt < self.max_retries:
                    retry_delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds (attempt {attempt + 1})")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # Re-raise the exception if it's not a rate limit error or we've exhausted retries
                    raise

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is related to rate limiting"""
        
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit",
            "too many requests",
            "429",
            "quota exceeded",
            "throttle",
            "rate exceeded"
        ]
        
        return any(indicator in error_str for indicator in rate_limit_indicators)

    def get_current_rate(self) -> float:
        """Get the current rate of API calls per second"""
        
        if self.last_call_time is None:
            return 0.0
        
        current_time = time.time()
        window_duration = current_time - self.window_start
        
        if window_duration > 0:
            return self.call_count / window_duration
        else:
            return 0.0

    def reset(self) -> None:
        """Reset the rate limiter state"""
        
        self.call_count = 0
        self.window_start = time.time()
        self.last_call_time = None
        self.logger.debug("Rate limiter reset")

    def adjust_rate(self, new_calls_per_second: int) -> None:
        """Adjust the rate limit dynamically"""
        
        old_rate = self.calls_per_second
        self.calls_per_second = max(1, new_calls_per_second)  # Ensure minimum of 1 call per second
        self.reset()  # Reset state when changing rate
        
        self.logger.info(f"Rate limit adjusted from {old_rate} to {self.calls_per_second} calls per second")

    def get_stats(self) -> dict:
        """Get current rate limiter statistics"""
        
        current_time = time.time()
        window_duration = current_time - self.window_start
        
        return {
            "calls_per_second_limit": self.calls_per_second,
            "current_calls_in_window": self.call_count,
            "window_duration": window_duration,
            "current_rate": self.get_current_rate(),
            "last_call_time": self.last_call_time,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay
        }