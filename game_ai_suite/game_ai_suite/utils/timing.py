import time


class RateLimiter:
    """Simple rate limiter to run a loop at approximately target_hz."""

    def __init__(self, target_hz: float) -> None:
        self.period = 1.0 / max(target_hz, 1e-6)
        self._next = time.perf_counter()

    def wait(self) -> None:
        now = time.perf_counter()
        if now < self._next:
            time.sleep(self._next - now)
        self._next += self.period
