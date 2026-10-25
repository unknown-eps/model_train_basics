"""Track successful completions using elapsed wall-clock time."""

import time


class ThroughputMeter:
    """Create before starting workers; call record from the event-loop thread."""

    def __init__(self, report_every: int = 1000):
        if report_every <= 0:
            raise ValueError("report_every must be positive")

        self.report_every = report_every
        self.start_time = time.perf_counter()
        self.previous_report_time = self.start_time
        self.total_requests = 0
        self.total_tokens = 0
        self.interval_requests = 0
        self.interval_tokens = 0

    def record(self, num_tokens: int) -> None:
        """Record one successful response and report at the configured interval."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be nonnegative")

        self.total_requests += 1
        self.total_tokens += num_tokens
        self.interval_requests += 1
        self.interval_tokens += num_tokens

        if self.interval_requests >= self.report_every:
            self._report(time.perf_counter())

    def finish(self) -> None:
        """Report any remaining completions; call after processing all results."""
        if self.interval_requests:
            self._report(time.perf_counter())

    def _report(self, now: float) -> None:
        elapsed = now - self.previous_report_time
        total_elapsed = now - self.start_time
        print(
            f"Completed: {self.total_requests} | "
            f"Last {self.interval_requests}: "
            f"{self.interval_requests / elapsed:.2f} requests/s | "
            f"{self.interval_tokens / elapsed:.2f} tokens/s | "
            f"Overall: {self.total_requests / total_elapsed:.2f} requests/s | "
            f"{self.total_tokens / total_elapsed:.2f} tokens/s",
            flush=True,
        )
        self.previous_report_time = now
        self.interval_requests = 0
        self.interval_tokens = 0
