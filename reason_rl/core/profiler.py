import time
from collections import defaultdict
from contextlib import contextmanager

from ..logger import get_logger

logger = get_logger(__name__)


class Profiler:
    """
    A simple profiler to track time spent in different sections of code.
    """""

    def __init__(self):
        self.stats = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def timer(self, name: str):
        """Context manager to time a code block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stats[name] += elapsed
            self.counts[name] += 1

    def log_stats(self):
        """Log the accumulated statistics."""
        logger.info("=== Profiling Stats ===")
        total_time = sum(self.stats.values())
        if total_time == 0:
            return

        for name, elapsed in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
            count = self.counts[name]
            avg = elapsed / count if count > 0 else 0
            percent = (elapsed / total_time) * 100
            logger.info(
                f"{name:<20}: {elapsed:>8.3f}s ({percent:>5.1f}%) | Count: {count:>4} | Avg: {avg:>6.3f}s"
            )
        logger.info("=======================")

    def reset(self):
        """Reset stats."""
        self.stats.clear()
        self.counts.clear()
