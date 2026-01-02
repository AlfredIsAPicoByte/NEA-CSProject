# MemoryProfiler.py
import tracemalloc
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from MemoryUtils import get_current_memory_usage

@dataclass
class MemoryProfileResult:
    time_seconds: float
    rss_start_mb: float
    rss_end_mb: float
    rss_delta_mb: float
    tracemalloc_current_kb: float
    tracemalloc_peak_kb: float
    top_stats: List[Dict[str, Any]]  # {trace: str, size_kb: float, count: int}

class MemoryProfiler:
    """Context manager for measuring memory + time; optionally uses tracemalloc."""
    def __init__(self, *, enable_tracemalloc: bool = True, top: int = 10):
        self.enable_tracemalloc = enable_tracemalloc
        self.top = top
        self._start_snapshot = None
        self._start_time = 0.0
        self._start_rss = 0.0
        self.result: Optional[MemoryProfileResult] = None

    def __enter__(self) -> "MemoryProfiler":
        self._start_time = time.perf_counter()
        self._start_rss = get_current_memory_usage()
        if self.enable_tracemalloc:
            tracemalloc.start()
            # Capture a baseline snapshot
            self._start_snapshot = tracemalloc.take_snapshot()
        return self

    def __exit__(self, exc_type, exc, tb):
        end_time = time.perf_counter()
        end_rss = get_current_memory_usage()
        time_seconds = end_time - self._start_time
        rss_delta = end_rss - self._start_rss

        if self.enable_tracemalloc:
            current_b, peak_b = tracemalloc.get_traced_memory()
            snapshot2 = tracemalloc.take_snapshot()
            raw_stats = snapshot2.compare_to(self._start_snapshot, 'lineno')[:self.top]
            top_stats = [
                {"trace": str(s.traceback), "size_kb": s.size / 1024.0, "count": s.count}
                for s in raw_stats
            ]
            tracemalloc.stop()
        else:
            current_b = peak_b = 0
            top_stats = []

        self.result = MemoryProfileResult(
            time_seconds=time_seconds,
            rss_start_mb=self._start_rss,
            rss_end_mb=end_rss,
            rss_delta_mb=rss_delta,
            tracemalloc_current_kb=current_b / 1024.0,
            tracemalloc_peak_kb=peak_b / 1024.0,
            top_stats=top_stats
        )

    def format_report(self, concise: bool = False) -> str:
        r = self.result
        if r is None:
            return "No profile captured."
        s = []
        s.append(f"Time: {r.time_seconds:.4f}s | RSS Δ: {r.rss_delta_mb:.3f} MB")
        if not concise:
            s.append(f"RSS start: {r.rss_start_mb:.3f} MB | RSS end: {r.rss_end_mb:.3f} MB")
            s.append(f"tracemalloc current: {r.tracemalloc_current_kb:.2f} KB | peak: {r.tracemalloc_peak_kb:.2f} KB")
            if r.top_stats:
                s.append("Top allocations (most recent frame):")
                for i, t in enumerate(r.top_stats, start=1):
                    s.append(f"  {i}. {t['trace'].splitlines()[-1]} | {t['size_kb']:.2f} KB ({t['count']} allocs)")
        return "\n".join(s)

def profile(*, enable_tracemalloc: bool = True, top: int = 6, verbose: bool = True, return_result: bool = False):
    """Decorator to profile a function call. Attaches last profile as `.last_profile` to the wrapper."""
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            with MemoryProfiler(enable_tracemalloc=enable_tracemalloc, top=top) as mp:
                result = func(*args, **kwargs)
            if verbose:
                print(mp.format_report())
            wrapper.last_profile = mp.result
            if return_result:
                return result, mp.result
            return result
        wrapper.last_profile = None
        return wrapper
    return decorator