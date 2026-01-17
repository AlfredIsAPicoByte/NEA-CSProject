import psutil
import os
from functools import wraps

from contextlib import contextmanager
import time


def get_process_id() -> int:
    """Pure wrapper to get current PID."""
    return os.getpid()

def get_memory_mb(pid: int) -> float:
    """Returns Resident Set Size (RSS) in MB for a specific process."""
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / (1024 * 1024)
    except psutil.NoSuchProcess:
        return 0.0

def get_cpu_percent(pid: int, interval: float = 0.0) -> float:
    """Returns CPU usage. Interval 0.0 makes it non-blocking."""
    try:
        process = psutil.Process(pid)
        return process.cpu_percent(interval=interval)
    except psutil.NoSuchProcess:
        return 0.0
    
def get_current_memory_usage() -> float:
    """Returns the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    # rss = Resident Set Size (Physical RAM used)
    mem_bytes = process.memory_info().rss 
    return mem_bytes / (1024 * 1024)  # Convert to MB

def track_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pid = get_process_id()
        mem_before = get_memory_mb(pid)
        t_before = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        mem_after = get_memory_mb(pid)
        t_after = time.perf_counter()
        
        print(f"[{func.__name__}] Mem: {mem_after - mem_before:.2f} MB | Time: {t_after - t_before:.4f}s")
        return result
    return wrapper

@contextmanager
def measure_resources():
    """
    Yields a dictionary containing the resource usage DELTA 
    (difference between start and end).
    """
    pid = get_process_id()
    
    # Snapshot before
    start_mem = get_memory_mb(pid)
    start_time = time.perf_counter()
    
    stats = {"memory_delta_mb": 0.0, "time_seconds": 0.0}
    
    try:
        yield stats
    finally:
        # Snapshot after
        end_mem = get_memory_mb(pid)
        end_time = time.perf_counter()
        
        # Update the dictionary we yielded
        stats["memory_delta_mb"] = end_mem - start_mem
        stats["time_seconds"] = end_time - start_time