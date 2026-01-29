import time
import atexit
from functools import wraps

_stats = {}


def benchmark(func):
    """Decorator that tracks execution time and call count"""
    name = func.__qualname__

    if name not in _stats:
        _stats[name] = {"calls": 0, "total_time": 0.0, "min": float("inf"), "max": 0.0}

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        _stats[name]["calls"] += 1
        _stats[name]["total_time"] += elapsed
        _stats[name]["min"] = min(_stats[name]["min"], elapsed)
        _stats[name]["max"] = max(_stats[name]["max"], elapsed)

        return result

    return wrapper


def print_benchmark_stats():
    """Print all collected benchmark statistics"""
    if not _stats:
        return

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Function':<30} {'Calls':>8} {'Total':>10} {'Avg':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for name, stats in sorted(_stats.items(), key=lambda x: -x[1]["total_time"]):
        calls = stats["calls"]
        total = stats["total_time"]
        avg = total / calls if calls > 0 else 0
        min_t = stats["min"] if stats["min"] != float("inf") else 0
        max_t = stats["max"]

        print(f"{name:<30} {calls:>8} {total:>9.4f}s {avg:>9.4f}s {min_t:>9.4f}s {max_t:>9.4f}s")

    print("=" * 70)


def reset_stats():
    """Reset all benchmark statistics"""
    _stats.clear()


# Auto-print stats when program exits
atexit.register(print_benchmark_stats)
