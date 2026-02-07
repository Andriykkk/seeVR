import time
import atexit
from functools import wraps

_stats = {}
_enabled = True


def enable_benchmark(enabled=True):
    """Enable or disable benchmarking globally"""
    global _enabled
    _enabled = enabled


def is_enabled_benchmark():
    """Check if benchmarking is enabled"""
    return _enabled


def benchmark(func):
    """Decorator that tracks execution time and call count (only when enabled)"""
    name = func.__qualname__

    if name not in _stats:
        _stats[name] = {"calls": 0, "total_time": 0.0, "times": []}

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _enabled:
            return func(*args, **kwargs)

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        _stats[name]["calls"] += 1
        _stats[name]["total_time"] += elapsed
        _stats[name]["times"].append(elapsed)

        return result

    return wrapper


def percentile(times, p):
    """Calculate the p-th percentile of a sorted list"""
    if not times:
        return 0
    k = (len(times) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(times) else f
    return times[f] + (times[c] - times[f]) * (k - f)


def print_benchmark_stats():
    """Print all collected benchmark statistics"""
    if not _enabled or not _stats:
        return

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    print(f"{'Function':<30} {'Calls':>8} {'Total':>9} {'Avg':>9} {'P50':>9} {'P95':>9} {'P99':>9} {'Min':>9} {'Max':>9}")
    print("-" * 100)

    for name, stats in sorted(_stats.items(), key=lambda x: -x[1]["total_time"]):
        if stats["calls"] == 0:
            continue
        calls = stats["calls"]
        total = stats["total_time"]
        times = sorted(stats["times"])

        avg = total / calls if calls > 0 else 0
        min_t = times[0] if times else 0
        max_t = times[-1] if times else 0
        p50 = percentile(times, 50)
        p95 = percentile(times, 95)
        p99 = percentile(times, 99)

        print(f"{name:<30} {calls:>8} {total:>8.4f}s {avg:>8.5f}s {p50:>8.5f}s {p95:>8.5f}s {p99:>8.5f}s {min_t:>8.5f}s {max_t:>8.5f}s")

    print("=" * 100)


def reset_stats():
    """Reset all benchmark statistics"""
    _stats.clear()


# Auto-print stats when program exits
atexit.register(print_benchmark_stats)
