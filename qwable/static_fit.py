"""Static fit check for M5 Max memory constraints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FitCheckResult:
    ok: bool
    required_gb: float
    limit_gb: float
    reason: str | None


def check_static_fit(
    model_estimates_gb: list[float],
    parallel_count: int,
    unified_memory_gb: int = 128,
    reserved_memory_gb: int = 28,
    kv_cache_reserve_gb_per_parallel: int = 5,
) -> FitCheckResult:
    """Check if the combined models fit in M5 Max memory.

    Does NOT use psutil for dynamic degradation — purely static estimation.
    """
    total_models = sum(model_estimates_gb)
    kv_reserve = parallel_count * kv_cache_reserve_gb_per_parallel
    required_gb = total_models + kv_reserve
    limit_gb = unified_memory_gb - reserved_memory_gb

    if required_gb <= limit_gb:
        return FitCheckResult(
            ok=True,
            required_gb=required_gb,
            limit_gb=limit_gb,
            reason=None,
        )
    return FitCheckResult(
        ok=False,
        required_gb=required_gb,
        limit_gb=limit_gb,
        reason=f"Required {required_gb:.1f}GB exceeds limit {limit_gb:.1f}GB "
               f"(models={total_models:.1f}GB + KV reserve={kv_reserve:.1f}GB)",
    )
