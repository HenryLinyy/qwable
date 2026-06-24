"""Tests for static fit check."""

from qwable.static_fit import check_static_fit


def test_fit_ok():
    """Models should fit within M5 Max memory."""
    # fast-agent only: 24GB model + 5GB KV reserve = 29GB
    result = check_static_fit(
        model_estimates_gb=[24.0],
        parallel_count=1,
        unified_memory_gb=128,
        reserved_memory_gb=28,
        kv_cache_reserve_gb_per_parallel=5,
    )
    assert result.ok
    assert result.required_gb == 29.0
    assert result.reason is None


def test_fit_heavy():
    """Heavy-agent with ds4 should fit."""
    # fast-agent + coder + critic + judge + ds4 + formatter
    # 24 + 24 + 28 + 28 + 90 + 16 = 210GB — too much!
    # Actually heavy-agent uses only some models concurrently
    result = check_static_fit(
        model_estimates_gb=[90.0, 24.0, 28.0, 28.0],
        parallel_count=1,
        unified_memory_gb=128,
        reserved_memory_gb=28,
        kv_cache_reserve_gb_per_parallel=5,
    )
    # 90+24+28+28 + 5 = 175GB > 100GB limit
    assert not result.ok
    assert "175.0GB exceeds limit" in result.reason


def test_fit_parallel():
    """Parallel models with KV reserve."""
    # 2 parallel models: 24 + 24 + 10 (KV reserve) = 58GB
    result = check_static_fit(
        model_estimates_gb=[24.0, 24.0],
        parallel_count=2,
        unified_memory_gb=128,
        reserved_memory_gb=28,
        kv_cache_reserve_gb_per_parallel=5,
    )
    assert result.ok
    assert result.required_gb == 58.0
