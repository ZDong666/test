"""``app.budget`` 的离线单元测试（FR-05 / PRD §12.1）。"""

from __future__ import annotations

import numpy as np
import pytest

from app.budget import (
    CANDIDATE_WEIGHTS_ORDER,
    DEFAULT_TIERS,
    FALLBACK_TIERS,
    BudgetPolicy,
    BudgetTiers,
    FeatureNormalizer,
    allocate_batch,
    allocate_budget,
    fit_thresholds,
    level_for_k,
    make_policies,
    mean_k,
    policy_score,
    select_weights,
)
from app.features import MaskFeatures


def _features(index: int, count: int) -> list[MaskFeatures]:
    return [
        MaskFeatures(area_ratio=0.05 + 0.5 * i / max(count - 1, 1), boundary_norm=0.0)
        for i in range(count)
    ]


def test_default_and_fallback_tiers() -> None:
    assert (DEFAULT_TIERS.low, DEFAULT_TIERS.middle, DEFAULT_TIERS.high) == (2, 5, 8)
    assert (FALLBACK_TIERS.low, FALLBACK_TIERS.middle, FALLBACK_TIERS.high) == (1, 3, 5)
    assert DEFAULT_TIERS.low + DEFAULT_TIERS.high == 2 * DEFAULT_TIERS.middle
    assert FALLBACK_TIERS.low + FALLBACK_TIERS.high == 2 * FALLBACK_TIERS.middle


def test_invalid_tiers_raise() -> None:
    with pytest.raises(ValueError, match="递增"):
        BudgetTiers(low=5, middle=3, high=2)
    with pytest.raises(ValueError, match="等预算"):
        BudgetTiers(low=1, middle=5, high=8)


def test_level_for_k_roundtrip() -> None:
    for level in ("low", "middle", "high"):
        assert level_for_k(DEFAULT_TIERS, DEFAULT_TIERS.k(level)) == level
    with pytest.raises(ValueError):
        level_for_k(DEFAULT_TIERS, 4)


def test_fixed_policies_assign_single_level() -> None:
    policies = make_policies()
    ids = [f"s{i:02d}" for i in range(9)]
    features = _features(0, 9)

    for name, expected_k in (("B1", 2), ("B2", 5), ("B3", 8)):
        assignments = allocate_batch(ids, features, policies[name])
        assert {item.k for item in assignments} == {expected_k}
        assert mean_k(assignments) == float(expected_k)


def test_batch_counts_and_mean_budget() -> None:
    policies = make_policies()
    count = 12
    ids = [f"s{i:02d}" for i in range(count)]
    features = _features(0, count)

    for name in ("M1", "A1", "C1"):
        assignments = allocate_batch(ids, features, policies[name])
        levels = [item.level for item in assignments]
        assert levels.count("low") == count // 3
        assert levels.count("middle") == count // 3
        assert levels.count("high") == count // 3
        assert mean_k(assignments) == pytest.approx(float(DEFAULT_TIERS.middle))


def test_batch_ordering_follows_score() -> None:
    policy = BudgetPolicy("A1", "score", weights=(1.0, 0.0))
    count = 12
    ids = [f"s{i:02d}" for i in range(count)]
    features = _features(0, count)

    assignments = allocate_batch(ids, features, policy)
    by_id = {item.sample_id: item for item in assignments}

    assert by_id["s00"].level == "low"
    assert by_id["s11"].level == "high"
    assert by_id["s05"].level == "middle"
    assert by_id["s00"].score < by_id["s11"].score


def test_ties_are_broken_by_sample_id() -> None:
    policy = BudgetPolicy("M1", "score", weights=(1.0, 1.0))
    count = 12
    ids = [f"s{i:02d}" for i in range(count)]
    features = [MaskFeatures(area_ratio=0.3, boundary_norm=3.0) for _ in range(count)]

    assignments = allocate_batch(ids, features, policy)
    by_id = {item.sample_id: item for item in assignments}

    for index in range(4):
        assert by_id[f"s{index:02d}"].level == "low"
    for index in range(4, 8):
        assert by_id[f"s{index:02d}"].level == "middle"
    for index in range(8, 12):
        assert by_id[f"s{index:02d}"].level == "high"


def test_batch_requires_count_divisible_by_three() -> None:
    policy = BudgetPolicy("M1", "score")
    with pytest.raises(ValueError, match="能被 3 整除"):
        allocate_batch([f"s{i}" for i in range(10)], _features(0, 10), policy)
    with pytest.raises(ValueError, match="数量不一致"):
        allocate_batch(["s0", "s1"], _features(0, 3), policy)


def test_c1_is_reproducible_and_preserves_budget_multiset() -> None:
    policies = make_policies()
    count = 12
    ids = [f"s{i:02d}" for i in range(count)]
    features = _features(0, count)

    m1 = allocate_batch(ids, features, policies["M1"])
    c1_a = allocate_batch(ids, features, policies["C1"])
    c1_b = allocate_batch(ids, features, policies["C1"])

    assert [(item.sample_id, item.k) for item in c1_a] == [
        (item.sample_id, item.k) for item in c1_b
    ]
    assert sorted(item.k for item in c1_a) == sorted(item.k for item in m1)
    assert mean_k(c1_a) == pytest.approx(mean_k(m1))
    assert [item.k for item in c1_a] != [item.k for item in m1]

    other_seed = BudgetPolicy(
        "C1", "score", weights=(1.0, 1.0), shuffle_seed=7, normalizer=FeatureNormalizer()
    )
    c1_other = allocate_batch(ids, features, other_seed)
    assert [item.k for item in c1_other] != [item.k for item in c1_a]


def test_single_image_threshold_allocation() -> None:
    policy = BudgetPolicy("M1", "score", weights=(1.0, 0.0), thresholds=(-1.0, 1.0))

    assert allocate_budget(MaskFeatures(area_ratio=-2.0, boundary_norm=0.0), policy) == 2
    assert allocate_budget(MaskFeatures(area_ratio=-1.0, boundary_norm=0.0), policy) == 2
    assert allocate_budget(MaskFeatures(area_ratio=0.5, boundary_norm=0.0), policy) == 5
    assert allocate_budget(MaskFeatures(area_ratio=1.0, boundary_norm=0.0), policy) == 8
    assert allocate_budget(MaskFeatures(area_ratio=3.0, boundary_norm=0.0), policy) == 8

    fixed = make_policies()["B3"]
    assert allocate_budget(MaskFeatures(area_ratio=0.1, boundary_norm=0.1), fixed) == 8


def test_policy_score_uses_normalizer() -> None:
    normalizer = FeatureNormalizer(area_mean=0.3, area_std=0.1, boundary_mean=3.0, boundary_std=1.0)
    policy = BudgetPolicy("M1", "score", weights=(1.0, -1.0), normalizer=normalizer)

    score = policy_score(MaskFeatures(area_ratio=0.4, boundary_norm=4.0), policy)

    assert score == pytest.approx(1.0 - 1.0)


def test_normalizer_handles_zero_std() -> None:
    features = [MaskFeatures(area_ratio=0.3, boundary_norm=3.0) for _ in range(4)]
    normalizer = FeatureNormalizer.fit(features)

    assert normalizer.area_std == 1.0
    assert normalizer.boundary_std == 1.0
    assert normalizer.transform(features[0]) == (0.0, 0.0)

    with pytest.raises(ValueError):
        FeatureNormalizer.fit([])


def test_fit_thresholds_matches_quantiles() -> None:
    scores = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = fit_thresholds(scores)
    assert low == pytest.approx(float(np.quantile(scores, 1 / 3)))
    assert high == pytest.approx(float(np.quantile(scores, 2 / 3)))
    assert low < high

    with pytest.raises(ValueError):
        fit_thresholds([])


def test_select_weights_prefers_lowest_lpips_with_stable_tiebreak() -> None:
    lpips = {(1.0, 1.0): 0.50, (1.0, -1.0): 0.40, (-1.0, 1.0): 0.40}
    assert select_weights(lpips) == (1.0, -1.0)

    only_area = {(1.0, 0.0): 0.30, (-1.0, 0.0): 0.60}
    assert select_weights(only_area, candidates=CANDIDATE_WEIGHTS_ORDER) == (1.0, 0.0)

    with pytest.raises(ValueError):
        select_weights({})


def test_make_policies_contract() -> None:
    policies = make_policies()

    assert set(policies) == {"B1", "B2", "B3", "M1", "A1", "C1"}
    assert policies["A1"].weights == (1.0, 0.0)
    assert policies["M1"].weights == (1.0, 1.0)
    assert policies["C1"].weights == policies["M1"].weights
    assert policies["C1"].shuffle_seed == 42
    assert policies["B2"].kind == "fixed"


def test_batch_allocation_is_deterministic() -> None:
    policy = make_policies()["M1"]
    ids = [f"s{i:02d}" for i in range(9)]
    features = _features(0, 9)

    first = allocate_batch(ids, features, policy)
    second = allocate_batch(ids, features, policy)

    assert first == second
