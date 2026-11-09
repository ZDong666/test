"""预算分配（FR-05）。

包含四类策略与批量等预算分档：

- ``B1`` / ``B2`` / ``B3``：固定低 / 中 / 高预算；
- ``M1``：面积 + 边界规则（权重由开发集冻结）；
- ``A1``：仅面积规则；
- ``C1``：把 M1 的预算集合按固定种子随机打乱。

等预算约束（PRD §7.6 / FR-05）：批量 N 个样本按分数升序排序，最低 ``floor(N/3)`` 个
分配低预算、最高 ``floor(N/3)`` 个分配高预算、其余中预算；同分按固定样本 ID 排序。
因此要求 ``N % 3 == 0`` 且档位满足 ``low + high == 2 * middle``，此时总体平均 K 恒等于中预算。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from app.features import MaskFeatures

LEVEL_LOW = "low"
LEVEL_MIDDLE = "middle"
LEVEL_HIGH = "high"
LEVELS = (LEVEL_LOW, LEVEL_MIDDLE, LEVEL_HIGH)

#: 面积 + 边界候选权重（PRD §7.4，按优先级顺序）
CANDIDATE_WEIGHTS_AREA_BOUNDARY = (
    (1.0, 1.0),
    (1.0, -1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
)
#: 面积规则候选权重
CANDIDATE_WEIGHTS_AREA = ((1.0, 0.0), (-1.0, 0.0))
#: 全部候选，按 PRD §7.4 的列举顺序（同分时靠前者优先）
CANDIDATE_WEIGHTS_ORDER = CANDIDATE_WEIGHTS_AREA_BOUNDARY + CANDIDATE_WEIGHTS_AREA


@dataclass(frozen=True)
class BudgetTiers:
    """预算档位（PRD §7.3）。"""

    low: int = 2
    middle: int = 5
    high: int = 8

    def __post_init__(self) -> None:
        if not (self.low < self.middle < self.high):
            raise ValueError(f"预算档位必须严格递增，收到 {self.low}/{self.middle}/{self.high}")
        if self.low + self.high != 2 * self.middle:
            raise ValueError(
                f"等预算要求 low + high == 2 * middle，收到 {self.low}/{self.middle}/{self.high}"
            )

    def k(self, level: str) -> int:
        if level == LEVEL_LOW:
            return self.low
        if level == LEVEL_MIDDLE:
            return self.middle
        if level == LEVEL_HIGH:
            return self.high
        raise ValueError(f"未知预算档位：{level!r}")


#: 主档位 K=2/5/8（PRD §7.3 优先方案）
DEFAULT_TIERS = BudgetTiers()
#: 降级档位 K=1/3/5（D2 超预算时启用）
FALLBACK_TIERS = BudgetTiers(low=1, middle=3, high=5)


@dataclass(frozen=True)
class FeatureNormalizer:
    """z-score 归一化参数，只在开发集拟合、测试集只应用（PRD §7.4）。"""

    area_mean: float = 0.0
    area_std: float = 1.0
    boundary_mean: float = 0.0
    boundary_std: float = 1.0

    @classmethod
    def fit(cls, features: Sequence[MaskFeatures]) -> FeatureNormalizer:
        if not features:
            raise ValueError("拟合归一化参数需要至少一个样本")
        areas = np.array([f.area_ratio for f in features], dtype=np.float64)
        boundaries = np.array([f.boundary_norm for f in features], dtype=np.float64)

        def _safe_std(values: np.ndarray) -> float:
            std = float(values.std())
            return std if std > 0.0 else 1.0

        return cls(
            area_mean=float(areas.mean()),
            area_std=_safe_std(areas),
            boundary_mean=float(boundaries.mean()),
            boundary_std=_safe_std(boundaries),
        )

    def transform(self, features: MaskFeatures) -> tuple[float, float]:
        area = (features.area_ratio - self.area_mean) / self.area_std
        boundary = (features.boundary_norm - self.boundary_mean) / self.boundary_std
        return float(area), float(boundary)


@dataclass(frozen=True)
class BudgetPolicy:
    """一条预算策略。

    ``kind="fixed"`` 只用 ``level``；``kind="score"`` 用 ``weights`` 计算分数，
    单图按 ``thresholds`` 阈值决策，批量按排序分档；``shuffle_seed`` 非空时（C1）
    在分档后按该种子打乱预算集合。
    """

    name: str
    kind: str
    tiers: BudgetTiers = DEFAULT_TIERS
    weights: tuple[float, float] = (1.0, 1.0)
    level: str = LEVEL_MIDDLE
    thresholds: tuple[float, float] = (-1.0, 1.0)
    normalizer: FeatureNormalizer = field(default_factory=FeatureNormalizer)
    shuffle_seed: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("fixed", "score"):
            raise ValueError(f"未知策略类型：{self.kind!r}")
        if self.kind == "fixed" and self.level not in LEVELS:
            raise ValueError(f"未知预算档位：{self.level!r}")
        if self.kind == "score" and self.thresholds[0] > self.thresholds[1]:
            raise ValueError(f"阈值需满足 low <= high，收到 {self.thresholds}")


@dataclass(frozen=True)
class BudgetAssignment:
    """批量分档结果（一条样本记录）。"""

    sample_id: str
    score: float
    level: str
    k: int


def level_for_k(tiers: BudgetTiers, k: int) -> str:
    """按档位把 K 映射回档位名。"""
    for level in LEVELS:
        if tiers.k(level) == k:
            return level
    raise ValueError(f"K={k} 不属于档位 {tiers.low}/{tiers.middle}/{tiers.high}")


def policy_score(features: MaskFeatures, policy: BudgetPolicy) -> float:
    """``w_area * z(area) + w_boundary * z(boundary)``，分数越高预算越高。"""
    area_z, boundary_z = policy.normalizer.transform(features)
    weight_area, weight_boundary = policy.weights
    return weight_area * area_z + weight_boundary * boundary_z


def allocate_budget(features: MaskFeatures, policy: BudgetPolicy) -> int:
    """单图预算决策（PRD §8.1 ``allocate_budget``）。

    固定策略直接返回对应档位；分数策略按冻结阈值切分低 / 中 / 高。
    """
    tiers = policy.tiers
    if policy.kind == "fixed":
        return tiers.k(policy.level)
    score = policy_score(features, policy)
    low_threshold, high_threshold = policy.thresholds
    if score <= low_threshold:
        return tiers.low
    if score >= high_threshold:
        return tiers.high
    return tiers.middle


def allocate_batch(
    sample_ids: Sequence[str],
    features_list: Sequence[MaskFeatures],
    policy: BudgetPolicy,
) -> list[BudgetAssignment]:
    """批量等预算分档（FR-05）。

    要求 ``len(sample_ids) % 3 == 0``；返回顺序与输入一致。
    """
    if len(sample_ids) != len(features_list):
        raise ValueError(
            f"样本 ID 与特征数量不一致：{len(sample_ids)} != {len(features_list)}"
        )
    tiers = policy.tiers
    count = len(sample_ids)

    if policy.kind == "fixed":
        level = policy.level
        k = tiers.k(level)
        return [BudgetAssignment(sid, 0.0, level, k) for sid in sample_ids]

    if count % 3 != 0:
        raise ValueError(f"批量等预算分档要求 N 能被 3 整除，实际 N={count}")

    scores = [policy_score(features, policy) for features in features_list]
    order = sorted(range(count), key=lambda i: (scores[i], sample_ids[i]))
    share = count // 3

    levels: list[str] = [LEVEL_MIDDLE] * count
    for position, index in enumerate(order):
        if position < share:
            levels[index] = LEVEL_LOW
        elif position >= count - share:
            levels[index] = LEVEL_HIGH

    ks = [tiers.k(level) for level in levels]
    if policy.shuffle_seed is not None:
        shuffled = list(ks)
        random.Random(policy.shuffle_seed).shuffle(shuffled)
        ks = shuffled
        levels = [level_for_k(tiers, k) for k in ks]

    return [
        BudgetAssignment(sample_ids[i], scores[i], levels[i], ks[i]) for i in range(count)
    ]


def mean_k(assignments: Sequence[BudgetAssignment]) -> float:
    """平均预算，用于核对等预算约束。"""
    if not assignments:
        raise ValueError("空分档结果没有平均预算")
    return float(sum(item.k for item in assignments)) / len(assignments)


def fit_thresholds(
    scores: Sequence[float], low_quantile: float = 1 / 3, high_quantile: float = 2 / 3
) -> tuple[float, float]:
    """在开发集分数上取分位点作为单图阈值（冻结后只用于应用）。"""
    if not scores:
        raise ValueError("拟合阈值需要至少一个分数")
    values = np.array(list(scores), dtype=np.float64)
    return (
        float(np.quantile(values, low_quantile)),
        float(np.quantile(values, high_quantile)),
    )


def select_weights(
    lpips_by_weights: dict[tuple[float, float], float],
    candidates: Sequence[tuple[float, float]] = CANDIDATE_WEIGHTS_ORDER,
) -> tuple[float, float]:
    """在开发集按平均全图 LPIPS 最低选权重；同分按 ``candidates`` 顺序取前者（PRD §7.4）。"""
    best: tuple[float, float] | None = None
    best_key: tuple[float, int] | None = None
    for position, weights in enumerate(candidates):
        if weights not in lpips_by_weights:
            continue
        key = (lpips_by_weights[weights], position)
        if best_key is None or key < best_key:
            best_key = key
            best = weights
    if best is None:
        raise ValueError("候选权重与 LPIPS 分数没有交集，无法选择规则")
    return best


def make_policies(
    tiers: BudgetTiers = DEFAULT_TIERS,
    normalizer: FeatureNormalizer | None = None,
    thresholds: tuple[float, float] = (-1.0, 1.0),
    m1_weights: tuple[float, float] = (1.0, 1.0),
    a1_weights: tuple[float, float] = (1.0, 0.0),
    shuffle_seed: int = 42,
) -> dict[str, BudgetPolicy]:
    """构造 B1/B2/B3/M1/A1/C1 六条策略（PRD §7.5）。"""
    resolved_normalizer = normalizer if normalizer is not None else FeatureNormalizer()
    score_kwargs = {
        "tiers": tiers,
        "kind": "score",
        "thresholds": thresholds,
        "normalizer": resolved_normalizer,
    }
    return {
        "B1": BudgetPolicy("B1", "fixed", tiers=tiers, level=LEVEL_LOW),
        "B2": BudgetPolicy("B2", "fixed", tiers=tiers, level=LEVEL_MIDDLE),
        "B3": BudgetPolicy("B3", "fixed", tiers=tiers, level=LEVEL_HIGH),
        "M1": BudgetPolicy("M1", weights=m1_weights, **score_kwargs),
        "A1": BudgetPolicy("A1", weights=a1_weights, **score_kwargs),
        "C1": BudgetPolicy(
            "C1", weights=m1_weights, shuffle_seed=shuffle_seed, **score_kwargs
        ),
    }
