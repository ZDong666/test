"""``app.masks`` 的离线单元测试（FR-03 / PRD §12.1）。

覆盖要点：
- 三类掩码都是确定性生成：同一 (类型, 面积, 种子) 结果逐位一致；
- 实际面积落在 ±2pp 容差内，``scratch`` / ``scatter`` 精确到目标像素数；
- 掩码为二值 {0, 1}、形状 (1, 1, size, size)；
- 形状语义：center 连通且最紧凑；scratch 细长（归一化边界最长）；
  scatter 由多个互不相连的小块组成；
- 参数非法时报错。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage

from app.features import extract_features
from app.masks import (
    AREA_TARGETS,
    AREA_TOLERANCE,
    MASK_TYPES,
    generate_mask,
    mask_area_ratio,
)

SIZE = 256


def _binary(mask: torch.Tensor) -> np.ndarray:
    return mask[0, 0].numpy() > 0.5


@pytest.mark.parametrize("mask_type", MASK_TYPES)
@pytest.mark.parametrize("target_area", AREA_TARGETS)
def test_mask_is_deterministic(mask_type: str, target_area: float) -> None:
    first = generate_mask(mask_type, size=SIZE, target_area=target_area, seed=42)
    second = generate_mask(mask_type, size=SIZE, target_area=target_area, seed=42)
    assert torch.equal(first, second)


@pytest.mark.parametrize("mask_type", MASK_TYPES)
@pytest.mark.parametrize("target_area", AREA_TARGETS)
def test_mask_shape_dtype_and_values(mask_type: str, target_area: float) -> None:
    mask = generate_mask(mask_type, size=SIZE, target_area=target_area, seed=42)
    assert mask.shape == (1, 1, SIZE, SIZE)
    assert mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("mask_type", MASK_TYPES)
@pytest.mark.parametrize("target_area", AREA_TARGETS)
def test_actual_area_within_tolerance(mask_type: str, target_area: float) -> None:
    mask = generate_mask(mask_type, size=SIZE, target_area=target_area, seed=42)
    actual = mask_area_ratio(mask)
    assert abs(actual - target_area) <= AREA_TOLERANCE


@pytest.mark.parametrize("mask_type", ["scratch", "scatter"])
@pytest.mark.parametrize("target_area", AREA_TARGETS)
def test_random_masks_hit_exact_pixel_count(mask_type: str, target_area: float) -> None:
    mask = generate_mask(mask_type, size=SIZE, target_area=target_area, seed=42)
    expected = round(target_area * SIZE * SIZE)
    assert int(_binary(mask).sum()) == expected


def test_seed_changes_random_masks() -> None:
    base = generate_mask("scratch", size=SIZE, target_area=0.30, seed=42)
    other = generate_mask("scratch", size=SIZE, target_area=0.30, seed=43)
    assert not torch.equal(base, other)


def test_area_versions_are_not_nested() -> None:
    """不同面积使用不同随机流，15% 掩码不应是 30% 掩码的子集。"""
    small = _binary(generate_mask("scatter", size=SIZE, target_area=0.15, seed=42))
    large = _binary(generate_mask("scatter", size=SIZE, target_area=0.30, seed=42))
    assert (small & large).sum() < small.sum()
    assert not np.array_equal(small, small & large)


def test_center_mask_is_single_connected_block() -> None:
    mask = _binary(generate_mask("center", size=SIZE, target_area=0.30, seed=42))
    _labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    assert count == 1
    # 居中的正方形
    ys, xs = np.nonzero(mask)
    side = ys.max() - ys.min() + 1
    assert side == xs.max() - xs.min() + 1


def test_scatter_mask_has_many_disconnected_blocks() -> None:
    mask = _binary(generate_mask("scatter", size=SIZE, target_area=0.30, seed=42))
    _, count = ndimage.label(mask, structure=np.ones((3, 3)))
    assert count >= 10


def test_shape_semantics_via_boundary_norm() -> None:
    """中心块最紧凑（归一化边界最小），细长划痕最大。"""
    center = extract_features(None, generate_mask("center", size=SIZE, target_area=0.30, seed=42))
    scratch = extract_features(None, generate_mask("scratch", size=SIZE, target_area=0.30, seed=42))
    scatter = extract_features(None, generate_mask("scatter", size=SIZE, target_area=0.30, seed=42))

    assert center.boundary_norm < scratch.boundary_norm
    assert center.boundary_norm < scatter.boundary_norm


def test_masks_differ_across_images_through_seed() -> None:
    """逐图派生种子必须让同类型掩码在不同原图上不同（保证 boundary_norm 有区分度）。"""
    first = generate_mask("scratch", size=SIZE, target_area=0.30, seed=42)
    second = generate_mask("scratch", size=SIZE, target_area=0.30, seed=43)
    assert extract_features(None, first).boundary_norm != extract_features(None, second).boundary_norm


def test_unknown_mask_type_raises() -> None:
    with pytest.raises(ValueError, match="未知掩码类型"):
        generate_mask("blob", size=SIZE, target_area=0.30, seed=42)


@pytest.mark.parametrize("bad_area", [0.0, 1.0, -0.1, 1.5])
def test_invalid_target_area_raises(bad_area: float) -> None:
    with pytest.raises(ValueError, match="target_area"):
        generate_mask("center", size=SIZE, target_area=bad_area, seed=42)
