"""``app.features`` 的离线单元测试（FR-04 / PRD §12.1）。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from app.features import AllMissingMaskError, boundary_pixel_count, extract_features
from app.masks import generate_mask


def _numpy_mask(size: int = 256) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)


def test_center_mask_matches_expected_area_and_boundary() -> None:
    mask = generate_mask("center", size=256, target_area=0.30)

    features = extract_features(None, mask)

    side = round(math.sqrt(0.30) * 256)
    missing = side * side
    boundary = 4 * side - 4

    assert missing == 19600
    assert features.area_ratio == pytest.approx(missing / (256 * 256), rel=1e-12)
    assert boundary_pixel_count(mask) == boundary
    assert features.boundary_norm == pytest.approx(boundary / math.sqrt(missing), rel=1e-12)


def test_features_do_not_depend_on_hidden_pixels() -> None:
    rng = np.random.default_rng(0)
    image_a = rng.random((1, 3, 32, 32), dtype=np.float32)
    image_b = image_a.copy()

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1  # 1 = 缺失
    hidden = mask.astype(bool)
    image_b[0][:, hidden] = rng.random((3, int(hidden.sum())), dtype=np.float32)

    assert not np.allclose(image_a, image_b)
    assert extract_features(image_a, mask) == extract_features(image_b, mask)


def test_boundary_conditions() -> None:
    # 单个缺失像素：它自身即唯一边界像素（四邻域全是已知像素）
    single = _numpy_mask(16)
    single[8, 8] = 1
    assert boundary_pixel_count(single) == 1
    assert extract_features(None, single).boundary_norm == pytest.approx(1.0)

    # 2×2 缺失块：每个像素都是边界像素
    block = _numpy_mask(16)
    block[7:9, 7:9] = 1
    assert boundary_pixel_count(block) == 4
    assert extract_features(None, block).boundary_norm == pytest.approx(4 / math.sqrt(4))

    # 横贯整行：越界邻居按缺失处理，因此上下两侧共 2 行边界像素
    row = _numpy_mask(16)
    row[8, :] = 1
    assert boundary_pixel_count(row) == 16
    assert extract_features(None, row).boundary_norm == pytest.approx(16 / math.sqrt(16))

    # 空掩码：常量特征取 0
    empty = _numpy_mask(16)
    features = extract_features(None, empty)
    assert boundary_pixel_count(empty) == 0
    assert features.area_ratio == 0.0
    assert features.boundary_norm == 0.0


def test_all_missing_mask_raises() -> None:
    mask = np.ones((16, 16), dtype=np.uint8)
    with pytest.raises(AllMissingMaskError):
        extract_features(None, mask)


def test_all_missing_mask_on_torch_tensor_raises() -> None:
    with pytest.raises(AllMissingMaskError):
        extract_features(None, torch.ones((1, 1, 16, 16)))


def test_non_binary_mask_raises() -> None:
    mask = _numpy_mask(16)
    mask[4, 4] = 7
    with pytest.raises(ValueError, match="二值"):
        extract_features(None, mask)


def test_image_mask_size_mismatch_raises() -> None:
    mask = _numpy_mask(16)
    mask[8, 8] = 1
    image = np.zeros((1, 3, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="不一致"):
        extract_features(image, mask)


def test_torch_and_numpy_masks_agree() -> None:
    mask_np = _numpy_mask(32)
    mask_np[10:20, 10:20] = 1

    from_numpy = extract_features(None, mask_np)
    from_torch = extract_features(None, torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0))

    assert from_numpy == from_torch


def test_non_2d_mask_raises() -> None:
    mask = np.zeros((3, 16, 16), dtype=np.uint8)
    mask[:, 4, 4] = 1
    with pytest.raises(ValueError, match="二维"):
        extract_features(None, mask)
