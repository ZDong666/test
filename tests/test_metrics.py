"""``app.metrics`` 的离线单元测试（FR-08 / PRD §12.1）。

覆盖要点：
- 缺失区 PSNR 的精确值与「缺失区完全一致 → inf」；
- 已知区域变化不影响缺失区 PSNR（证明只统计缺失像素）；
- 全图 LPIPS 不会被「把已知区置零」污染（已知区变化必须改变 LPIPS）；
- 边界带宽度与 MAE；
- 同一输入重复评价结果一致；
- 评价失败时转为 ``status="error"`` 而不抛出。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.metrics import (
    BOUNDARY_BAND_PX,
    boundary_band,
    boundary_mae_3px,
    evaluate,
    full_lpips,
    masked_psnr,
)

SIZE = 16


def _reference(size: int = SIZE) -> torch.Tensor:
    """全 -1（unit 0.0）的参考图。"""
    return torch.full((1, 3, size, size), -1.0)


def _mask_with(missing: np.ndarray, size: int = SIZE) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[missing] = 1
    return mask


# --- 缺失区 PSNR ---------------------------------------------------------------


def test_masked_psnr_exact_value() -> None:
    reference = _reference()
    result = reference.clone()
    # 单像素：unit 参考 0.0 -> 结果 0.25（差值 0.25）
    result[0, :, 3, 4] = -0.5
    mask = _mask_with(np.s_[3, 4])

    # mse = 0.25^2 = 0.0625 -> -10 * log10(0.0625) = 12.0411998...
    assert masked_psnr(result, reference, mask) == pytest.approx(-10.0 * np.log10(0.0625), rel=1e-6)


def test_masked_psnr_is_infinite_when_missing_region_matches() -> None:
    reference = _reference()
    result = reference.clone()
    mask = _mask_with(np.s_[4:8, 4:8])
    assert masked_psnr(result, reference, mask) == float("inf")


def test_masked_psnr_ignores_known_region() -> None:
    reference = _reference()
    # 只改动已知区域；缺失区保持与参考一致
    result = reference.clone()
    known = np.ones((SIZE, SIZE), dtype=bool)
    known[4:8, 4:8] = False
    result[0][:, known] = 1.0
    mask = _mask_with(np.s_[4:8, 4:8])

    assert not torch.allclose(result, reference)
    assert masked_psnr(result, reference, mask) == float("inf")


def test_masked_psnr_requires_missing_pixels() -> None:
    empty = np.zeros((SIZE, SIZE), dtype=np.uint8)
    with pytest.raises(ValueError, match="空掩码"):
        masked_psnr(_reference(), _reference(), empty)


# --- 边界带 -------------------------------------------------------------------


def test_boundary_band_widths_on_square_block() -> None:
    size = 16
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[4:12, 4:12] = 1  # 8x8 缺失块

    # 切比雪夫距离 ≤ w 的缺失像素：8x8 块累计 28 / 48 / 60
    assert int(boundary_band(mask, width=1).sum()) == 28
    assert int(boundary_band(mask, width=2).sum()) == 48
    assert int(boundary_band(mask, width=3).sum()) == 60
    # 默认宽度即 3 像素
    assert int(boundary_band(mask).sum()) == 60
    assert BOUNDARY_BAND_PX == 3


def test_boundary_band_excludes_known_pixels() -> None:
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[6:10, 6:10] = 1
    band = boundary_band(mask)
    assert not (band & ~mask.astype(bool)).any()


def test_boundary_mae_value() -> None:
    reference = _reference()
    result = reference.clone()
    result += 0.4  # unit 0.2 均匀偏移
    mask = _mask_with(np.s_[4:12, 4:12])

    assert boundary_mae_3px(result, reference, mask) == pytest.approx(0.2, abs=1e-6)


def test_boundary_mae_requires_band() -> None:
    all_missing = np.ones((SIZE, SIZE), dtype=np.uint8)
    with pytest.raises(ValueError, match="边界带"):
        boundary_mae_3px(_reference(), _reference(), all_missing)


# --- 全图 LPIPS ---------------------------------------------------------------


def test_full_lpips_zero_for_identical_images() -> None:
    reference = torch.full((1, 3, 64, 64), -1.0)
    assert full_lpips(reference.clone(), reference) == pytest.approx(0.0, abs=1e-6)


def test_full_lpips_reflects_known_region_changes() -> None:
    """证明全图 LPIPS 不使用被置零的掩码图，而是直接比较完整图像。"""
    reference = torch.full((1, 3, 64, 64), -1.0)
    result = reference.clone()
    result[:, :, :32, :] = 1.0  # 只改动一半（若被置零则差异会消失）

    value = full_lpips(result, reference)
    assert value > 0.0
    assert value != float("inf")


# --- evaluate 汇总与容错 -------------------------------------------------------


def test_evaluate_ok_and_repeatable() -> None:
    size = 64  # LPIPS(AlexNet) 需要较大空间尺寸
    reference = _reference(size)
    result = reference.clone()
    result[0, :, 16:32, 16:32] = 0.2
    mask = _mask_with(np.s_[16:32, 16:32], size)

    first = evaluate(result, reference, mask)
    second = evaluate(result, reference, mask)

    assert first.status == "ok"
    assert first.error is None
    assert first == second
    assert np.isfinite(first.masked_psnr)
    assert np.isfinite(first.full_lpips)
    assert np.isfinite(first.boundary_mae_3px)


def test_evaluate_returns_error_status_without_raising() -> None:
    empty = np.zeros((SIZE, SIZE), dtype=np.uint8)

    metrics = evaluate(_reference(), _reference(), empty)

    assert metrics.status == "error"
    assert metrics.error is not None and "空掩码" in metrics.error
    assert np.isnan(metrics.masked_psnr)
    assert np.isnan(metrics.full_lpips)
    assert np.isnan(metrics.boundary_mae_3px)
