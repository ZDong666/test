"""掩码特征提取（FR-04）。

特征只依赖掩码本身，**不得读取缺失区域的真实像素**——``image`` 参数仅用于尺寸校验，
不参与任何数值计算（便于满足「修改隐藏区域内容后特征完全不变」的验收条件）。

约定：掩码中 **1 表示缺失**（PRD §5.2），取值必须为二值 {0, 1}。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

MISSING = 1


class AllMissingMaskError(ValueError):
    """全缺失掩码（area_ratio = 1.0）超出第一版预算实验范围（FR-04）。"""


@dataclass(frozen=True)
class MaskFeatures:
    """单张掩码的特征。"""

    area_ratio: float
    boundary_norm: float


def _as_bool_mask(mask: Any) -> np.ndarray:
    """把掩码统一为二维 bool 数组（True = 缺失），并校验二值性。"""
    raw = mask
    if hasattr(mask, "detach"):  # torch.Tensor 等
        raw = mask.detach().cpu().numpy()
    arr = np.asarray(raw)
    squeezed = np.squeeze(arr)
    if squeezed.ndim != 2:
        raise ValueError(f"掩码应可压缩为二维 (H, W)，实际形状为 {arr.shape}")
    if squeezed.size == 0:
        raise ValueError("掩码为空")
    if not np.all((squeezed == 0) | (squeezed == 1)):
        raise ValueError("掩码必须为二值 {0, 1}（1 = 缺失）")
    return squeezed.astype(bool)


def boundary_pixel_count(mask: Any) -> int:
    """四邻域边界像素数：缺失像素中至少有一个非缺失邻居的像素数量。

    越界邻居视为「缺失」，因此画布边缘不会被误判为边界。
    """
    m = _as_bool_mask(mask)
    if not m.any():
        return 0
    padded = np.pad(m, 1, constant_values=True)
    neighbour_non_missing = (
        ~padded[:-2, 1:-1]  # 上
        | ~padded[2:, 1:-1]  # 下
        | ~padded[1:-1, :-2]  # 左
        | ~padded[1:-1, 2:]  # 右
    )
    return int(np.count_nonzero(m & neighbour_non_missing))


def extract_features(image: Any, mask: Any) -> MaskFeatures:
    """计算 ``area_ratio`` 与归一化边界长度 ``boundary_norm``。

    - ``area_ratio = 缺失像素数 / 总像素数``
    - ``boundary_norm = 边界像素数 / sqrt(缺失像素数)``（无缺失时为 0）
    - 全缺失掩码抛 :class:`AllMissingMaskError`
    """
    m = _as_bool_mask(mask)

    if image is not None:
        raw = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
        img_shape = np.asarray(raw).shape
        if len(img_shape) >= 2 and tuple(img_shape[-2:]) != m.shape:
            raise ValueError(f"图片空间尺寸 {tuple(img_shape[-2:])} 与掩码 {m.shape} 不一致")

    missing = int(np.count_nonzero(m))
    total = int(m.size)
    if missing == total:
        raise AllMissingMaskError("全缺失掩码（area_ratio = 1.0）不进入第一版预算实验")

    area_ratio = missing / total
    boundary_norm = boundary_pixel_count(m) / math.sqrt(missing) if missing > 0 else 0.0
    return MaskFeatures(area_ratio=area_ratio, boundary_norm=boundary_norm)
