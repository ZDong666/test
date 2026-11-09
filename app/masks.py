"""掩码生成（S1 只实现中心块 center）。

张量约定：返回 (1, 1, size, size) float32，取值 {0, 1}，其中 **1 表示缺失**。
划痕（scratch）与零散块（scatter）留待 S4。
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_SIZE = 256

S1_MASK_TYPES = ("center",)


def generate_mask(
    mask_type: str,
    size: int = MODEL_SIZE,
    target_area: float = 0.30,
    seed: int = 42,
) -> torch.Tensor:
    """生成确定性掩码。

    S1 仅支持 ``center``：目标面积由居中的正方形近似，边长 ``round(sqrt(target_area) * size)``。
    ``seed`` 仅为保持接口一致，center 不使用随机性。
    """
    if mask_type not in S1_MASK_TYPES:
        raise NotImplementedError(
            f"S1 只实现 {S1_MASK_TYPES}，收到 {mask_type!r}；划痕/零散块见 S4。"
        )
    mask = torch.zeros((1, 1, size, size), dtype=torch.float32)
    side = max(1, round(sqrt(target_area) * size))
    start = (size - side) // 2
    mask[:, :, start : start + side, start : start + side] = 1.0
    return mask


def mask_area_ratio(mask: torch.Tensor) -> float:
    """缺失像素占比。"""
    return float(mask.to(torch.float32).mean().item())


def save_mask_png(path: str | Path, mask: torch.Tensor) -> None:
    """把 {0, 1} 掩码保存为 uint8 PNG（1 -> 255）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = mask.detach().cpu()
    while m.dim() > 2:
        m = m[0]
    arr = np.clip(m.numpy() * 255.0, 0.0, 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)
