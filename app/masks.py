"""掩码生成（FR-03）。

三类掩码（张量约定：返回 (1, 1, size, size) float32，取值 {0, 1}，**1 表示缺失**）：

- ``center``：居中的连续矩形块，边长 ``round(sqrt(target_area) * size)``（S1 起沿用）；
- ``scratch``：多条细长线段（长度约为画布的 1/8 ~ 1/3，厚度 ``max(2, size // 128)``）；
- ``scatter``：多个互不相连的小块（边长约为画布的 1/32 ~ 1/12，块间保留 1 像素间隔）。

确定性与面积控制：

- ``scratch`` / ``scatter`` 使用 ``np.random.default_rng([seed, 类型编码, 面积编码])``，
  因此「同一 (seed, 类型, 面积)」永远得到同一掩码；不同面积使用不同随机流，互不为子集。
- ``scratch`` / ``scatter`` 精确填充到 ``round(target_area * size * size)`` 个缺失像素；
  ``center`` 为近似正方形，实际面积与目标偏差在 ``AREA_TOLERANCE`` 容差内。
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_SIZE = 256

#: 三类掩码（PRD §4 范围 / FR-03）
MASK_TYPES = ("center", "scratch", "scatter")
#: 两种目标缺失面积（FR-03）
AREA_TARGETS = (0.15, 0.30)
#: 目标面积容差 ±2 个百分点（FR-03）
AREA_TOLERANCE = 0.02


def _area_code(target_area: float) -> int:
    return round(target_area * 100)


def _target_pixels(size: int, target_area: float) -> int:
    return round(target_area * size * size)


def _make_rng(mask_type: str, target_area: float, seed: int) -> np.random.Generator:
    """按 (种子, 类型, 面积) 派生确定性随机流。"""
    return np.random.default_rng([int(seed), MASK_TYPES.index(mask_type), _area_code(target_area)])


def _center_mask(size: int, target_area: float) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    side = max(1, round(sqrt(target_area) * size))
    start = (size - side) // 2
    mask[start : start + side, start : start + side] = True
    return mask


def _stroke_pixels(rng: np.random.Generator, size: int, thickness: int):
    """生成一条随机细线段的像素（含厚度），按沿线顺序产出。"""
    half = thickness // 2
    y0 = int(rng.integers(0, size))
    x0 = int(rng.integers(0, size))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    length = int(rng.integers(max(8, size // 8), max(9, size // 3) + 1))
    y1 = round(float(y0 + length * np.sin(angle)))
    x1 = round(float(x0 + length * np.cos(angle)))
    steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
    for step in range(steps):
        y = round(y0 + (y1 - y0) * step / steps)
        x = round(x0 + (x1 - x0) * step / steps)
        for dy in range(-half, half + 1):
            yy = y + dy
            if not 0 <= yy < size:
                continue
            for dx in range(-half, half + 1):
                xx = x + dx
                if 0 <= xx < size:
                    yield yy, xx


def _scratch_mask(size: int, target: int, seed: int, target_area: float) -> np.ndarray:
    rng = _make_rng("scratch", target_area, seed)
    thickness = max(2, size // 128)
    filled: set[tuple[int, int]] = set()
    for _ in range(4096):  # 上限保护，正常远小于此值
        if len(filled) >= target:
            break
        for pixel in _stroke_pixels(rng, size, thickness):
            filled.add(pixel)
            if len(filled) >= target:
                break
    mask = np.zeros((size, size), dtype=bool)
    for y, x in filled:
        mask[y, x] = True
    return mask


def _scatter_mask(size: int, target: int, seed: int, target_area: float) -> np.ndarray:
    rng = _make_rng("scatter", target_area, seed)
    small = max(4, size // 32)
    large = max(small + 1, size // 12)
    mask = np.zeros((size, size), dtype=bool)
    count = 0
    for _ in range(8192):  # 上限保护
        if count >= target:
            break
        height = int(rng.integers(small, large + 1))
        width = int(rng.integers(small, large + 1))
        top = int(rng.integers(0, size - height + 1))
        left = int(rng.integers(0, size - width + 1))
        # 与已有块保持至少 1 像素间隔，保证「不连通」
        padded = np.pad(mask, 1, constant_values=False)
        if padded[top : top + height + 2, left : left + width + 2].any():
            continue
        for y in range(top, top + height):
            for x in range(left, left + width):
                mask[y, x] = True
                count += 1
                if count >= target:
                    break
            if count >= target:
                break
    return mask


def generate_mask(
    mask_type: str,
    size: int = MODEL_SIZE,
    target_area: float = 0.30,
    seed: int = 42,
) -> torch.Tensor:
    """生成确定性掩码（FR-03）。

    ``scratch`` / ``scatter`` 精确填充目标像素数；``center`` 为近似正方形。
    同一 ``(mask_type, target_area, seed)`` 必然得到相同掩码。
    """
    if mask_type not in MASK_TYPES:
        raise ValueError(f"未知掩码类型 {mask_type!r}，可选：{MASK_TYPES}")
    if not 0.0 < target_area < 1.0:
        raise ValueError(f"target_area 需位于 (0, 1)，收到 {target_area}")

    if mask_type == "center":
        mask = _center_mask(size, target_area)
    else:
        target = _target_pixels(size, target_area)
        if mask_type == "scratch":
            mask = _scratch_mask(size, target, seed, target_area)
        else:
            mask = _scatter_mask(size, target, seed, target_area)

    return torch.from_numpy(mask.astype(np.float32)).view(1, 1, size, size)


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
