"""图片 / 掩码 I/O、256 对齐、数值范围转换与掩码极性转换（S1）。

约定（PRD）：
- 内部张量约定：图片 (1, 3, 256, 256) float32，范围 [-1, 1]；掩码 (1, 1, 256, 256) float32，
  取值 {0, 1}，其中 **1 表示缺失**。
- 上游 LanPaintBench 约定相反（1 表示已知/保留），调用前必须经 :func:`mask_to_upstream` 取反。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_SIZE = 256


def unit_to_model(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1]。"""
    return x * 2.0 - 1.0


def model_to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]。"""
    return (x + 1.0) / 2.0


def align_image(img: Image.Image, size: int = MODEL_SIZE) -> Image.Image:
    """把图片转换为 RGB 并双三次缩放到 size×size。"""
    return img.convert("RGB").resize((size, size), Image.BICUBIC)


def align_mask(img: Image.Image, size: int = MODEL_SIZE) -> Image.Image:
    """把掩码转换为单通道并最近邻缩放到 size×size（保持二值）。"""
    return img.convert("L").resize((size, size), Image.NEAREST)


def load_image(path: str | Path, size: int = MODEL_SIZE) -> torch.Tensor:
    """读取图片并返回 (1, 3, size, size) float32，范围 [-1, 1]。灰度图复制为 3 通道。"""
    img = align_image(Image.open(path), size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return unit_to_model(tensor).contiguous()


def load_mask(path: str | Path, size: int = MODEL_SIZE) -> torch.Tensor:
    """读取掩码并返回 (1, 1, size, size) float32，二值化为 {0, 1}（1 = 缺失）。"""
    img = align_mask(Image.open(path), size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return (tensor >= 0.5).to(torch.float32).contiguous()


def mask_to_upstream(mask: torch.Tensor) -> torch.Tensor:
    """本项目掩码（1 = 缺失）转上游掩码（1 = 保留）。"""
    return 1.0 - mask


def save_result(path: str | Path, image: torch.Tensor) -> None:
    """把 [-1, 1] 的图片张量保存为 uint8 PNG（自动 clip 到 [0, 1]）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = image.detach().to(torch.float32).cpu()
    if x.dim() == 4:
        x = x[0]
    x = model_to_unit(x).clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
