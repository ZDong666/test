"""离线评价指标（FR-08 / PRD §8.1 ``evaluate``）。

计算三类指标，全部在 CPU 上完成，避免与扩散模型争用显存：

- ``masked_psnr``：仅统计缺失像素的 PSNR；
- ``full_lpips``：全图 LPIPS（固定 AlexNet 权重）；
- ``boundary_mae_3px``：缺失区域内侧 3 像素带的 MAE。

约定：输入图片为 (1, 3, H, W) float32、范围 [-1, 1]；掩码为 (1, 1, H, W) float32、
取值 {0, 1}，**1 表示缺失**。

注意：全图 LPIPS 直接在完整图像上计算，**不得**把已知区域置零后再计算（FR-08 验收）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from app.data import model_to_unit

#: 缺失区域内侧边界带宽度（像素）
BOUNDARY_BAND_PX = 3


@dataclass(frozen=True)
class EvalMetrics:
    """一次评价的结果（PRD §8.1）。"""

    masked_psnr: float
    full_lpips: float
    boundary_mae_3px: float
    status: str
    error: str | None = None


def _to_hwc_unit(image: Any) -> np.ndarray:
    """把图片张量转为 (H, W, 3) float64、范围 [0, 1]。"""
    tensor = image.detach().to(torch.float32).cpu() if hasattr(image, "detach") else torch.as_tensor(image, dtype=torch.float32)
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3:
        raise ValueError(f"图片应为 (1, 3, H, W) 或 (3, H, W)，实际形状 {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    unit = model_to_unit(tensor).clamp(0.0, 1.0)
    return unit.permute(1, 2, 0).numpy().astype(np.float64)


def _to_bool_mask(mask: Any) -> np.ndarray:
    """把掩码转为二维 bool 数组（True = 缺失）。"""
    raw = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    squeezed = np.squeeze(np.asarray(raw))
    if squeezed.ndim != 2:
        raise ValueError(f"掩码应可压缩为二维 (H, W)，实际形状为 {np.asarray(raw).shape}")
    return (squeezed >= 0.5) if squeezed.dtype != bool else squeezed


def masked_psnr(result: Any, reference: Any, mask: Any) -> float:
    """仅缺失像素的 PSNR（dB）；数据范围 [0, 1]，因此峰值取 1。"""
    missing = _to_bool_mask(mask)
    if not missing.any():
        raise ValueError("空掩码无法计算缺失像素 PSNR")
    result_unit = _to_hwc_unit(result)
    reference_unit = _to_hwc_unit(reference)
    if result_unit.shape != reference_unit.shape:
        raise ValueError(f"结果与参考图尺寸不一致：{result_unit.shape} vs {reference_unit.shape}")
    diff = (result_unit - reference_unit)[missing]
    mse = float(np.mean(diff ** 2))
    if mse == 0.0:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def _dilate_3x3(binary: np.ndarray) -> np.ndarray:
    """8 邻域二值膨胀一步。"""
    padded = np.pad(binary, 1, constant_values=False)
    height, width = binary.shape
    out = np.zeros_like(binary)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + height, dx : dx + width]
    return out


def boundary_band(mask: Any, width: int = BOUNDARY_BAND_PX) -> np.ndarray:
    """缺失区域内侧 ``width`` 像素带：到已知区域切比雪夫距离 ≤ ``width`` 的缺失像素。"""
    missing = _to_bool_mask(mask)
    frontier = ~missing
    for _ in range(width):
        frontier = _dilate_3x3(frontier)
    return missing & frontier


def boundary_mae_3px(result: Any, reference: Any, mask: Any) -> float:
    """缺失区内侧边界带的平均绝对误差。"""
    band = boundary_band(mask)
    if not band.any():
        raise ValueError("掩码没有可利用的边界带（空掩码或全缺失掩码）")
    result_unit = _to_hwc_unit(result)
    reference_unit = _to_hwc_unit(reference)
    diff = np.abs(result_unit - reference_unit)[band]
    return float(np.mean(diff))


_LPIPS_NET: Any = None


def _get_lpips() -> Any:
    """惰性构造 LPIPS(alex)，CPU 推理，全局复用。"""
    global _LPIPS_NET
    if _LPIPS_NET is None:
        import lpips

        net = lpips.LPIPS(net="alex", verbose=False)
        net.eval()
        _LPIPS_NET = net
    return _LPIPS_NET


def full_lpips(result: Any, reference: Any) -> float:
    """全图 LPIPS（AlexNet）；输入维持 [-1, 1]，不置零任何区域。"""
    def _to_4d(image: Any) -> torch.Tensor:
        tensor = image.detach().to(torch.float32).cpu() if hasattr(image, "detach") else torch.as_tensor(image, dtype=torch.float32)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        return tensor.contiguous()

    net = _get_lpips()
    with torch.no_grad():
        value = net(_to_4d(result), _to_4d(reference))
    return float(value.item())


def evaluate(result: Any, reference: Any, mask: Any) -> EvalMetrics:
    """计算全部指标；单个指标失败时记录为 ``status="error"`` 而不抛出（PRD §8.1）。"""
    try:
        psnr = masked_psnr(result, reference, mask)
        lpips_value = full_lpips(result, reference)
        mae = boundary_mae_3px(result, reference, mask)
    except Exception as exc:  # noqa: BLE001 - 评价需把失败原因写入结果
        return EvalMetrics(
            masked_psnr=float("nan"),
            full_lpips=float("nan"),
            boundary_mae_3px=float("nan"),
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    return EvalMetrics(
        masked_psnr=psnr,
        full_lpips=lpips_value,
        boundary_mae_3px=mae,
        status="ok",
        error=None,
    )
