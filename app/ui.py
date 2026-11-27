from __future__ import annotations

import base64
import io
import json
import os
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.data import align_image, align_mask, save_result
from app.restore import PROJECT_ROOT, RestoreConfig, restore

OUTPUT_DIR = PROJECT_ROOT / "results" / "ui"
STATIC_DIR = Path(__file__).resolve().parent / "ui_static"
GPU_LOCK = threading.Lock()
BUDGETS = (2, 5, 8)
MODE_ALIASES = {
    "brush": "brush",
    "upload": "upload",
    "画笔涂抹": "brush",
    "上传掩码": "upload",
}
MODE_LABELS = {"brush": "画笔涂抹", "upload": "上传掩码"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class UIValidationError(ValueError):
    pass


class UICudaUnavailableError(RuntimeError):
    pass


class UIOutOfMemoryError(RuntimeError):
    pass


class UIInferenceError(RuntimeError):
    pass


def _normalize_mode(mask_mode: str) -> str:
    mode = MODE_ALIASES.get(str(mask_mode).strip())
    if mode is None:
        raise UIValidationError("请选择画笔涂抹或上传掩码。")
    return mode


def _validate_budget(budget: int | str) -> int:
    try:
        value = int(str(budget).strip())
    except (TypeError, ValueError) as exc:
        raise UIValidationError("请选择低、中或高预算。") from exc
    if value not in BUDGETS:
        raise UIValidationError("预算只能是 K=2、K=5 或 K=8。")
    return value


def _validate_seed(seed: int | str) -> int:
    try:
        value = int(str(seed).strip())
    except (TypeError, ValueError) as exc:
        raise UIValidationError("随机种子应为 0 到 4294967295 的整数。") from exc
    if not 0 <= value <= 2**32 - 1:
        raise UIValidationError("随机种子应为 0 到 4294967295 的整数。")
    return value


def _decode_image(data: bytes, label: str) -> Image.Image:
    if not data:
        raise UIValidationError(f"请上传{label}。")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UIValidationError(f"{label}无法解析，请上传 PNG 或 JPG 图片。") from exc


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def prepare_input(image: Image.Image, mask_image: Image.Image | None, mask_mode: str):
    mode = _normalize_mode(mask_mode)
    original = align_image(image)
    if mask_image is None:
        mask_array = np.zeros((256, 256), dtype=bool)
    elif mode == "upload":
        mask_array = np.asarray(align_mask(mask_image)) >= 128
    else:
        alpha = mask_image.convert("RGBA").getchannel("A")
        mask_array = np.asarray(align_mask(alpha)) > 0
    if mask_array.all():
        raise UIValidationError("请保留一部分原图；当前版本不支持整张图片全部遮挡。")
    pixels = np.asarray(original).copy()
    pixels[mask_array] = 0
    binary_mask = Image.fromarray(mask_array.astype(np.uint8) * 255)
    preview = Image.fromarray(pixels)
    return original, binary_mask, preview


def _make_status(k: int, result, missing_area: float, empty_mask: bool) -> dict:
    message = "修复完成"
    if empty_mask:
        message = "未选择修复区域，已直接返回原图。"
    return {
        "k": k,
        "nfe": result.nfe,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_vram_mb": result.peak_vram_mb,
        "missing_area": missing_area,
        "message": message,
    }


def run_repair(
    image: Image.Image,
    mask_image: Image.Image | None,
    mask_mode: str,
    budget: int | str,
    seed: int | str,
) -> dict:
    mode = _normalize_mode(mask_mode)
    original, binary_mask, preview = prepare_input(image, mask_image, mode)
    k = _validate_budget(budget)
    value = _validate_seed(seed)
    image_tensor = torch.from_numpy(np.asarray(original, dtype=np.float32) / 127.5 - 1)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    mask_tensor = torch.from_numpy((np.asarray(binary_mask) > 0).astype(np.float32))[None, None]
    missing_area = float(mask_tensor.mean())
    config = RestoreConfig(n_steps=k, seed=value)
    try:
        if mask_tensor.any() and not torch.cuda.is_available():
            raise UICudaUnavailableError("未检测到可用的 NVIDIA GPU，请在 LanPaint 环境中启动。")
        with GPU_LOCK:
            result = restore(image_tensor, mask_tensor, config)
        if not torch.isfinite(result.image).all():
            raise ValueError("输出包含无效数值，请更换种子重试。")
        run_dir = OUTPUT_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
        run_dir.mkdir(parents=True)
        result_path = run_dir / "result.png"
        save_result(result_path, result.image)
        original.save(run_dir / "original.png")
        binary_mask.save(run_dir / "mask.png")
        preview.save(run_dir / "input_preview.png")
        payload = {
            "config": asdict(config),
            "mask_source": MODE_LABELS[mode],
            "missing_area": missing_area,
            "nfe": result.nfe,
            "elapsed_seconds": result.elapsed_seconds,
            "peak_vram_mb": result.peak_vram_mb,
        }
        (run_dir / "config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise UIOutOfMemoryError("显存不足，请关闭占用 GPU 的程序后重试。") from exc
    except UIValidationError:
        raise
    except UICudaUnavailableError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise UIInferenceError(f"修复失败：{exc}") from exc
    base = f"/runs/{run_dir.name}"
    return {
        "run_id": run_dir.name,
        "preview_url": f"{base}/input_preview.png",
        "mask_url": f"{base}/mask.png",
        "result_url": f"{base}/result.png",
        "config_url": f"{base}/config.json",
        "status": _make_status(k, result, missing_area, not bool(mask_tensor.any())),
    }


app = FastAPI(title="LanPaint 图像修复工作台", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/runs", StaticFiles(directory=OUTPUT_DIR), name="runs")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/preview")
def preview(
    image: Annotated[UploadFile, File()],
    mask: Annotated[UploadFile | None, File()] = None,
    mode: Annotated[str, Form()] = "brush",
):
    try:
        source = _decode_image(image.file.read(), "图片")
        mask_data = mask.file.read() if mask is not None else None
        mask_image = _decode_image(mask_data, "掩码") if mask_data else None
        normalized_mode = _normalize_mode(mode)
        if normalized_mode == "upload" and mask_image is None:
            raise UIValidationError("请上传掩码：白色表示修复区域，黑色表示保留区域。")
        _, binary_mask, preview_image = prepare_input(source, mask_image, normalized_mode)
    except UIValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    mask_array = np.asarray(binary_mask) > 0
    return {
        "preview_data": _png_data_url(preview_image),
        "mask_data": _png_data_url(binary_mask),
        "missing_area": float(mask_array.mean()),
    }


@app.post("/api/repair")
def repair(
    image: Annotated[UploadFile, File()],
    mask: Annotated[UploadFile | None, File()] = None,
    mode: Annotated[str, Form()] = "brush",
    budget: Annotated[str, Form()] = "5",
    seed: Annotated[str, Form()] = "42",
):
    try:
        source = _decode_image(image.file.read(), "图片")
        mask_data = mask.file.read() if mask is not None else None
        mask_image = _decode_image(mask_data, "掩码") if mask_data else None
        result = run_repair(source, mask_image, mode, budget, seed)
    except UIValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UICudaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UIOutOfMemoryError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except UIInferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


def main():
    cache_dir = PROJECT_ROOT / ".cache" / "cuda"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache_dir))
    host = os.environ.get("LANPAINT_HOST", "127.0.0.1")
    port = int(os.environ.get("LANPAINT_PORT", "7860"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
