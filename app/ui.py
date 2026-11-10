"""本地 Gradio 修复界面，复用现有固定预算推理。"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import torch
from PIL import Image

from app.data import align_image, align_mask, save_result
from app.restore import PROJECT_ROOT, RestoreConfig, restore

OUTPUT_DIR = PROJECT_ROOT / "results" / "ui"
GPU_LOCK = threading.Lock()


def prepare_input(editor, uploaded_mask, mask_mode):
    """从背景读取原图，只从绘画层 alpha 读取笔迹，避免涂抹污染原图。"""
    if not editor or editor.get("background") is None:
        raise gr.Error("请先上传一张图片。")
    original = align_image(editor["background"])
    if mask_mode == "上传掩码":
        if uploaded_mask is None:
            raise gr.Error("请上传掩码：白色表示修复区域，黑色表示保留区域。")
        mask_array = np.asarray(align_mask(uploaded_mask)) >= 128
    else:
        mask_array = np.zeros((256, 256), dtype=bool)
        for layer in editor.get("layers") or []:
            alpha = layer.convert("RGBA").getchannel("A")
            mask_array |= np.asarray(align_mask(alpha)) > 0
    if mask_array.all():
        raise gr.Error("请保留一部分原图；当前版本不支持整张图片全部遮挡。")
    pixels = np.asarray(original).copy()
    pixels[mask_array] = 0
    mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255)
    return original, mask_image, Image.fromarray(pixels)


def preview_input(editor, uploaded_mask, mask_mode):
    _, mask, preview = prepare_input(editor, uploaded_mask, mask_mode)
    return preview, mask


def run_repair(editor, uploaded_mask, mask_mode, budget, seed, progress=gr.Progress()):  # noqa: B008 - Gradio 注入进度对象
    original, mask_image, preview = prepare_input(editor, uploaded_mask, mask_mode)
    if budget not in (2, 5, 8):
        raise gr.Error("请选择低、中或高预算。")
    if seed is None or int(seed) != seed or not 0 <= seed <= 2**32 - 1:
        raise gr.Error("随机种子应为 0 到 4294967295 的整数。")
    image = torch.from_numpy(np.asarray(original, dtype=np.float32) / 127.5 - 1)
    image = image.permute(2, 0, 1).unsqueeze(0).contiguous()
    mask = torch.from_numpy((np.asarray(mask_image) > 0).astype(np.float32))[None, None]
    if mask.any() and not torch.cuda.is_available():
        raise gr.Error("未检测到可用的 NVIDIA GPU，请在 LanPaint 环境中启动。")
    config = RestoreConfig(n_steps=int(budget), seed=int(seed))
    progress(0.1, desc="等待模型加载与修复，首次运行需要更长时间")
    try:
        with GPU_LOCK:
            result = restore(image, mask, config)
        if not torch.isfinite(result.image).all():
            raise ValueError("输出包含无效数值，请更换种子重试。")
        run_dir = OUTPUT_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
        run_dir.mkdir(parents=True)
        result_path = run_dir / "result.png"
        save_result(result_path, result.image)
        original.save(run_dir / "original.png")
        mask_image.save(run_dir / "mask.png")
        preview.save(run_dir / "input_preview.png")
        payload = {
            "config": asdict(config),
            "mask_source": mask_mode,
            "missing_area": float(mask.mean()),
            "nfe": result.nfe,
            "elapsed_seconds": result.elapsed_seconds,
            "peak_vram_mb": result.peak_vram_mb,
        }
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except torch.cuda.OutOfMemoryError as exc:
        raise gr.Error("显存不足，请关闭占用 GPU 的程序后重试。") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error(f"修复失败：{exc}") from exc
    progress(1, desc="修复完成")
    status = (
        f"**修复完成** · K={int(budget)} · NFE={result.nfe} · "
        f"采样耗时 {result.elapsed_seconds:.2f} 秒 · 峰值显存 {result.peak_vram_mb:.0f} MB"
        f"\n\n修复面积 {float(mask.mean()):.1%}。"
    )
    if not mask.any():
        status += "未选择修复区域，已直接返回原图。"
    return preview, mask_image, str(result_path), status, str(result_path), str(config_path)


def load_example():
    paths = sorted((PROJECT_ROOT / "data" / "samples" / "original").glob("*.png"))
    if not paths:
        raise gr.Error("本地尚无示例图片，请直接上传图片。")
    with Image.open(paths[0]) as image:
        original = align_image(image)
    return {"background": original, "layers": [], "composite": original}


def build_ui():
    theme = gr.themes.Soft(font=["Arial", "sans-serif"], font_mono=["Consolas", "monospace"])
    with gr.Blocks(title="LanPaint 图像修复", theme=theme) as demo:
        gr.Markdown("# LanPaint 图像修复\n上传图片，标记需要修复的区域，让 AI 补全画面。")
        gr.Markdown("当前使用基础模型，输出尺寸为 **256 × 256**。建议先使用中预算体验。")
        with gr.Row():
            with gr.Column():
                editor = gr.ImageEditor(
                    label="① 上传图片并涂抹修复区域", type="pil", image_mode="RGBA",
                    sources=["upload"], transforms=(), height=420,
                    brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                )
                example = gr.Button("载入本地示例图片")
                mode = gr.Radio(["画笔涂抹", "上传掩码"], value="画笔涂抹", label="掩码来源")
                uploaded = gr.Image(label="上传掩码（白色修复，黑色保留）", type="pil", visible=False)
                with gr.Row():
                    budget = gr.Dropdown(
                        [("低预算 · K=2", 2), ("中预算 · K=5（推荐）", 5), ("高预算 · K=8", 8)],
                        value=5, label="② 修复预算",
                    )
                    seed = gr.Number(value=42, precision=0, label="随机种子")
                with gr.Row():
                    preview_button = gr.Button("预览修复区域")
                    run = gr.Button("③ 开始修复", variant="primary")
            with gr.Column():
                output = gr.Image(label="修复结果", type="filepath", interactive=False, height=420)
                status = gr.Markdown("等待开始。首次运行需要加载模型，请耐心等待。")
                with gr.Row():
                    result_file = gr.File(label="下载结果图", interactive=False)
                    config_file = gr.File(label="下载本次配置", interactive=False)
                with gr.Row():
                    preview = gr.Image(label="遮挡输入预览", interactive=False)
                    mask_preview = gr.Image(label="实际修复区域", interactive=False)
        gr.Markdown("提示：高预算不保证更好的效果；可更换种子尝试。上传图片没有完整参考答案，因此不展示 PSNR / LPIPS。")
        mode.change(lambda value: gr.update(visible=value == "上传掩码"), mode, uploaded, queue=False)
        example.click(load_example, outputs=editor, queue=False)
        preview_button.click(preview_input, [editor, uploaded, mode], [preview, mask_preview], queue=False)
        run.click(
            run_repair, [editor, uploaded, mode, budget, seed],
            [preview, mask_preview, output, status, result_file, config_file],
            concurrency_limit=1, concurrency_id="lanpaint_gpu", api_name="repair",
        )
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    cache_dir = PROJECT_ROOT / ".cache" / "cuda"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache_dir))
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    build_ui().launch(server_name="127.0.0.1", share=False, inbrowser=False)


if __name__ == "__main__":
    main()
