"""本地 Gradio 修复界面，复用现有固定预算推理。"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
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
UI_VERSION = "workbench-20261127"
READY_STATUS = "**准备开始**\n\n上传图片并标记区域，然后点击「开始修复」。"


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
            "ui_version": UI_VERSION,
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


def clear_workspace():
    """输入变更时清除旧结果和下载，避免与新输入或参数混淆。"""
    return (None, READY_STATUS, gr.update(value=None, interactive=False),
            gr.update(value=None, interactive=False), None, None, None, None)


def comparison_pair(snapshot, source):
    if not snapshot:
        return None
    key = "original" if source == "原图" else "input"
    return snapshot[key], snapshot["result"]


def preview_workspace(editor, uploaded_mask, mask_mode):
    preview, mask = preview_input(editor, uploaded_mask, mask_mode)
    area = float((np.asarray(mask) > 0).mean())
    return preview, mask, f"**区域已确认 · {area:.1%}**\n\n在「区域预览」中检查白色区域；准备好后开始修复。"


def workspace_repair(editor, uploaded_mask, mask_mode, budget, seed, progress=gr.Progress()):  # noqa: B008
    """界面任务状态；保留原 /repair API 的六项输出契约。"""
    disabled = [gr.update(interactive=False)] * 8
    enabled = [gr.update(interactive=True)] * 8
    no_download = gr.update(value=None, interactive=False)
    yield (None, None, None, "**正在修复**\n\n首次运行包含模型加载，请等待。生成完成后可查看对比与下载。",
           no_download, no_download, None, None, *disabled)
    try:
        values = run_repair(editor, uploaded_mask, mask_mode, budget, seed, progress)
        run_dir = Path(values[2]).parent
        snapshot = {"original": str(run_dir / "original.png"),
                    "input": str(run_dir / "input_preview.png"), "result": values[2]}
        yield (*values[:4], gr.update(value=values[4], interactive=True),
               gr.update(value=values[5], interactive=True),
               comparison_pair(snapshot, "遮挡输入"), snapshot, *enabled)
    except Exception:  # 恢复控件并通过 Gradio 显示原始错误
        yield (None, None, None, "**本次修复未完成**\n\n请根据错误提示检查输入或运行环境，再次尝试。",
               no_download, no_download, None, None, *enabled)
        raise


def build_ui():
    theme = gr.themes.Soft(
        primary_hue="indigo", secondary_hue="slate", neutral_hue="slate",
        font=["Microsoft YaHei", "Segoe UI", "Arial", "sans-serif"],
        font_mono=["Consolas", "monospace"], radius_size="lg",
    ).set(
        body_background_fill="#f5f6fa", body_background_fill_dark="#111622",
        block_background_fill="#ffffff", block_background_fill_dark="#1b2232",
        button_primary_background_fill="#5367df", button_primary_background_fill_hover="#4355c6",
        button_primary_text_color="#ffffff", button_primary_background_fill_dark="#5367df",
    )
    with gr.Blocks(title="LanPaint · 图像修复工作台", theme=theme,
                   css_paths=Path(__file__).parent / "assets" / "workbench.css",
                   analytics_enabled=False) as demo:
        snapshot = gr.State(None)
        gr.HTML('<div class="brand"><span class="brand-mark">L</span><strong>LanPaint</strong>'
                '<span class="brand-sub">/ 图像修复工作台</span>'
                '<span class="local-label">本地运行 · 256 × 256</span></div>', elem_id="lp-header")
        gr.Markdown("# 圈出缺失，补全画面。\n上传图片，标记修复区域，再对比生成效果。", elem_id="lp-intro")
        gr.HTML('<div class="step-list"><span><b>01</b> 上传图片</span><span><b>02</b> 标记区域</span>'
                '<span><b>03</b> 修复与对比</span></div>', elem_id="lp-steps")
        with gr.Row(elem_id="lp-workspace"):
            with gr.Column(scale=0, min_width=240, elem_id="lp-settings"):
                gr.Markdown("## 修复设置\n默认标准预算，适合首次体验。", elem_classes="lp-panel-title")
                mode = gr.Radio(["画笔涂抹", "上传掩码"], value="画笔涂抹", label="区域标记方式")
                uploaded = gr.Image(label="白色修复 · 黑色保留", type="pil", visible=False,
                                    height=180, elem_id="lp-uploaded-mask")
                budget = gr.Dropdown(
                    [("快速 · K=2", 2), ("标准 · K=5（推荐）", 5), ("充分 · K=8", 8)],
                    value=5, label="修复预算", elem_id="lp-budget",
                )
                gr.Markdown("预算越高耗时越长，不保证效果更好。", elem_classes="lp-helper")
                with gr.Accordion("高级设置", open=False):
                    seed = gr.Number(value=42, precision=0, label="随机种子", minimum=0, maximum=2**32 - 1)
                    gr.Markdown("相同输入和种子可复现；更换种子可尝试另一种结果。", elem_classes="lp-helper")
                run = gr.Button("开始修复", variant="primary", elem_id="lp-run")
                gr.Markdown("首次运行需要加载模型。任务串行执行，运行中请等待。", elem_classes="lp-helper")
                status = gr.Markdown(READY_STATUS, elem_id="lp-status")
            with gr.Column(min_width=300, elem_id="lp-editor-panel"):
                gr.Markdown("## 编辑画布\n涂抹区域将重新生成，其余区域保留。", elem_classes="lp-panel-title")
                editor = gr.ImageEditor(
                    label="上传图片并涂抹修复区域", type="pil", image_mode="RGBA",
                    sources=["upload"], transforms=(), height=430, elem_id="lp-editor",
                    brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                    placeholder="上传一张图片，开始标记修复区域", show_download_button=False,
                )
                with gr.Row():
                    example = gr.Button("载入示例图片")
                    preview_button = gr.Button("预览修复区域")
            with gr.Column(min_width=300, elem_id="lp-results-panel"):
                gr.Markdown("## 结果与对比\n生成后查看细节，下载满意的结果。", elem_classes="lp-panel-title")
                with gr.Tabs(elem_id="lp-result-tabs") as result_tabs:
                    with gr.Tab("修复结果", id="result"):
                        output = gr.Image(label="修复结果", type="filepath", interactive=False,
                                          height=430, elem_id="lp-output")
                    with gr.Tab("前后对比", id="compare"):
                        comparison = gr.ImageSlider(label="左：对比来源 / 右：修复结果", type="filepath",
                                                    interactive=False, height=370, elem_id="lp-comparison")
                        comparison_source = gr.Radio(["遮挡输入", "原图"], value="遮挡输入", label="左侧对比来源")
                    with gr.Tab("区域预览", id="preview"):
                        preview = gr.Image(label="遮挡输入", interactive=False, height=220)
                        mask_preview = gr.Image(label="实际掩码 · 白色为修复区域", interactive=False, height=180)
                with gr.Row(elem_id="lp-downloads"):
                    result_file = gr.DownloadButton(label="下载结果图", interactive=False)
                    config_file = gr.DownloadButton(label="下载配置", interactive=False)
        gr.Markdown("LanPaint · 本地修复原型　｜　生成内容可能与原物体不同，请结合画面判断。"
                    "\n\n输出统一为 256 × 256；上传图片没有完整参考答案，不显示参考型质量分数。", elem_id="lp-footer")
        mode.change(lambda value: gr.update(visible=value == "上传掩码"), mode, uploaded, queue=False, api_name=False)
        clear_outputs = [output, status, result_file, config_file, comparison, snapshot, preview, mask_preview]
        example.click(load_example, outputs=editor, queue=False, api_name=False).then(
            clear_workspace, outputs=clear_outputs, queue=False, api_name=False)
        for component in (editor, uploaded, mode, budget, seed):
            component.input(clear_workspace, outputs=clear_outputs, queue=False, api_name=False)
        preview_button.click(preview_workspace, [editor, uploaded, mode],
                             [preview, mask_preview, status], queue=False, api_name=False).success(
            lambda: gr.update(selected="preview"), outputs=result_tabs, queue=False, api_name=False)
        comparison_source.input(comparison_pair, [snapshot, comparison_source], comparison, queue=False, api_name=False)
        controls = [run, editor, uploaded, mode, budget, seed, example, preview_button]
        run.click(
            workspace_repair, [editor, uploaded, mode, budget, seed],
            [preview, mask_preview, output, status, result_file, config_file, comparison, snapshot, *controls],
            concurrency_limit=1, concurrency_id="lanpaint_gpu", api_name=False,
        ).success(lambda: (gr.update(selected="result"), "遮挡输入"),
                  outputs=[result_tabs, comparison_source], queue=False, api_name=False)
        api_button = gr.Button(visible=False)
        api_button.click(
            run_repair, [editor, uploaded, mode, budget, seed],
            [preview, mask_preview, output, status, result_file, config_file],
            concurrency_limit=1, concurrency_id="lanpaint_gpu", api_name="repair", show_api=False,
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
