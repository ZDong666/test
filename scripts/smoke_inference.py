"""S1 冒烟测试：单次或连续 N 次真实 LanPaint 推理，输出可行性证据 report.json。

判定字段：ok_runs / nfe_per_run / nan_or_inf / blank / peak_vram_mb / 中位耗时，
并据此估算 PRD 最小实验规模 495 次的 GPU 小时。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from app.data import load_image, load_mask, save_result
from app.restore import RestoreConfig, restore

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "raw" / "s1_smoke"
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples" / "original"
MASK_DIR = PROJECT_ROOT / "data" / "masks"


def _pick_default_pair() -> tuple[Path, Path]:
    samples = sorted(p for p in SAMPLE_DIR.glob("*.png"))
    masks = sorted(p for p in MASK_DIR.glob("*.png"))
    if not samples:
        raise FileNotFoundError(f"未找到样本图片，请先运行 prepare_samples.py（{SAMPLE_DIR}）")
    if not masks:
        raise FileNotFoundError(f"未找到掩码，请先运行 prepare_samples.py（{MASK_DIR}）")
    index = min(len(samples), len(masks)) - 1  # 取最后一张，保证两者成对
    return samples[index], masks[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="S1 真实推理冒烟测试。")
    parser.add_argument("--runs", type=int, default=1, help="连续推理次数。")
    parser.add_argument("--sample", default=None, help="样本图片路径（默认取 data/samples/original 最后一张）。")
    parser.add_argument("--mask", default=None, help="掩码路径（默认取 data/masks 最后一张）。")
    parser.add_argument("--n-steps", type=int, default=5, help="内层预算 K（LanPaint-K）。")
    parser.add_argument("--outer-steps", type=int, default=20, help="外层扩散步数 T。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu。")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="输出目录。")
    args = parser.parse_args()

    default_sample, default_mask = _pick_default_pair()
    sample_path = Path(args.sample) if args.sample else default_sample
    mask_path = Path(args.mask) if args.mask else default_mask
    output_dir = Path(args.out)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    image = load_image(sample_path)
    mask = load_mask(mask_path)
    config = RestoreConfig(
        n_steps=args.n_steps,
        outer_steps=args.outer_steps,
        seed=args.seed,
        device=args.device,
    )

    use_cuda = torch.device(args.device).type == "cuda"
    print(f"[info] sample={sample_path.name} mask={mask_path.name} runs={args.runs} "
          f"K={config.n_steps} T={config.outer_steps} device={args.device}", flush=True)

    seconds = []
    nfes = []
    peak_vram_mb = 0.0
    ok_runs = 0
    any_nan_or_inf = False
    any_blank = False

    for run_index in range(args.runs):
        if use_cuda:
            torch.cuda.empty_cache()
        result = restore(image, mask, config)

        nan_or_inf = not bool(torch.isfinite(result.image).all().item())
        blank = float(result.image.std().item()) < 1e-3
        ok = (not nan_or_inf) and (not blank)

        any_nan_or_inf = any_nan_or_inf or nan_or_inf
        any_blank = any_blank or blank
        ok_runs += int(ok)
        seconds.append(result.elapsed_seconds)
        nfes.append(result.nfe)
        peak_vram_mb = max(peak_vram_mb, result.peak_vram_mb)

        save_result(output_dir / "images" / f"run{run_index:02d}.png", result.image)
        print(
            f"[run {run_index:02d}] ok={ok} nfe={result.nfe} "
            f"elapsed={result.elapsed_seconds:.2f}s peak_vram={result.peak_vram_mb:.0f}MB "
            f"nan_or_inf={nan_or_inf} blank={blank}",
            flush=True,
        )

    median_seconds = statistics.median(seconds)
    report = {
        "runs": args.runs,
        "ok_runs": ok_runs,
        "nfe_per_run": int(statistics.median(nfes)),
        "nfe_total": int(sum(nfes)),
        "nan_or_inf": any_nan_or_inf,
        "blank": any_blank,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "median_seconds": round(median_seconds, 4),
        "mean_seconds": round(statistics.fmean(seconds), 4),
        "estimated_gpu_hours_495": round(median_seconds * 495 / 3600.0, 3),
        "config": {
            "n_steps": config.n_steps,
            "outer_steps": config.outer_steps,
            "seed": config.seed,
            "precision": "fp32",
            "sample": sample_path.name,
            "mask": mask_path.name,
        },
        "notes": f"generated_at={time.strftime('%Y-%m-%d %H:%M:%S')}",
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = (
        report["ok_runs"] == report["runs"]
        and report["nfe_per_run"] > 0
        and not report["nan_or_inf"]
        and not report["blank"]
        and report["peak_vram_mb"] < 8000
    )
    print(f"[report] {report_path}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"[verdict] {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
