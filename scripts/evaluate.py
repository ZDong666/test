"""离线评价入口（FR-08 / FR-09）。

对某个实验目录 ``results/raw/<exp>/images/*.png`` 逐张计算缺失像素 PSNR、
全图 LPIPS 与边界带 MAE，并把逐样本、按图汇总与总表写入
``results/metrics/<exp>/``。

参考图与掩码的确定顺序：
1. 命令行 ``--reference`` / ``--mask``；
2. 该实验 ``report.json`` 中记录的 ``config.sample`` / ``config.mask``；
3. 报错退出。

评价默认在 CPU 上执行，避免与扩散模型争用显存。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from app.data import load_image, load_mask
from app.metrics import evaluate

RAW_ROOT = PROJECT_ROOT / "results" / "raw"
METRICS_ROOT = PROJECT_ROOT / "results" / "metrics"
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples" / "original"
MASK_DIR = PROJECT_ROOT / "data" / "masks"

METRIC_COLUMNS = ["masked_psnr", "full_lpips", "boundary_mae_3px"]


def _resolve_reference_and_mask(
    raw_dir: Path, reference: str | None, mask: str | None
) -> tuple[Path, Path]:
    if reference and mask:
        return Path(reference), Path(mask)

    report_path = raw_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"未找到 {report_path}；请用 --reference / --mask 显式指定参考图与掩码。"
        )
    config = json.loads(report_path.read_text(encoding="utf-8")).get("config", {})
    sample_name = config.get("sample")
    mask_name = config.get("mask")
    if not sample_name or not mask_name:
        raise ValueError(f"{report_path} 的 config 缺少 sample/mask 字段")

    resolved_reference = Path(reference) if reference else SAMPLE_DIR / sample_name
    resolved_mask = Path(mask) if mask else MASK_DIR / mask_name
    return resolved_reference, resolved_mask


def main() -> int:
    parser = argparse.ArgumentParser(description="离线评价：PSNR / LPIPS / 边界带 MAE。")
    parser.add_argument("--exp", default="s1_smoke", help="实验 ID（对应 results/raw/<exp>）。")
    parser.add_argument("--raw-dir", default=None, help="实验原始结果目录，默认 results/raw/<exp>。")
    parser.add_argument("--reference", default=None, help="参考图路径。")
    parser.add_argument("--mask", default=None, help="掩码路径（1 = 缺失）。")
    parser.add_argument("--out", default=None, help="指标输出目录，默认 results/metrics/<exp>。")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_ROOT / args.exp
    out_dir = Path(args.out) if args.out else METRICS_ROOT / args.exp
    image_paths = sorted((raw_dir / "images").glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"未找到结果图：{raw_dir / 'images'}")

    reference_path, mask_path = _resolve_reference_and_mask(raw_dir, args.reference, args.mask)
    reference = load_image(reference_path)
    mask = load_mask(mask_path)
    print(
        f"[info] exp={args.exp} images={len(image_paths)} "
        f"reference={reference_path.name} mask={mask_path.name} device=cpu",
        flush=True,
    )

    rows = []
    for image_path in image_paths:
        result = load_image(image_path)
        metrics = evaluate(result, reference, mask)
        rows.append(
            {
                "run": image_path.stem,
                "image": image_path.name,
                "reference_image": reference_path.name,
                "mask": mask_path.name,
                "masked_psnr": metrics.masked_psnr,
                "full_lpips": metrics.full_lpips,
                "boundary_mae_3px": metrics.boundary_mae_3px,
                "status": metrics.status,
                "error": metrics.error or "",
            }
        )
        print(
            f"[{image_path.stem}] status={metrics.status} psnr={metrics.masked_psnr:.3f} "
            f"lpips={metrics.full_lpips:.4f} boundary_mae={metrics.boundary_mae_3px:.4f}"
            + (f" error={metrics.error}" if metrics.error else ""),
            flush=True,
        )

    per_sample = pd.DataFrame(rows)
    ok = per_sample[per_sample["status"] == "ok"]
    summary = pd.DataFrame(
        [
            {
                "experiment_id": args.exp,
                "num_runs": len(per_sample),
                "num_ok": len(ok),
                "reference": reference_path.name,
                "mask": mask_path.name,
                **{f"mean_{column}": float(ok[column].mean()) if len(ok) else float("nan") for column in METRIC_COLUMNS},
            }
        ]
    )
    per_image = ok.groupby("reference_image", as_index=False).agg(
        num_runs=("run", "count"),
        **{column: (column, "mean") for column in METRIC_COLUMNS},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(out_dir / "per_sample.csv", index=False, encoding="utf-8")
    per_image.to_csv(out_dir / "per_image.csv", index=False, encoding="utf-8")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")

    print(f"[out] {out_dir}", flush=True)
    print(summary.to_string(index=False), flush=True)

    return 0 if len(ok) == len(per_sample) else 1


if __name__ == "__main__":
    raise SystemExit(main())
