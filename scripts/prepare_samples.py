"""准备 S1 样本：取少量 ImageNet 256 图片并生成中心块掩码。

数据来源（按优先级）：
1. ``--parquet`` 指定的本地 parquet 分片；
2. ``data/_cache/`` 下自动发现的 parquet 分片（推荐）；
3. ``--local-dir`` 指定的本地图片目录；
4. Hugging Face 流式下载（``datasets.load_dataset``，需要可用的 HF_ENDPOINT/代理）。

图片保存到 data/samples/original，中心块掩码保存到 data/masks，
最后写出 data/manifest.csv。

> 说明：本数据集的 parquet 为 Xet 存储，hf-mirror 的 resolve 会 302 到
> ``cas-bridge.xethub.hf.co``，而 huggingface_hub 会拒绝该跳转（报
> "Distant resource does not seem to be on huggingface.co"）；直连 huggingface.co
> 走代理又极慢。因此推荐直接用 curl 从 hf-mirror 整文件下载一个 parquet 分片：
> ``curl -L -o data/_cache/test-00000-of-00003.parquet \
> https://hf-mirror.com/datasets/benjamin-paine/imagenet-1k-256x256/resolve/main/data/test-00000-of-00003.parquet``
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from app.data import MODEL_SIZE, align_image, sha256_file
from app.masks import generate_mask, mask_area_ratio, save_mask_png

HF_DATASET = "benjamin-paine/imagenet-1k-256x256"
HF_DATA_FILES = {"test": "data/test-*.parquet"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CACHE_DIR = PROJECT_ROOT / "data" / "_cache"

MANIFEST_FIELDS = [
    "sample_id",
    "original_image_id",
    "split",
    "mask_type",
    "target_area",
    "actual_area",
    "seed",
    "image_path",
    "mask_path",
    "sha256",
]


def _iter_hf_images(count: int):
    from datasets import load_dataset

    dataset = load_dataset(
        HF_DATASET,
        data_files=HF_DATA_FILES,
        streaming=True,
        verification_mode="no_checks",
    )["test"]
    for index, row in enumerate(dataset):
        if index >= count:
            break
        yield index, row["image"]


def _iter_local_images(local_dir: Path, count: int):
    paths = sorted(p for p in local_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for index, path in enumerate(paths[:count]):
        yield index, Image.open(path)


def _iter_parquet_images(parquet_path: Path, count: int):
    """从本地 parquet 分片读取前 count 张图片（列：image=struct<bytes, path>）。"""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    produced = 0
    for batch in parquet_file.iter_batches(batch_size=count, columns=["image"]):
        for row in batch.to_pylist():
            if produced >= count:
                return
            yield produced, Image.open(io.BytesIO(row["image"]["bytes"]))
            produced += 1
        break


def _find_cached_parquet() -> Path | None:
    if not CACHE_DIR.is_dir():
        return None
    shards = sorted(CACHE_DIR.glob("*.parquet"))
    return shards[0] if shards else None


def main() -> int:
    parser = argparse.ArgumentParser(description="准备 S1 样本图片与中心块掩码。")
    parser.add_argument("--count", type=int, default=10, help="抓取图片数量。")
    parser.add_argument("--target-area", type=float, default=0.30, help="目标缺失面积比例。")
    parser.add_argument("--seed", type=int, default=42, help="掩码种子（center 不使用随机性）。")
    parser.add_argument("--out", default="data", help="输出根目录（相对项目根）。")
    parser.add_argument("--local-dir", default=None, help="本地图片目录兜底（HF 不可用时）。")
    parser.add_argument("--parquet", default=None, help="本地 parquet 分片路径（优先于其它来源）。")
    args = parser.parse_args()

    out_root = PROJECT_ROOT / args.out
    image_dir = out_root / "samples" / "original"
    mask_dir = out_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_type = "center"
    tag = f"{mask_type}{int(round(args.target_area * 100))}"

    parquet_path = Path(args.parquet) if args.parquet else _find_cached_parquet()
    if parquet_path is not None:
        print(f"[info] 从本地 parquet 读取 {args.count} 张图片：{parquet_path}", flush=True)
        source = _iter_parquet_images(parquet_path, args.count)
    elif args.local_dir:
        print(f"[warn] 使用本地目录兜底：{args.local_dir}", flush=True)
        source = _iter_local_images(Path(args.local_dir), args.count)
    else:
        print(f"[info] 通过 Hugging Face 流式获取 {args.count} 张图片", flush=True)
        source = _iter_hf_images(args.count)

    rows = []
    for index, image in source:
        name = f"img{index:04d}"
        image_path = image_dir / f"{name}.png"
        mask_path = mask_dir / f"{name}_{tag}_mask.png"

        align_image(image, MODEL_SIZE).save(image_path)

        mask = generate_mask(mask_type, size=MODEL_SIZE, target_area=args.target_area, seed=args.seed)
        save_mask_png(mask_path, mask)

        actual_area = mask_area_ratio(mask)
        if abs(actual_area - args.target_area) > 0.02:
            print(
                f"[warn] {name} 实际面积 {actual_area:.4f} 与目标 {args.target_area:.4f} 偏差超过 2pp",
                flush=True,
            )

        rows.append(
            {
                "sample_id": f"s{index:04d}",
                "original_image_id": name,
                "split": "check",
                "mask_type": mask_type,
                "target_area": f"{args.target_area:.4f}",
                "actual_area": f"{actual_area:.6f}",
                "seed": args.seed,
                "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "mask_path": mask_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(image_path),
            }
        )
        print(f"[ok] {name}  actual_area={actual_area:.4f}", flush=True)

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] 写入 {len(rows)} 条样本清单：{manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
