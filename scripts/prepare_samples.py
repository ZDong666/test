"""准备样本：取 ImageNet 256 图片，生成三类掩码 × 两种面积的 50 图清单（S4 / FR-03）。

数据来源（按优先级）：
1. ``--parquet`` 指定的本地 parquet 分片；
2. ``data/_cache/`` 下自动发现的 parquet 分片（推荐）；
3. ``--local-dir`` 指定的本地图片目录；
4. Hugging Face 流式下载（``datasets.load_dataset``，需要可用的 HF_ENDPOINT/代理）。

图片保存到 data/samples/original，掩码保存到 data/masks，最后写出 data/manifest.csv。

清单粒度为「原图 × 掩码类型 × 目标面积」（PRD §7.2 / §7.6）：
50 张原图（检查集 10 + 开发集 10 + 正式测试集 30，划分只依赖原图，见 PRD §7.1）
× 3 类掩码（center / scratch / scatter）× 2 种目标面积（15% / 30%）= 300 行。

掩码种子逐图派生为 ``seed + 原图索引``，使同类型掩码在不同原图上形状不同；
配合 :func:`app.masks.generate_mask` 内部的 (类型, 面积) 编码，同图不同掩码版本互不为子集。

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
from app.features import extract_features
from app.masks import (
    AREA_TARGETS,
    AREA_TOLERANCE,
    MASK_TYPES,
    generate_mask,
    mask_area_ratio,
    save_mask_png,
)

HF_DATASET = "benjamin-paine/imagenet-1k-256x256"
HF_DATA_FILES = {"test": "data/test-*.parquet"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CACHE_DIR = PROJECT_ROOT / "data" / "_cache"

#: 数据划分（PRD §7.1）：检查集 10 + 开发集 10 + 正式测试集 30 = 50 张原图
CHECK_COUNT = 10
DEV_COUNT = 10
TEST_COUNT = 30
TOTAL_COUNT = CHECK_COUNT + DEV_COUNT + TEST_COUNT

MANIFEST_FIELDS = [
    "sample_id",
    "original_image_id",
    "split",
    "mask_type",
    "target_area",
    "actual_area",
    "boundary_norm",
    "seed",
    "image_path",
    "mask_path",
    "sha256",
]


def _split_for(index: int) -> str:
    """按原图索引划分；只依赖原图，保证同一原图的掩码版本不跨划分（PRD §7.1）。"""
    if index < CHECK_COUNT:
        return "check"
    if index < CHECK_COUNT + DEV_COUNT:
        return "dev"
    return "test"


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
    parser = argparse.ArgumentParser(
        description="准备样本：50 张图片 × 3 类掩码 × 2 种面积的清单。"
    )
    parser.add_argument("--count", type=int, default=TOTAL_COUNT, help="抓取图片数量（默认 50）。")
    parser.add_argument("--seed", type=int, default=42, help="掩码基准种子；逐图派生为 seed + 索引。")
    parser.add_argument("--out", default="data", help="输出根目录（相对项目根）。")
    parser.add_argument("--local-dir", default=None, help="本地图片目录兜底（HF 不可用时）。")
    parser.add_argument("--parquet", default=None, help="本地 parquet 分片路径（优先于其它来源）。")
    args = parser.parse_args()

    out_root = PROJECT_ROOT / args.out
    image_dir = out_root / "samples" / "original"
    mask_dir = out_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

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
        align_image(image, MODEL_SIZE).save(image_path)
        image_sha = sha256_file(image_path)
        split = _split_for(index)
        mask_seed = args.seed + index

        for mask_type in MASK_TYPES:
            for target_area in AREA_TARGETS:
                tag = f"{mask_type}{round(target_area * 100)}"
                mask_path = mask_dir / f"{name}_{tag}_mask.png"
                mask = generate_mask(
                    mask_type, size=MODEL_SIZE, target_area=target_area, seed=mask_seed
                )
                save_mask_png(mask_path, mask)

                actual_area = mask_area_ratio(mask)
                if abs(actual_area - target_area) > AREA_TOLERANCE:
                    print(
                        f"[warn] {name} {tag} 实际面积 {actual_area:.4f} "
                        f"与目标 {target_area:.4f} 偏差超过 {AREA_TOLERANCE:.0%}",
                        flush=True,
                    )

                rows.append(
                    {
                        "sample_id": f"{name}_{tag}",
                        "original_image_id": name,
                        "split": split,
                        "mask_type": mask_type,
                        "target_area": f"{target_area:.4f}",
                        "actual_area": f"{actual_area:.6f}",
                        "boundary_norm": f"{extract_features(None, mask).boundary_norm:.6f}",
                        "seed": mask_seed,
                        "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                        "mask_path": mask_path.relative_to(PROJECT_ROOT).as_posix(),
                        "sha256": image_sha,
                    }
                )
        print(
            f"[ok] {name} split={split} 掩码 {len(MASK_TYPES) * len(AREA_TARGETS)} 个",
            flush=True,
        )

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] 写入 {len(rows)} 条样本清单：{manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
