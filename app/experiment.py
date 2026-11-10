"""批量等预算实验编排（FR-05 / FR-09 / FR-10，PRD §7.5 / §7.6 / §8.3）。

执行流程：

1. 读取实验配置与数据清单，按 ``split`` / 掩码类型 / 目标面积 / 原图区间选出本次样本；
2. 对每个「图片 + 掩码 + 种子」分别跑低 / 中 / 高三档预算 ``K``（PRD §7.5），
   每档先查 :class:`app.cache.ResultCache`，命中则直接复用、不重复推理（FR-09）；
   单个样本失败写入 ``failures.jsonl`` 且不阻断后续样本（PRD §9.2）；
3. 复用同一批三档输出，按 B1/B2/B3/M1/A1/C1 分配预算并取对应档位结果；
   同一输出被多个策略复用，不计为独立重复实验（PRD §7.6）；
4. 每个档位输出只评价一次（缺失区 PSNR / 全图 LPIPS / 边界带 MAE），
   再按策略与按原图汇总（PRD §7.4 / §7.7）。

输出结构（PRD §8.3）::

    results/raw/<experiment_id>/
    ├─ config.yaml
    ├─ manifest.csv
    ├─ samples.csv
    ├─ run_log.jsonl
    ├─ images/
    └─ failures.jsonl
    results/metrics/<experiment_id>/
    ├─ per_sample.csv
    ├─ per_image.csv
    └─ summary.csv

``samples.csv`` 覆盖每一个「样本 × 策略」组合（策略只决定取哪一档输出），
因此六条策略共用同一次真实推理结果，真实推理次数等于「样本数 × 3」。
"""

from __future__ import annotations

import csv
import json
import shutil
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from app.budget import (
    LEVEL_HIGH,
    LEVEL_LOW,
    LEVEL_MIDDLE,
    BudgetPolicy,
    allocate_batch,
    make_policies,
    mean_k,
)
from app.cache import ResultCache, current_code_commit, make_cache_entry
from app.config import PRECISION_FP32, PROJECT_ROOT, ExperimentConfig
from app.data import load_image, load_mask, save_result, sha256_file
from app.features import MaskFeatures
from app.metrics import EvalMetrics, evaluate
from app.restore import RestoreConfig, RestoreResult, restore

#: 数据清单字段（PRD §7.2）
MANIFEST_FIELDS = (
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
)

#: 逐样本运行记录字段（results/raw/<exp>/samples.csv）
SAMPLE_FIELDS = (
    "sample_id",
    "original_image_id",
    "split",
    "mask_type",
    "target_area",
    "policy",
    "level",
    "k",
    "nfe",
    "elapsed_seconds",
    "peak_vram_mb",
    "cache_hit",
    "image",
    "masked_psnr",
    "full_lpips",
    "boundary_mae_3px",
)

#: 逐样本指标字段（results/metrics/<exp>/per_sample.csv）
METRIC_FIELDS = (
    "sample_id",
    "original_image_id",
    "split",
    "mask_type",
    "target_area",
    "policy",
    "k",
    "masked_psnr",
    "full_lpips",
    "boundary_mae_3px",
)

METRIC_COLUMNS = ("masked_psnr", "full_lpips", "boundary_mae_3px")

#: 三档执行的顺序（始终低 -> 中 -> 高，便于断点续跑时结果可预测）
TIER_LEVELS = (LEVEL_LOW, LEVEL_MIDDLE, LEVEL_HIGH)

RestoreRunner = Callable[[torch.Tensor, torch.Tensor, RestoreConfig], RestoreResult]
Evaluator = Callable[[Any, Any, Any], EvalMetrics]


@dataclass(frozen=True)
class SampleRow:
    """清单中的一条样本（PRD §7.2）。"""

    sample_id: str
    original_image_id: str
    split: str
    mask_type: str
    target_area: float
    actual_area: float
    boundary_norm: float
    seed: int
    image_path: Path
    mask_path: Path
    sha256: str

    @property
    def features(self) -> MaskFeatures:
        """掩码特征直接取自清单记录，无需重新解码图片（FR-04 特征只依赖掩码）。"""
        return MaskFeatures(area_ratio=self.actual_area, boundary_norm=self.boundary_norm)


@dataclass(frozen=True)
class TierRun:
    """某一档预算的一次运行记录（真实推理或缓存命中）。"""

    sample_id: str
    level: str
    k: int
    image_path: Path
    nfe: int
    elapsed_seconds: float
    peak_vram_mb: float
    cache_hit: bool
    blank: bool


@dataclass
class ExperimentReport:
    """一次批量实验的汇总。"""

    experiment_id: str
    output_dir: Path
    metrics_dir: Path
    num_images: int
    num_samples: int
    num_tier_runs: int
    num_cache_hits: int
    num_failures: int
    num_records: int
    policy_mean_k: dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """``ok``（无失败）/ ``partial``（有失败但仍产出结果图）/ ``failed``（未产出任何结果）。

        判定依据是**是否跑出结果**而非是否写出策略记录：小批量实验可能因
        ``usable % 3 != 0`` 跳过策略分配，此时仍属正常产出（PRD §9.2）。
        """
        if self.num_tier_runs == 0:
            return "failed"
        return "ok" if self.num_failures == 0 else "partial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "num_images": self.num_images,
            "num_samples": self.num_samples,
            "num_tier_runs": self.num_tier_runs,
            "num_cache_hits": self.num_cache_hits,
            "num_failures": self.num_failures,
            "num_records": self.num_records,
            "policy_mean_k": {name: round(value, 4) for name, value in sorted(self.policy_mean_k.items())},
            "output_dir": str(self.output_dir),
            "metrics_dir": str(self.metrics_dir),
        }


# --------------------------------------------------------------------------- #
# 清单读取与样本筛选
# --------------------------------------------------------------------------- #


def _resolve_path(text: str | Path, project_root: Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else project_root / path


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _row_from_mapping(raw: Mapping[str, str], project_root: Path) -> SampleRow:
    return SampleRow(
        sample_id=raw["sample_id"],
        original_image_id=raw["original_image_id"],
        split=raw["split"],
        mask_type=raw["mask_type"],
        target_area=float(raw["target_area"]),
        actual_area=float(raw["actual_area"]),
        boundary_norm=float(raw["boundary_norm"]),
        seed=int(raw["seed"]),
        image_path=_resolve_path(raw["image_path"], project_root),
        mask_path=_resolve_path(raw["mask_path"], project_root),
        sha256=raw["sha256"],
    )


def load_manifest(path: str | Path, *, project_root: Path = PROJECT_ROOT) -> list[SampleRow]:
    """读取数据清单；相对路径相对 ``project_root`` 解析（PRD §7.2）。"""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"未找到数据清单：{manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [name for name in MANIFEST_FIELDS if name not in header]
        if missing:
            raise ValueError(f"清单 {manifest_path} 缺少字段：{missing}")
        rows = [_row_from_mapping(raw, project_root) for raw in reader]
    if not rows:
        raise ValueError(f"清单为空：{manifest_path}")
    return rows


def select_samples(
    rows: Sequence[SampleRow],
    *,
    split: str | None = None,
    mask_types: Sequence[str] | None = None,
    target_areas: Sequence[float] | None = None,
    image_offset: int = 0,
    image_limit: int | None = None,
) -> list[SampleRow]:
    """按划分 / 掩码类型 / 目标面积 / 原图区间筛选样本，保持清单原顺序。

    ``image_offset`` 与 ``image_limit`` 以**原图**为单位（PRD §7.6 的各子集都按原图计），
    用于从测试集中切出「主测试前 20 张 / 面积扩展 5 张 / 重复实验 5 张」。
    """
    selected = [row for row in rows if split is None or row.split == split]
    if mask_types:
        wanted = set(mask_types)
        selected = [row for row in selected if row.mask_type in wanted]
    if target_areas:
        areas = [float(area) for area in target_areas]
        selected = [
            row for row in selected if any(abs(row.target_area - area) < 1e-6 for area in areas)
        ]

    image_ids = list(dict.fromkeys(row.original_image_id for row in selected))
    start = max(image_offset, 0)
    kept = image_ids[start:] if image_limit is None else image_ids[start : start + image_limit]
    allowed = set(kept)
    return [row for row in selected if row.original_image_id in allowed]


# --------------------------------------------------------------------------- #
# 缓存键与权重校验
# --------------------------------------------------------------------------- #


def sampling_parameters(config: RestoreConfig) -> dict[str, float]:
    """全部会影响输出的采样参数（缓存键的第 9 个字段，FR-09）。"""
    return {
        "step_size": float(config.step_size),
        "alpha": float(config.alpha),
        "friction": float(config.friction),
        "m": float(config.m),
        "chara_lamb": float(config.chara_lamb),
        "chara_beta": float(config.chara_beta),
    }


def resolve_checkpoint_sha256(config: ExperimentConfig) -> str:
    """返回实际权重 SHA-256：配置已给出则校验一致，未给出则就地计算。

    校验失败直接抛错——缓存键以权重哈希为准，静默复用旧结果会污染实验记录（FR-09）。
    哈希比较忽略大小写（Windows 的 ``Get-FileHash`` 输出大写，``hashlib`` 输出小写）。
    """
    expected = config.checkpoint_sha256.strip()
    if not config.checkpoint.is_file():
        if expected:
            raise FileNotFoundError(f"未找到模型权重：{config.checkpoint}")
        return ""
    actual = sha256_file(config.checkpoint)
    if expected and actual.lower() != expected.lower():
        raise ValueError(
            f"权重 SHA-256 与配置不符：配置 {expected}，实际 {actual}（{config.checkpoint}）"
        )
    return actual


# --------------------------------------------------------------------------- #
# 单档执行
# --------------------------------------------------------------------------- #


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _run_tier(
    *,
    row: SampleRow,
    level: str,
    k: int,
    image: torch.Tensor,
    mask: torch.Tensor,
    mask_sha256: str,
    images_dir: Path,
    config: ExperimentConfig,
    cache: ResultCache,
    code_commit: str,
    checkpoint_sha256: str,
    device: str,
    sampling: Mapping[str, float],
    runner: RestoreRunner,
) -> TierRun:
    """跑一档预算：命中缓存则复制结果图，否则真实推理并回写缓存。"""
    target = images_dir / f"{row.sample_id}__k{k}.png"
    restore_config = RestoreConfig(
        n_steps=k,
        outer_steps=config.outer_steps,
        seed=config.seed,
        device=device,
        checkpoint=config.checkpoint,
    )
    entry = make_cache_entry(
        code_commit=code_commit,
        checkpoint_sha256=checkpoint_sha256,
        image_sha256=row.sha256,
        mask_sha256=mask_sha256,
        seed=config.seed,
        precision=config.precision,
        budget_k=k,
        outer_steps=config.outer_steps,
        sampling=dict(sampling),
    )

    record = cache.lookup(entry)
    if record is not None:
        shutil.copyfile(record.image_path, target)
        extra = record.meta.get("extra", {})
        return TierRun(
            sample_id=row.sample_id,
            level=level,
            k=k,
            image_path=target,
            nfe=int(extra.get("nfe", 0)),
            elapsed_seconds=float(extra.get("elapsed_seconds", 0.0)),
            peak_vram_mb=float(extra.get("peak_vram_mb", 0.0)),
            cache_hit=True,
            blank=bool(extra.get("blank", False)),
        )

    result = runner(image, mask, restore_config)
    if not bool(torch.isfinite(result.image).all().item()):
        raise ValueError(f"推理结果包含 NaN/Inf（sample={row.sample_id} K={k}）")
    blank = float(result.image.std().item()) < 1e-3

    save_result(target, result.image)
    cache.store(
        entry,
        result.image,
        extra={
            "nfe": int(result.nfe),
            "elapsed_seconds": float(result.elapsed_seconds),
            "peak_vram_mb": float(result.peak_vram_mb),
            "blank": blank,
        },
    )
    return TierRun(
        sample_id=row.sample_id,
        level=level,
        k=k,
        image_path=target,
        nfe=int(result.nfe),
        elapsed_seconds=float(result.elapsed_seconds),
        peak_vram_mb=float(result.peak_vram_mb),
        cache_hit=False,
        blank=blank,
    )


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def _dump_config(
    path: Path, config: ExperimentConfig, checkpoint_sha256: str, project_root: Path
) -> None:
    payload = config.to_dict()
    for key in ("data_manifest", "checkpoint", "output_dir", "cache_dir"):
        payload[key] = _relative(Path(payload[key]), project_root)
    payload["checkpoint_sha256"] = checkpoint_sha256
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_manifest(path: Path, rows: Sequence[SampleRow], project_root: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "original_image_id": row.original_image_id,
                    "split": row.split,
                    "mask_type": row.mask_type,
                    "target_area": f"{row.target_area:.4f}",
                    "actual_area": f"{row.actual_area:.6f}",
                    "boundary_norm": f"{row.boundary_norm:.6f}",
                    "seed": row.seed,
                    "image_path": _relative(row.image_path, project_root),
                    "mask_path": _relative(row.mask_path, project_root),
                    "sha256": row.sha256,
                }
            )


def _summarize(records: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    """按策略汇总；无记录时返回仅含表头的空表。"""
    frame = pd.DataFrame(records, columns=list(columns))
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "experiment_id", "policy", "num_samples", "mean_k", "nfe_total",
                *[f"mean_{name}" for name in METRIC_COLUMNS],
            ]
        )
    grouped = frame.groupby("policy", as_index=False)
    summary = grouped.agg(
        num_samples=("sample_id", "count"),
        mean_k=("k", "mean"),
        nfe_total=("nfe", "sum"),
        **{f"mean_{name}": (name, "mean") for name in METRIC_COLUMNS},
    )
    return summary


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def run_experiment(
    config: ExperimentConfig,
    *,
    split: str | None = None,
    mask_types: Sequence[str] | None = None,
    target_areas: Sequence[float] | None = None,
    image_offset: int = 0,
    image_limit: int | None = None,
    manifest_path: str | Path | None = None,
    device: str = "cuda",
    policies: Mapping[str, BudgetPolicy] | None = None,
    runner: RestoreRunner = restore,
    evaluator: Evaluator = evaluate,
    project_root: Path = PROJECT_ROOT,
    code_commit: str | None = None,
    log: Callable[[str], None] = print,
) -> ExperimentReport:
    """执行一次批量等预算实验，返回汇总报告（PRD §7.5 / §8.3）。"""
    if config.precision != PRECISION_FP32:
        raise ValueError(
            f"当前 restore 实现仅支持 {PRECISION_FP32}，收到 precision={config.precision!r}"
        )

    output_dir = config.output_dir / config.experiment_id
    metrics_dir = project_root / "results" / "metrics" / config.experiment_id
    images_dir = output_dir / "images"
    for directory in (output_dir, images_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path or config.data_manifest, project_root=project_root)
    selected = select_samples(
        rows,
        split=split,
        mask_types=mask_types,
        target_areas=target_areas,
        image_offset=image_offset,
        image_limit=image_limit,
    )
    if not selected:
        available = sorted({row.split for row in rows})
        raise ValueError(
            f"筛选后没有样本（split={split!r}）；清单可用划分：{available}"
        )

    checkpoint_sha256 = resolve_checkpoint_sha256(config)
    resolved_commit = code_commit if code_commit is not None else current_code_commit(project_root)
    cache = ResultCache(config.cache_dir)
    tiers = config.budget.to_tiers()
    resolved_policies = (
        dict(policies) if policies is not None else make_policies(tiers=tiers)
    )

    _dump_config(output_dir / "config.yaml", config, checkpoint_sha256, project_root)
    _write_manifest(output_dir / "manifest.csv", selected, project_root)

    run_log_path = output_dir / "run_log.jsonl"
    failures_path = output_dir / "failures.jsonl"
    run_log_path.write_text("", encoding="utf-8")
    failures_path.write_text("", encoding="utf-8")

    sampling = sampling_parameters(RestoreConfig(outer_steps=config.outer_steps))
    _append_jsonl(
        run_log_path,
        {
            "event": "start",
            "experiment_id": config.experiment_id,
            "num_samples": len(selected),
            "num_images": len({row.original_image_id for row in selected}),
            "split": split,
            "device": device,
            "seed": config.seed,
            "precision": config.precision,
            "tiers": {"low": tiers.low, "middle": tiers.middle, "high": tiers.high},
            "code_commit": resolved_commit,
            "checkpoint_sha256": checkpoint_sha256,
        },
    )

    tier_runs: dict[tuple[str, int], TierRun] = {}
    tier_metrics: dict[tuple[str, int], EvalMetrics] = {}
    failed_samples: set[str] = set()
    num_failures = 0
    num_cache_hits = 0

    for row in selected:
        try:
            image = load_image(row.image_path, config.image_size)
            mask = load_mask(row.mask_path, config.image_size)
            mask_sha256 = sha256_file(row.mask_path)
        except (OSError, ValueError) as exc:
            num_failures += 1
            failed_samples.add(row.sample_id)
            _append_jsonl(
                run_log_path,
                {"event": "fail", "sample_id": row.sample_id, "stage": "load"},
            )
            _append_jsonl(
                failures_path,
                {
                    "sample_id": row.sample_id,
                    "stage": "load",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            log(f"[fail] {row.sample_id} 读取失败：{exc}")
            continue

        for level in TIER_LEVELS:
            k = tiers.k(level)
            try:
                run = _run_tier(
                    row=row,
                    level=level,
                    k=k,
                    image=image,
                    mask=mask,
                    mask_sha256=mask_sha256,
                    images_dir=images_dir,
                    config=config,
                    cache=cache,
                    code_commit=resolved_commit,
                    checkpoint_sha256=checkpoint_sha256,
                    device=device,
                    sampling=sampling,
                    runner=runner,
                )
                metrics = evaluator(load_image(run.image_path, config.image_size), image, mask)
            except Exception as exc:  # noqa: BLE001 - 单样本失败必须可记录并继续
                num_failures += 1
                failed_samples.add(row.sample_id)
                _append_jsonl(
                    run_log_path,
                    {"event": "fail", "sample_id": row.sample_id, "level": level, "k": k},
                )
                _append_jsonl(
                    failures_path,
                    {
                        "sample_id": row.sample_id,
                        "level": level,
                        "k": k,
                        "stage": "tier",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                log(f"[fail] {row.sample_id} K={k}：{exc}")
                continue

            tier_runs[(row.sample_id, k)] = run
            tier_metrics[(row.sample_id, k)] = metrics
            num_cache_hits += int(run.cache_hit)
            _append_jsonl(
                run_log_path,
                {
                    "event": "tier",
                    "sample_id": row.sample_id,
                    "level": level,
                    "k": k,
                    "nfe": run.nfe,
                    "elapsed_seconds": round(run.elapsed_seconds, 4),
                    "peak_vram_mb": round(run.peak_vram_mb, 1),
                    "cache_hit": run.cache_hit,
                    "blank": run.blank,
                    "status": metrics.status,
                },
            )
            if metrics.status != "ok":
                num_failures += 1
                failed_samples.add(row.sample_id)
                _append_jsonl(
                    failures_path,
                    {
                        "sample_id": row.sample_id,
                        "level": level,
                        "k": k,
                        "stage": "evaluate",
                        "error": metrics.error or "评价失败",
                    },
                )
        log(f"[ok] {row.sample_id} 三档完成")

    usable = [row for row in selected if row.sample_id not in failed_samples]
    records: list[dict[str, Any]] = []
    policy_mean_k: dict[str, float] = {}

    if usable and len(usable) % 3 == 0:
        sample_ids = [row.sample_id for row in usable]
        features = [row.features for row in usable]
        for name, policy in resolved_policies.items():
            assignments = allocate_batch(sample_ids, features, policy)
            policy_mean_k[name] = mean_k(assignments)
            for row, assignment in zip(usable, assignments, strict=True):
                run = tier_runs[(row.sample_id, assignment.k)]
                metrics = tier_metrics[(row.sample_id, assignment.k)]
                records.append(
                    {
                        "sample_id": row.sample_id,
                        "original_image_id": row.original_image_id,
                        "split": row.split,
                        "mask_type": row.mask_type,
                        "target_area": row.target_area,
                        "policy": name,
                        "level": assignment.level,
                        "k": assignment.k,
                        "nfe": run.nfe,
                        "elapsed_seconds": run.elapsed_seconds,
                        "peak_vram_mb": run.peak_vram_mb,
                        "cache_hit": run.cache_hit,
                        "image": _relative(run.image_path, project_root),
                        "masked_psnr": metrics.masked_psnr,
                        "full_lpips": metrics.full_lpips,
                        "boundary_mae_3px": metrics.boundary_mae_3px,
                    }
                )
            _append_jsonl(
                run_log_path,
                {
                    "event": "policy",
                    "policy": name,
                    "num_samples": len(sample_ids),
                    "mean_k": round(policy_mean_k[name], 4),
                    "nfe_total": int(
                        sum(tier_runs[(row.sample_id, item.k)].nfe for row, item in zip(usable, assignments, strict=True))
                    ),
                },
            )
            log(f"[policy] {name} 平均 K={policy_mean_k[name]:.3f}（{len(sample_ids)} 个样本）")
    else:
        _append_jsonl(
            run_log_path,
            {
                "event": "policy_skip",
                "num_usable": len(usable),
                "reason": f"等预算分档要求可用样本数能被 3 整除，实际 {len(usable)}",
            },
        )
        log(f"[warn] 跳过策略分配：可用样本 {len(usable)} 个，无法满足等预算分块")

    frame = pd.DataFrame(records, columns=list(SAMPLE_FIELDS))
    frame.to_csv(output_dir / "samples.csv", index=False, encoding="utf-8")

    metric_frame = frame.loc[:, list(METRIC_FIELDS)] if not frame.empty else pd.DataFrame(
        columns=list(METRIC_FIELDS)
    )
    metric_frame.to_csv(metrics_dir / "per_sample.csv", index=False, encoding="utf-8")

    if frame.empty:
        per_image = pd.DataFrame(columns=["original_image_id", "policy", "num_samples", *METRIC_COLUMNS])
        summary = pd.DataFrame(
            columns=[
                "experiment_id", "policy", "num_samples", "mean_k", "nfe_total",
                *[f"mean_{name}" for name in METRIC_COLUMNS],
            ]
        )
    else:
        per_image = (
            frame.groupby(["original_image_id", "policy"], as_index=False)
            .agg(
                num_samples=("sample_id", "count"),
                **{name: (name, "mean") for name in METRIC_COLUMNS},
            )
            .sort_values(["policy", "original_image_id"])
        )
        summary = _summarize(records, SAMPLE_FIELDS)
        summary.insert(0, "experiment_id", config.experiment_id)
    per_image.to_csv(metrics_dir / "per_image.csv", index=False, encoding="utf-8")
    summary.to_csv(metrics_dir / "summary.csv", index=False, encoding="utf-8")

    report = ExperimentReport(
        experiment_id=config.experiment_id,
        output_dir=output_dir,
        metrics_dir=metrics_dir,
        num_images=len({row.original_image_id for row in selected}),
        num_samples=len(selected),
        num_tier_runs=len(tier_runs),
        num_cache_hits=num_cache_hits,
        num_failures=num_failures,
        num_records=len(records),
        policy_mean_k=policy_mean_k,
    )
    _append_jsonl(run_log_path, {"event": "end", **report.to_dict()})
    log(f"[report] {json.dumps(report.to_dict(), ensure_ascii=False)}")
    return report
