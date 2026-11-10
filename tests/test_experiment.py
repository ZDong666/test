"""``app.experiment`` 的离线单元测试（FR-05 / FR-09，PRD §7.5 / §7.6 / §12.1 / §12.2）。

用假 ``runner`` 与假 ``evaluator`` 替换真实扩散推理与 LPIPS，因此完全离线，
不依赖 GPU、模型权重与评估权重。
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.budget import BudgetTiers
from app.config import ExperimentConfig
from app.data import sha256_file
from app.experiment import (
    MANIFEST_FIELDS,
    SAMPLE_FIELDS,
    SampleRow,
    load_manifest,
    run_experiment,
    select_samples,
)
from app.metrics import EvalMetrics
from app.restore import RestoreResult

TIERS = BudgetTiers()  # 2 / 5 / 8
MASK_TYPES = ("center", "scratch", "scatter")
POLICIES = ("B1", "B2", "B3", "M1", "A1", "C1")
#: 让三类掩码在特征上有区分度，便于策略排序
BOUNDARY_BY_TYPE = {"center": 4.0, "scratch": 50.0, "scatter": 27.0}


def _silent(_: str) -> None:
    """测试中静默日志。"""


# --------------------------------------------------------------------------- #
# 假数据与假执行器
# --------------------------------------------------------------------------- #


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path, tag: str) -> None:
    """写入与该 ``tag`` 一一对应的掩码。

    缓存键包含掩码 SHA-256（FR-09），故不同样本必须字节不同；否则不同样本会被
    判为同一次推理而互相复用，使「按样本统计」的断言失真。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = np.frombuffer(hashlib.sha256(tag.encode("utf-8")).digest(), dtype=np.uint8)
    array = ((np.tile(digest, 2)[:64] % 2) * 255).astype(np.uint8).reshape(8, 8)
    array[0, 0] = 255  # 至少一个缺失像素
    array[7, 7] = 0  # 至少一个已知像素，避免整图缺失
    Image.fromarray(array, mode="L").save(path)


def _build_dataset(
    root: Path,
    *,
    image_ids: tuple[str, ...] = ("img0000",),
    splits: tuple[str, ...] | None = None,
    mask_types: tuple[str, ...] = MASK_TYPES,
    areas: tuple[float, ...] = (0.30,),
) -> Path:
    """在 ``root`` 下生成图片、掩码与 data/manifest.csv，返回清单路径。"""
    split_list = splits if splits is not None else ("dev",) * len(image_ids)
    image_dir = root / "data" / "samples" / "original"
    mask_dir = root / "data" / "masks"
    rows = []
    for index, name in enumerate(image_ids):
        image_path = image_dir / f"{name}.png"
        _write_image(image_path, 40 + index * 10)
        image_sha = sha256_file(image_path)
        for mask_type in mask_types:
            for area in areas:
                tag = f"{mask_type}{round(area * 100)}"
                mask_path = mask_dir / f"{name}_{tag}_mask.png"
                _write_mask(mask_path, f"{name}_{tag}")
                rows.append(
                    {
                        "sample_id": f"{name}_{tag}",
                        "original_image_id": name,
                        "split": split_list[index],
                        "mask_type": mask_type,
                        "target_area": f"{area:.4f}",
                        "actual_area": f"{area:.6f}",
                        "boundary_norm": f"{BOUNDARY_BY_TYPE[mask_type]:.6f}",
                        "seed": 42 + index,
                        "image_path": image_path.relative_to(root).as_posix(),
                        "mask_path": mask_path.relative_to(root).as_posix(),
                        "sha256": image_sha,
                    }
                )
    manifest_path = root / "data" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


class _FakeRunner:
    """确定性假推理：已知区域置零、缺失区域留原值，并记录调用次数。"""

    def __init__(self, fail_calls: set[int] | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self.fail_calls = fail_calls or set()

    def __call__(self, image: torch.Tensor, mask: torch.Tensor, config) -> RestoreResult:
        index = len(self.calls)
        self.calls.append((config.n_steps, config.seed))
        if index in self.fail_calls:
            raise RuntimeError(f"模拟推理失败（第 {index} 次调用）")
        output = image * (1.0 - mask)
        return RestoreResult(
            image=output,
            nfe=config.n_steps * 10,
            elapsed_seconds=0.5,
            peak_vram_mb=100.0,
            seed=config.seed,
            config=config,
        )


def _fake_evaluator(result, reference, mask) -> EvalMetrics:
    error = float((result - reference).abs().mean().item())
    return EvalMetrics(
        masked_psnr=20.0 - error,
        full_lpips=error,
        boundary_mae_3px=error,
        status="ok",
    )


def _config(root: Path, *, experiment_id: str = "exp_unit", precision: str = "fp32") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=experiment_id,
        method="lanpaint",
        policy="M1",
        data_manifest=root / "data" / "manifest.csv",
        checkpoint=root / "checkpoints" / "missing.pt",
        output_dir=root / "results" / "raw",
        cache_dir=root / "cache",
        seed=42,
        precision=precision,
    )


def _run(root: Path, config: ExperimentConfig, runner: _FakeRunner, **kwargs):
    return run_experiment(
        config,
        project_root=root,
        runner=runner,
        evaluator=_fake_evaluator,
        code_commit="testcommit",
        log=_silent,
        **kwargs,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- #
# 清单读取与筛选
# --------------------------------------------------------------------------- #


def test_load_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    _build_dataset(tmp_path, image_ids=("img0000",), areas=(0.15, 0.30))
    rows = load_manifest(tmp_path / "data" / "manifest.csv", project_root=tmp_path)

    assert len(rows) == 6
    assert all(isinstance(row, SampleRow) for row in rows)
    assert rows[0].image_path == tmp_path / "data" / "samples" / "original" / "img0000.png"
    assert rows[0].mask_path.is_file()
    assert rows[0].features.area_ratio == pytest.approx(rows[0].actual_area)


def test_load_manifest_missing_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("sample_id,split\ns0,dev\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少字段"):
        load_manifest(path, project_root=tmp_path)


def test_select_samples_filters_split_type_area_and_image_window(tmp_path: Path) -> None:
    _build_dataset(
        tmp_path,
        image_ids=("img0000", "img0001"),
        splits=("dev", "test"),
        areas=(0.15, 0.30),
    )
    rows = load_manifest(tmp_path / "data" / "manifest.csv", project_root=tmp_path)

    dev = select_samples(rows, split="dev")
    assert {row.original_image_id for row in dev} == {"img0000"}
    assert len(dev) == 6

    center_only = select_samples(rows, split="dev", mask_types=["center"])
    assert len(center_only) == 2
    assert all(row.mask_type == "center" for row in center_only)

    small_area = select_samples(rows, split="dev", target_areas=[0.15])
    assert len(small_area) == 3
    assert all(row.target_area == pytest.approx(0.15) for row in small_area)

    assert select_samples(rows, split="test", image_offset=1) == []
    in_window = select_samples(rows, split="test", image_offset=0, image_limit=1)
    assert {row.original_image_id for row in in_window} == {"img0001"}


# --------------------------------------------------------------------------- #
# 批量执行与输出结构
# --------------------------------------------------------------------------- #


def test_run_experiment_writes_prd_output_structure(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    config = _config(tmp_path)
    runner = _FakeRunner()

    report = _run(tmp_path, config, runner)

    assert report.status == "ok"
    assert report.num_samples == 3
    assert report.num_tier_runs == 9
    assert report.num_cache_hits == 0
    assert len(runner.calls) == 9

    output_dir = tmp_path / "results" / "raw" / config.experiment_id
    for name in ("config.yaml", "manifest.csv", "samples.csv", "run_log.jsonl", "failures.jsonl"):
        assert (output_dir / name).is_file(), name
    assert len(list((output_dir / "images").glob("*.png"))) == 9

    metrics_dir = tmp_path / "results" / "metrics" / config.experiment_id
    for name in ("per_sample.csv", "per_image.csv", "summary.csv"):
        assert (metrics_dir / name).is_file(), name

    samples = _read_csv(output_dir / "samples.csv")
    assert len(samples) == 3 * len(POLICIES)  # 3 个样本 × 6 条策略
    assert list(samples[0]) == list(SAMPLE_FIELDS)
    assert {row["policy"] for row in samples} == set(POLICIES)

    assert "experiment_id: exp_unit" in (output_dir / "config.yaml").read_text(encoding="utf-8")
    assert _read_csv(output_dir / "manifest.csv")[0]["sample_id"] == "img0000_center30"
    assert _read_jsonl(output_dir / "run_log.jsonl")[0]["event"] == "start"


def test_equal_budget_policies_keep_middle_mean_k(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    config = _config(tmp_path)

    report = _run(tmp_path, config, _FakeRunner())

    assert report.policy_mean_k["B1"] == pytest.approx(TIERS.low)
    assert report.policy_mean_k["B2"] == pytest.approx(TIERS.middle)
    assert report.policy_mean_k["B3"] == pytest.approx(TIERS.high)
    for policy in ("M1", "A1", "C1"):
        assert report.policy_mean_k[policy] == pytest.approx(TIERS.middle)

    summary = _read_csv(tmp_path / "results" / "metrics" / config.experiment_id / "summary.csv")
    assert {row["policy"] for row in summary} == set(POLICIES)
    assert all(int(row["num_samples"]) == 3 for row in summary)
    assert all(float(row["mean_k"]) in {2.0, 5.0, 8.0} for row in summary)


def test_adaptive_policies_use_all_three_tiers(tmp_path: Path) -> None:
    """3 个样本按分数排序后必须恰好分到低 / 中 / 高三档各一次（FR-05）。"""
    _build_dataset(tmp_path)
    config = _config(tmp_path)
    _run(tmp_path, config, _FakeRunner())

    samples = _read_csv(tmp_path / "results" / "raw" / config.experiment_id / "samples.csv")
    expected = sorted([TIERS.low, TIERS.middle, TIERS.high])
    for policy in ("M1", "A1", "C1"):
        levels = sorted(int(row["k"]) for row in samples if row["policy"] == policy)
        assert levels == expected


def test_all_policies_reuse_the_same_tier_outputs(tmp_path: Path) -> None:
    """六条策略不新增推理：全部只引用同一批三档输出（PRD §7.6）。"""
    _build_dataset(tmp_path)
    config = _config(tmp_path)
    report = _run(tmp_path, config, _FakeRunner())

    assert report.num_tier_runs == 3 * 3  # 3 个样本 × 3 档，而不随策略数放大
    samples = _read_csv(tmp_path / "results" / "raw" / config.experiment_id / "samples.csv")
    referenced = {row["image"] for row in samples}
    assert len(referenced) == report.num_tier_runs


def test_cache_hit_reuses_previous_results(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    config = _config(tmp_path)

    first_runner = _FakeRunner()
    _run(tmp_path, config, first_runner)
    metrics_path = tmp_path / "results" / "metrics" / config.experiment_id / "per_sample.csv"
    snapshot = metrics_path.read_text(encoding="utf-8")

    second_runner = _FakeRunner()
    report = _run(tmp_path, config, second_runner)

    assert len(first_runner.calls) == 9
    assert second_runner.calls == []
    assert report.num_cache_hits == 9
    assert report.num_failures == 0
    # 缓存结果与直接运行一致（PRD §12.2）
    assert metrics_path.read_text(encoding="utf-8") == snapshot


def test_changed_seed_invalidates_cache(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    config = _config(tmp_path)
    _run(tmp_path, config, _FakeRunner())

    other_runner = _FakeRunner()
    _run(tmp_path, replace(config, seed=7), other_runner)

    assert len(other_runner.calls) == 9  # 缓存键含种子，换种子必须重新推理（FR-09）


def test_failed_sample_is_recorded_and_run_continues(tmp_path: Path) -> None:
    _build_dataset(tmp_path)  # 1 张原图 × 3 类掩码 = 3 个样本
    config = _config(tmp_path)
    runner = _FakeRunner(fail_calls={3, 4, 5})  # 第 2 个样本（scratch）三档全部失败

    report = _run(tmp_path, config, runner)

    assert report.status == "partial"
    assert report.num_failures == 3
    assert len(runner.calls) == 9  # 失败不阻断后续样本

    output_dir = tmp_path / "results" / "raw" / config.experiment_id
    failures = _read_jsonl(output_dir / "failures.jsonl")
    assert {item["sample_id"] for item in failures} == {"img0000_scratch30"}
    assert all(item["stage"] == "tier" for item in failures)

    images = sorted(path.name for path in (output_dir / "images").glob("*.png"))
    assert len(images) == 6  # 其余两个样本各 3 档
    assert not any(name.startswith("img0000_scratch30") for name in images)

    # 可用样本 2 个，无法满足等预算分块 -> 只记录跳过原因，不写策略记录
    assert _read_csv(output_dir / "samples.csv") == []
    assert any(item["event"] == "policy_skip" for item in _read_jsonl(output_dir / "run_log.jsonl"))


def test_precision_other_than_fp32_rejected(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="fp32"):
        _run(tmp_path, _config(tmp_path, precision="fp16"), _FakeRunner())


def test_no_selected_samples_raises(tmp_path: Path) -> None:
    _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="没有样本"):
        _run(tmp_path, _config(tmp_path), _FakeRunner(), split="test")
