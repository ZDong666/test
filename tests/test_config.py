"""``app.config`` 的离线单元测试（FR-09 / PRD §8.2）。

覆盖要点：
- 完整 YAML 能被解析，相对路径相对项目根解析；
- 缺必填字段 / 非法预算档位 / 非法精度 / 开启动态调度 都要报错；
- 文件不存在报错；
- ``BudgetConfig.to_tiers()`` 复用 :class:`app.budget.BudgetTiers` 的校验。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.budget import BudgetTiers
from app.config import PROJECT_ROOT, BudgetConfig, ExperimentConfig, load_config

FULL_YAML = """
experiment_id: exp01
method: lanpaint
policy: M1
budget:
  low: 2
  middle: 5
  high: 8
outer_steps: 20
batch_size: 3
image_size: 256
seed: 7
precision: fp16
data_manifest: data/manifest.csv
checkpoint: checkpoints/256x256_diffusion_uncond.pt
checkpoint_sha256: deadbeef
output_dir: results/raw
cache_dir: data/_cache/results
enable_dynamic_stopping: false
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_full_config(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, FULL_YAML))

    assert config.experiment_id == "exp01"
    assert config.method == "lanpaint"
    assert config.policy == "M1"
    assert config.budget == BudgetConfig(low=2, middle=5, high=8)
    assert config.batch_size == 3
    assert config.seed == 7
    assert config.precision == "fp16"
    assert config.checkpoint_sha256 == "deadbeef"
    assert config.enable_dynamic_stopping is False
    # 相对路径相对项目根解析
    assert config.data_manifest == PROJECT_ROOT / "data" / "manifest.csv"
    assert config.checkpoint == PROJECT_ROOT / "checkpoints" / "256x256_diffusion_uncond.pt"
    assert config.cache_dir == PROJECT_ROOT / "data" / "_cache" / "results"


def test_load_config_defaults(tmp_path: Path) -> None:
    text = "experiment_id: e\nmethod: m\npolicy: B1\n"
    config = load_config(_write(tmp_path, text))

    assert config.budget == BudgetConfig()
    assert config.outer_steps == 20
    assert config.image_size == 256
    assert config.precision == "fp32"
    assert config.checkpoint_sha256 == ""


def test_missing_required_field_raises(tmp_path: Path) -> None:
    text = "experiment_id: e\nmethod: m\n"  # 缺 policy
    with pytest.raises(KeyError, match="policy"):
        load_config(_write(tmp_path, text))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_config_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="映射"):
        load_config(_write(tmp_path, "- just\n- a\n- list\n"))


def test_invalid_budget_tiers_raise(tmp_path: Path) -> None:
    text = FULL_YAML.replace("  high: 8", "  high: 7")  # 2 + 7 != 2 * 5
    with pytest.raises(ValueError, match="等预算"):
        load_config(_write(tmp_path, text))


def test_invalid_precision_raises(tmp_path: Path) -> None:
    text = FULL_YAML.replace("precision: fp16", "precision: bf16")
    with pytest.raises(ValueError, match="precision"):
        load_config(_write(tmp_path, text))


def test_dynamic_stopping_not_supported(tmp_path: Path) -> None:
    text = FULL_YAML.replace("enable_dynamic_stopping: false", "enable_dynamic_stopping: true")
    with pytest.raises(ValueError, match="动态调度"):
        load_config(_write(tmp_path, text))


def test_experiment_config_validates_itself() -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        ExperimentConfig(experiment_id="", method="m", policy="B1")
    with pytest.raises(ValueError, match="outer_steps"):
        ExperimentConfig(experiment_id="e", method="m", policy="B1", outer_steps=0)


def test_budget_config_to_tiers() -> None:
    assert BudgetConfig().to_tiers() == BudgetTiers(low=2, middle=5, high=8)


def test_to_dict_is_serializable() -> None:
    config = ExperimentConfig(experiment_id="e", method="m", policy="B1")
    payload = config.to_dict()
    assert payload["experiment_id"] == "e"
    assert isinstance(payload["data_manifest"], str)
    assert payload["budget"] == {"low": 2, "middle": 5, "high": 8}
