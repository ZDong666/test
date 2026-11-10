"""实验配置读取与校验（FR-09 / PRD §8.2）。

YAML 配置至少包含：``experiment_id``、``method``、``policy``、``budget``、
``outer_steps``、``batch_size``、``image_size``、``seed``、``precision``、
``data_manifest``、``checkpoint``、``checkpoint_sha256``、``output_dir``、
``cache_dir``、``enable_dynamic_stopping``。

相对路径一律相对项目根解析，便于在不同工作目录下复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.budget import BudgetTiers

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRECISION_FP32 = "fp32"
PRECISION_FP16 = "fp16"
SUPPORTED_PRECISIONS = (PRECISION_FP32, PRECISION_FP16)


@dataclass(frozen=True)
class BudgetConfig:
    """预算档位配置（PRD §7.3）。"""

    low: int = 2
    middle: int = 5
    high: int = 8

    def to_tiers(self) -> BudgetTiers:
        return BudgetTiers(low=self.low, middle=self.middle, high=self.high)


@dataclass(frozen=True)
class ExperimentConfig:
    """一次实验的完整配置。"""

    experiment_id: str
    method: str
    policy: str
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    outer_steps: int = 20
    batch_size: int = 1
    image_size: int = 256
    seed: int = 42
    precision: str = PRECISION_FP32
    data_manifest: Path = PROJECT_ROOT / "data" / "manifest.csv"
    checkpoint: Path = PROJECT_ROOT / "checkpoints" / "256x256_diffusion_uncond.pt"
    checkpoint_sha256: str = ""
    output_dir: Path = PROJECT_ROOT / "results" / "raw"
    cache_dir: Path = PROJECT_ROOT / "data" / "_cache" / "results"
    enable_dynamic_stopping: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id 不能为空")
        if not self.method:
            raise ValueError("method 不能为空")
        if not self.policy:
            raise ValueError("policy 不能为空")
        self.budget.to_tiers()  # 触发档位校验（严格递增且 low + high == 2 * middle）
        if self.outer_steps < 1:
            raise ValueError(f"outer_steps 必须 >= 1，收到 {self.outer_steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，收到 {self.batch_size}")
        if self.image_size < 1:
            raise ValueError(f"image_size 必须 >= 1，收到 {self.image_size}")
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"precision 需为 {SUPPORTED_PRECISIONS} 之一，收到 {self.precision!r}"
            )
        if self.enable_dynamic_stopping:
            raise ValueError("第一版不支持时间步级动态调度（PRD §3.2），enable_dynamic_stopping 必须为 false")

    def to_dict(self) -> dict[str, Any]:
        """转回可序列化的字典（路径转字符串）。"""
        return {
            "experiment_id": self.experiment_id,
            "method": self.method,
            "policy": self.policy,
            "budget": {"low": self.budget.low, "middle": self.budget.middle, "high": self.budget.high},
            "outer_steps": self.outer_steps,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            "seed": self.seed,
            "precision": self.precision,
            "data_manifest": str(self.data_manifest),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir),
            "enable_dynamic_stopping": self.enable_dynamic_stopping,
        }


def _resolve_path(value: Any, default: Path) -> Path:
    if value is None:
        return default
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise KeyError(f"配置缺少必填字段：{key}")
    return raw[key]


def _parse_budget(value: Any) -> BudgetConfig:
    if value is None:
        return BudgetConfig()
    if not isinstance(value, dict):
        raise TypeError(f"budget 需为映射（含 low/middle/high），收到 {type(value).__name__}")
    return BudgetConfig(
        low=int(value.get("low", 2)),
        middle=int(value.get("middle", 5)),
        high=int(value.get("high", 8)),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """读取 YAML 配置并校验。"""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件：{config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"配置文件内容应为映射，实际为 {type(raw).__name__}")
    return ExperimentConfig(
        experiment_id=str(_require(raw, "experiment_id")),
        method=str(_require(raw, "method")),
        policy=str(_require(raw, "policy")),
        budget=_parse_budget(raw.get("budget")),
        outer_steps=int(raw.get("outer_steps", 20)),
        batch_size=int(raw.get("batch_size", 1)),
        image_size=int(raw.get("image_size", 256)),
        seed=int(raw.get("seed", 42)),
        precision=str(raw.get("precision", PRECISION_FP32)),
        data_manifest=_resolve_path(raw.get("data_manifest"), PROJECT_ROOT / "data" / "manifest.csv"),
        checkpoint=_resolve_path(raw.get("checkpoint"), PROJECT_ROOT / "checkpoints" / "256x256_diffusion_uncond.pt"),
        checkpoint_sha256=str(raw.get("checkpoint_sha256", "")),
        output_dir=_resolve_path(raw.get("output_dir"), PROJECT_ROOT / "results" / "raw"),
        cache_dir=_resolve_path(raw.get("cache_dir"), PROJECT_ROOT / "data" / "_cache" / "results"),
        enable_dynamic_stopping=bool(raw.get("enable_dynamic_stopping", False)),
    )
