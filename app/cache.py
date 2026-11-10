"""结果缓存（FR-09）。

缓存键覆盖所有会影响输出的因素：代码 commit、权重 SHA-256、图片哈希、掩码哈希、
随机种子、精度、预算 ``K``、外层步数与全部采样参数。任何一项变化都会改变键，
从而不会命中旧结果。

缓存目录布局（``cache_dir/<key>.json`` + ``cache_dir/<key>.png``）：

- ``<key>.json`` 保存规范化后的 ``entry`` 与附加元数据；
- ``<key>.png`` 保存修复结果图。

命中判定同时校验文件名（键）与内嵌 ``entry``，二者不一致即视为未命中。
写入使用文件锁与「临时文件 + 原子替换」，任务中断不会留下半成品。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock

from app.data import save_result

#: 缓存条目规范化字段（PRD FR-09）
CACHE_ENTRY_FIELDS = (
    "code_commit",
    "checkpoint_sha256",
    "image_sha256",
    "mask_sha256",
    "seed",
    "precision",
    "budget_k",
    "outer_steps",
    "sampling",
)


def current_code_commit(project_root: str | Path | None = None) -> str:
    """返回当前 Git commit；不可用时返回 ``"unknown"``。"""
    cwd = str(project_root) if project_root is not None else str(Path(__file__).resolve().parents[1])
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def make_cache_entry(
    *,
    code_commit: str,
    checkpoint_sha256: str,
    image_sha256: str,
    mask_sha256: str,
    seed: int,
    precision: str,
    budget_k: int,
    outer_steps: int,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    """构造规范化缓存条目；``sampling`` 必须包含全部采样参数。"""
    return {
        "code_commit": str(code_commit),
        "checkpoint_sha256": str(checkpoint_sha256),
        "image_sha256": str(image_sha256),
        "mask_sha256": str(mask_sha256),
        "seed": int(seed),
        "precision": str(precision),
        "budget_k": int(budget_k),
        "outer_steps": int(outer_steps),
        "sampling": {str(key): sampling[key] for key in sorted(sampling)},
    }


def compute_cache_key(entry: dict[str, Any]) -> str:
    """对条目做规范化 JSON 后取 SHA-256。"""
    missing = [field for field in CACHE_ENTRY_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"缓存条目缺少字段：{missing}")
    canonical = json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheRecord:
    """一条命中的缓存记录。"""

    key: str
    image_path: Path
    meta: dict[str, Any]


class ResultCache:
    """基于文件的结果缓存。"""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.cache_dir / f"{key}.json", self.cache_dir / f"{key}.png"

    def lookup(self, entry: dict[str, Any]) -> CacheRecord | None:
        """按条目查缓存；文件缺失或内嵌条目不一致时返回 ``None``。"""
        key = compute_cache_key(entry)
        meta_path, image_path = self._paths(key)
        if not (meta_path.is_file() and image_path.is_file()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if meta.get("key") != key or meta.get("entry") != entry:
            return None
        return CacheRecord(key=key, image_path=image_path, meta=meta)

    def store(
        self,
        entry: dict[str, Any],
        image: torch.Tensor,
        extra: dict[str, Any] | None = None,
    ) -> CacheRecord:
        """写入缓存；使用文件锁与原子替换，保证中断后可恢复。"""
        key = compute_cache_key(entry)
        meta_path, image_path = self._paths(key)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with FileLock(str(self.cache_dir / f"{key}.lock")):
            tmp_png = image_path.with_name(f"{image_path.stem}.tmp.png")
            save_result(tmp_png, image)
            os.replace(tmp_png, image_path)

            payload = {"key": key, "entry": entry, "extra": extra or {}}
            tmp_json = meta_path.with_suffix(".json.tmp")
            tmp_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp_json, meta_path)

        meta = {"key": key, "entry": entry, "extra": extra or {}}
        return CacheRecord(key=key, image_path=image_path, meta=meta)
