"""``app.cache`` 的离线单元测试（FR-09 / PRD §12.1）。

覆盖要点：
- 缓存键覆盖全部字段：任一字段变化都必须换键；
- 条目缺字段即报错；
- 采样参数顺序不影响键（规范化）；
- 未命中返回 ``None``，写入后可命中，内嵌条目不符则不命中；
- ``current_code_commit`` 在仓库外也能安全返回字符串。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from app.cache import (
    CACHE_ENTRY_FIELDS,
    ResultCache,
    compute_cache_key,
    current_code_commit,
    make_cache_entry,
)


def _entry(**overrides: object) -> dict:
    base = {
        "code_commit": "abc123",
        "checkpoint_sha256": "c0ffee",
        "image_sha256": "img0",
        "mask_sha256": "mask0",
        "seed": 42,
        "precision": "fp32",
        "budget_k": 5,
        "outer_steps": 20,
        "sampling": {"n_steps": 5, "guidance": 0.0},
    }
    base.update(overrides)
    return make_cache_entry(**base)  # type: ignore[arg-type]


def _sample_image(size: int = 8) -> torch.Tensor:
    return torch.zeros((1, 3, size, size))


def test_cache_key_changes_with_every_field() -> None:
    baseline = _entry()
    base_key = compute_cache_key(baseline)

    changes = {
        "code_commit": "def456",
        "checkpoint_sha256": "beef",
        "image_sha256": "img1",
        "mask_sha256": "mask1",
        "seed": 7,
        "precision": "fp16",
        "budget_k": 8,
        "outer_steps": 10,
        "sampling": {"n_steps": 3, "guidance": 0.0},
    }
    assert set(changes) == set(CACHE_ENTRY_FIELDS)
    for field, value in changes.items():
        assert compute_cache_key(_entry(**{field: value})) != base_key, field


def test_cache_key_requires_all_fields() -> None:
    entry = _entry()
    entry.pop("mask_sha256")
    with pytest.raises(ValueError, match="缺少字段"):
        compute_cache_key(entry)


def test_cache_key_is_order_insensitive_for_sampling() -> None:
    a = compute_cache_key(_entry(sampling={"a": 1, "b": 2}))
    b = compute_cache_key(_entry(sampling={"b": 2, "a": 1}))
    assert a == b


def test_lookup_miss_on_empty_cache(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "results")
    assert cache.lookup(_entry()) is None


def test_store_then_lookup_hits(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "results")
    entry = _entry()

    stored = cache.store(entry, _sample_image(), extra={"note": "ok"})

    assert stored.image_path.is_file()
    hit = cache.lookup(entry)
    assert hit is not None
    assert hit.key == stored.key
    assert hit.image_path == stored.image_path
    assert hit.meta["extra"] == {"note": "ok"}


def test_lookup_rejects_mismatched_entry(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "results")
    entry = _entry()
    record = cache.store(entry, _sample_image())

    # 篡改内嵌条目：键名不变但内容不符 -> 视为未命中
    meta_path = record.image_path.with_suffix(".json")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["entry"]["seed"] = 999
    meta_path.write_text(json.dumps(payload), encoding="utf-8")

    assert cache.lookup(entry) is None


def test_lookup_rejects_missing_image_file(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "results")
    entry = _entry()
    record = cache.store(entry, _sample_image())

    record.image_path.unlink()

    assert cache.lookup(entry) is None


def test_different_entries_do_not_collide(tmp_path: Path) -> None:
    cache = ResultCache(tmp_path / "results")
    cache.store(_entry(budget_k=2), _sample_image())
    assert cache.lookup(_entry(budget_k=8)) is None


def test_current_code_commit_returns_string(tmp_path: Path) -> None:
    value = current_code_commit(tmp_path)
    assert isinstance(value, str)
    assert value  # 不可用时为 "unknown"，但始终非空
