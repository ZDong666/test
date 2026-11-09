"""Read-only environment audit for the LanPaint project."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


MODULES = {
    "core": [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "PIL",
    ],
    "experiment": [
        "scipy",
        "skimage",
        "lpips",
        "pandas",
        "matplotlib",
        "tqdm",
        "yaml",
    ],
    "application": ["gradio", "pytest", "ruff"],
}


def command_version(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if not path:
        return {"available": False, "path": None}
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        output = (result.stdout or result.stderr).splitlines()
        version = output[0].strip() if output else ""
    except (OSError, subprocess.SubprocessError):
        version = ""
    return {"available": True, "path": path, "version": version}


def total_ram_gb() -> float | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return round(status.ullTotalPhys / 1024**3, 2)


def check_modules() -> dict[str, dict[str, bool]]:
    return {
        group: {name: importlib.util.find_spec(name) is not None for name in names}
        for group, names in MODULES.items()
    }


def check_torch() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"available": False}

    result: dict[str, Any] = {"available": True}
    try:
        import torch

        result["version"] = torch.__version__
        result["compiled_cuda"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            return result

        result["device_count"] = torch.cuda.device_count()
        result["device_name"] = torch.cuda.get_device_name(0)
        result["capability"] = list(torch.cuda.get_device_capability(0))
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        result["vram_free_gb"] = round(free_bytes / 1024**3, 2)
        result["vram_total_gb"] = round(total_bytes / 1024**3, 2)

        x = torch.randn((256, 256), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        result["matmul_256_ok"] = tuple(y.shape) == (256, 256)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def collect_report() -> dict[str, Any]:
    disk = shutil.disk_usage(Path.cwd())
    modules = check_modules()
    torch_report = check_torch()
    missing_core = [name for name, ok in modules["core"].items() if not ok]
    checks = {
        "core_modules_present": not missing_core,
        "cuda_available": bool(torch_report.get("cuda_available")),
        "git_available": command_version("git")["available"],
        "ffmpeg_available": command_version("ffmpeg")["available"],
    }

    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "system": {
            "logical_cpus": os.cpu_count(),
            "total_ram_gb": total_ram_gb(),
            "cwd_free_gb": round(disk.free / 1024**3, 2),
        },
        "tools": {
            name: command_version(name)
            for name in ["git", "nvidia-smi", "curl", "ffmpeg"]
        },
        "modules": modules,
        "missing_core_modules": missing_core,
        "torch": torch_report,
        "checks": checks,
    }


def print_human(report: dict[str, Any]) -> None:
    python = report["python"]
    system = report["system"]
    print("LanPaint environment audit")
    print(f"Python: {python['version']} ({python['executable']})")
    print(f"Platform: {python['platform']}")
    print(f"CPU/threads: {system['logical_cpus']}")
    print(f"RAM: {system['total_ram_gb']} GB")
    print(f"Free disk in cwd: {system['cwd_free_gb']} GB")

    print("\nTools:")
    for name, details in report["tools"].items():
        state = details.get("version") or details.get("path") or "missing"
        print(f"- {name}: {state}")

    print("\nModules:")
    for group, modules in report["modules"].items():
        ok_count = sum(modules.values())
        print(f"- {group}: {ok_count}/{len(modules)} present")
        missing = [name for name, ok in modules.items() if not ok]
        if missing:
            print(f"  missing: {', '.join(missing)}")

    print("\nPyTorch/CUDA:")
    if not report["torch"].get("available"):
        print("- torch is not installed")
    else:
        for key, value in report["torch"].items():
            print(f"- {key}: {value}")

    print("\nChecks:")
    for name, ok in report["checks"].items():
        print(f"- {name}: {'PASS' if ok else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when core modules or CUDA are unavailable.",
    )
    args = parser.parse_args()

    report = collect_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)

    if args.strict and not all(
        [report["checks"]["core_modules_present"], report["checks"]["cuda_available"]]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
