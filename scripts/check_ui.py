"""启动临时本地服务，通过 HTTP API 验证真实修复和输出文件。"""
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".cache" / "cuda"))

from app.ui import app


class QuietServer(uvicorn.Server):
    def install_signal_handlers(self):
        return None


def local_path(url):
    relative = str(url).removeprefix("/runs/")
    return ROOT / "results" / "ui" / relative


def wait_for_server(url):
    for _ in range(100):
        try:
            if httpx.get(url, timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise RuntimeError("临时服务启动失败")


def main():
    config = uvicorn.Config(app, host="127.0.0.1", port=7862, log_level="warning")
    server = QuietServer(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        url = "http://127.0.0.1:7862"
        wait_for_server(url)
        with (ROOT / "data" / "manifest.csv").open(encoding="utf-8", newline="") as stream:
            row = next(row for row in csv.DictReader(stream) if row["sample_id"] == "img0010_center30")
        image_path = ROOT / row["image_path"]
        mask_path = ROOT / row["mask_path"]
        with image_path.open("rb") as image_file, mask_path.open("rb") as mask_file:
            response = httpx.post(
                f"{url}/api/repair",
                files={
                    "image": ("source.png", image_file, "image/png"),
                    "mask": ("mask.png", mask_file, "image/png"),
                },
                data={"mode": "upload", "budget": "5", "seed": "42"},
                timeout=600,
            )
        response.raise_for_status()
        result = response.json()
        print("UI_CLIENT_RESULT", json.dumps(result, ensure_ascii=False), flush=True)
        result_path = local_path(result["result_url"])
        assert result_path.is_file(), result_path
        report = json.loads(local_path(result["config_url"]).read_text(encoding="utf-8"))
        assert report["nfe"] > 0
        print("UI_REAL_GPU_PASS", json.dumps(report, ensure_ascii=True), flush=True)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    main()
