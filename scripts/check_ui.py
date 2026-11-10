"""启动临时本地服务，通过 Gradio 客户端验证真实修复和下载文件。"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["CUDA_CACHE_PATH"] = str(ROOT / ".cache" / "cuda")

from gradio_client import Client, handle_file

from app.ui import build_ui


def main():
    demo = build_ui()
    try:
        _, url, _ = demo.launch(server_name="127.0.0.1", server_port=7862, share=False, prevent_thread_lock=True)
        client = Client(url, verbose=False)
        with (ROOT / "data" / "manifest.csv").open(encoding="utf-8", newline="") as stream:
            row = next(row for row in csv.DictReader(stream) if row["sample_id"] == "img0010_center30")
        image_path = ROOT / row["image_path"]
        mask_path = ROOT / row["mask_path"]
        result = client.predict(
            {"background": handle_file(str(image_path)), "layers": [], "composite": handle_file(str(image_path))},
            handle_file(str(mask_path)), "上传掩码", 5, 42, api_name="/repair",
        )
        print("UI_CLIENT_RESULT", result, flush=True)
        for index in (2, 4, 5):
            assert Path(result[index]).is_file(), result[index]
        report = json.loads(Path(result[5]).read_text(encoding="utf-8"))
        assert report["nfe"] > 0
        print("UI_REAL_GPU_PASS", json.dumps(report, ensure_ascii=True), flush=True)
    finally:
        demo.close()


if __name__ == "__main__":
    main()
