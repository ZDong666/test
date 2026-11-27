import gradio as gr
import numpy as np
import pytest
from PIL import Image

from app.ui import prepare_input


def test_brush_uses_alpha_and_preserves_background():
    background = Image.new("RGB", (256, 256), (20, 80, 120))
    layer = np.zeros((256, 256, 4), dtype=np.uint8)
    layer[50:100, 60:110] = [255, 255, 255, 255]
    editor = {"background": background, "layers": [Image.fromarray(layer)]}
    original, mask, preview = prepare_input(editor, None, "画笔涂抹")
    assert np.array_equal(np.asarray(original), np.asarray(background))
    assert np.count_nonzero(np.asarray(mask)) == 2500
    assert np.asarray(preview)[70, 70].tolist() == [0, 0, 0]
    assert np.asarray(preview)[0, 0].tolist() == [20, 80, 120]


def test_upload_mask_overrides_brush_and_resizes():
    editor = {"background": Image.new("RGB", (512, 512)), "layers": []}
    uploaded = np.zeros((128, 128), dtype=np.uint8)
    uploaded[:64] = 255
    _, mask, _ = prepare_input(editor, Image.fromarray(uploaded), "上传掩码")
    assert np.count_nonzero(np.asarray(mask)) == 256 * 128


def test_all_missing_mask_rejected():
    editor = {"background": Image.new("RGB", (256, 256)), "layers": []}
    with pytest.raises(gr.Error):
        prepare_input(editor, Image.new("L", (256, 256), 255), "上传掩码")


def test_empty_brush_is_empty_mask():
    editor = {"background": Image.new("RGB", (256, 256)), "layers": []}
    _, mask, _ = prepare_input(editor, None, "画笔涂抹")
    assert not np.asarray(mask).any()


def test_workspace_recovers_controls_and_clears_results_on_error(monkeypatch):
    from app import ui

    def fail(*args):
        raise gr.Error("输入不完整")

    monkeypatch.setattr(ui, "run_repair", fail)
    task = ui.workspace_repair(None, None, "画笔涂抹", 5, 42)
    waiting = next(task)
    assert all(update["interactive"] is False for update in waiting[8:])
    failed = next(task)
    assert all(update["interactive"] is True for update in failed[8:])
    assert all(failed[i] is None for i in (0, 1, 2, 6, 7))
    assert failed[4]["value"] is None and failed[4]["interactive"] is False
    assert failed[5]["value"] is None and failed[5]["interactive"] is False
    with pytest.raises(gr.Error):
        next(task)


def test_workspace_comparison_uses_the_completed_run(monkeypatch, tmp_path):
    from app import ui

    result = str(tmp_path / "result.png")
    monkeypatch.setattr(ui, "run_repair", lambda *args: ("preview", "mask", result, "完成", result, "config"))
    updates = list(ui.workspace_repair(None, None, "画笔涂抹", 5, 42))
    completed = updates[-1]
    assert completed[6] == (str(tmp_path / "input_preview.png"), result)
    assert ui.comparison_pair(completed[7], "原图") == (str(tmp_path / "original.png"), result)
    assert all(update["interactive"] is True for update in completed[8:])
    cleared = ui.clear_workspace()
    assert all(cleared[i] is None for i in (0, 4, 5, 6, 7))
    assert cleared[2]["value"] is None and cleared[2]["interactive"] is False
    assert cleared[3]["value"] is None and cleared[3]["interactive"] is False
