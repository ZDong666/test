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
