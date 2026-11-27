import numpy as np
import pytest
from PIL import Image

from app.ui import UIValidationError, prepare_input


def test_brush_uses_alpha_and_preserves_background():
    background = Image.new("RGB", (256, 256), (20, 80, 120))
    layer = np.zeros((256, 256, 4), dtype=np.uint8)
    layer[50:100, 60:110] = [255, 255, 255, 255]
    original, mask, preview = prepare_input(background, Image.fromarray(layer), "brush")
    assert np.array_equal(np.asarray(original), np.asarray(background))
    assert np.count_nonzero(np.asarray(mask)) == 2500
    assert np.asarray(preview)[70, 70].tolist() == [0, 0, 0]
    assert np.asarray(preview)[0, 0].tolist() == [20, 80, 120]


def test_upload_mask_overrides_brush_and_resizes():
    background = Image.new("RGB", (512, 512))
    uploaded = np.zeros((128, 128), dtype=np.uint8)
    uploaded[:64] = 255
    _, mask, _ = prepare_input(background, Image.fromarray(uploaded), "upload")
    assert np.count_nonzero(np.asarray(mask)) == 256 * 128


def test_all_missing_mask_rejected():
    background = Image.new("RGB", (256, 256))
    with pytest.raises(UIValidationError):
        prepare_input(background, Image.new("L", (256, 256), 255), "upload")


def test_empty_brush_is_empty_mask():
    background = Image.new("RGB", (256, 256))
    _, mask, _ = prepare_input(background, Image.new("RGBA", (256, 256), (0, 0, 0, 0)), "brush")
    assert not np.asarray(mask).any()
    _, mask_none, _ = prepare_input(background, None, "brush")
    assert not np.asarray(mask_none).any()
