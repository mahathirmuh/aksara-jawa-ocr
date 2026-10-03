"""Test shape — stride lebar model diverifikasi pada output, bukan diasumsikan."""

import pytest
import torch

from src.dataset import DOWNSAMPLE, H
from src.model import CRNN

N_CLASSES = 50


@pytest.fixture(scope="module")
def model() -> CRNN:
    torch.manual_seed(0)
    return CRNN(N_CLASSES).eval()


@pytest.mark.parametrize("width", [4, 97, 400, 1601])
def test_output_length_equals_width_over_downsample(model, width):
    with torch.no_grad():
        out = model(torch.zeros(2, 1, H, width))
    assert out.shape == (2, width // DOWNSAMPLE, N_CLASSES)


def test_declared_stride_matches_dataset_downsample():
    assert CRNN.WIDTH_STRIDE == DOWNSAMPLE


def test_height_is_compressed_to_feature_rows(model):
    with torch.no_grad():
        feats = model.cnn(torch.zeros(1, 1, H, 64))
    assert feats.shape[2] == model.feat_height == H // 32


def test_rejects_wrong_height(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, 1, 32, 100))


def test_accepts_per_sample_lengths(model):
    x = torch.rand(3, 1, H, 400)
    lengths = torch.tensor([100, 37, 1])
    with torch.no_grad():
        out = model(x, lengths)
    assert out.shape == (3, 100, N_CLASSES)
    assert torch.isfinite(out).all()


def test_parameter_count_near_plan(model):
    n = sum(p.numel() for p in model.parameters())
    assert 3_000_000 < n < 7_000_000, n
