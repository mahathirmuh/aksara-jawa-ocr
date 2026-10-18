"""Pembanding arsitektur CNN + Transformer + CTC (src.model.TransformerCTC) dan pemilihannya lewat model_config."""

import torch

from src import train
from src.dataset import DOWNSAMPLE, H
from src.model import CRNN, TransformerCTC, build_model

N_CLASSES = 93


def params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_frames_follow_the_same_width_stride_as_crnn():
    model = TransformerCTC(N_CLASSES).eval()
    for width in (64, 257, 1203):
        with torch.no_grad():
            out = model(torch.zeros(1, 1, H, width))
        assert out.shape == (1, width // DOWNSAMPLE, N_CLASSES)


def test_parameter_count_is_close_to_crnn():
    crnn, transformer = params(CRNN(N_CLASSES)), params(TransformerCTC(N_CLASSES))
    assert abs(transformer - crnn) / crnn < 0.03
    assert not hasattr(TransformerCTC(N_CLASSES), "rnn")


def test_padding_frames_are_not_attended():
    """Isi padding tidak boleh sampai ke frame valid lewat atensi.

    CNN asli melihat beberapa kolom padding di tepi kanan baris, dan atensi lalu menyebarkan perbedaan frame tepi itu
    ke seluruh baris (sama seperti LSTM dua arah). Karena itu CNN diganti pemetaan per kolom yang tidak tumpang
    tindih antar-frame: yang tersisa hanya jalur atensi.
    """
    torch.manual_seed(0)
    model = TransformerCTC(N_CLASSES).double().eval()
    model.cnn = torch.nn.Sequential(torch.nn.AvgPool2d((32, DOWNSAMPLE)), torch.nn.Conv2d(1, 256, 1)).double()
    valid = torch.rand(1, 1, H, 400, dtype=torch.float64)
    a = torch.cat([valid, torch.zeros(1, 1, H, 200, dtype=torch.float64)], dim=3)
    b = torch.cat([valid, 5 * torch.rand(1, 1, H, 200, dtype=torch.float64)], dim=3)
    lengths = torch.tensor([400 // DOWNSAMPLE])
    with torch.no_grad():
        masked = (model(a, lengths) - model(b, lengths))[:, :100].abs().max()
        free = (model(a) - model(b))[:, :100].abs().max()
    assert float(masked) < 1e-9
    assert float(free) > 1e-3


def test_build_model_reads_old_and_new_configs():
    old = build_model({"n_classes": N_CLASSES, "channels": [32, 64, 128, 256], "hidden": 256})
    assert type(old) is CRNN
    new = build_model({"n_classes": N_CLASSES, "channels": [32, 64, 128, 256], "hidden": 256,
                       "arch": "transformer", "layers": 2, "heads": 4, "ff": 512})
    assert type(new) is TransformerCTC and len(new.encoder.layers) == 2
    assert new.encoder.layers[0].linear1.out_features == 512


def test_model_config_keeps_crnn_checkpoints_in_the_old_shape():
    class Tok:
        n_classes = N_CLASSES

    crnn = train.model_config(train.parse_args([]), (32, 64, 128, 256), Tok())
    assert crnn == {"n_classes": N_CLASSES, "channels": [32, 64, 128, 256], "hidden": 256}
    args = train.parse_args(["--arch", "transformer", "--layers", "3", "--heads", "8", "--ff", "640"])
    config = train.model_config(args, (32, 64, 128, 256), Tok())
    assert config | {} == {**crnn, "arch": "transformer", "layers": 3, "heads": 8, "ff": 640}
    assert type(build_model(config)) is TransformerCTC
