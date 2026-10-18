"""Fase 4 — CRNN: CNN -> BiLSTM x2 -> Linear, dilatih dengan CTC.

Tinggi 96 dipampatkan habis; lebar hanya di-downsample 4x karena lebar adalah
sumbu waktu CTC dan tumpukan pasangan memadatkan banyak codepoint ke sedikit
kolom (T >= L).
"""

import math

import torch
from torch import nn

from src.dataset import H


def conv_block(c_in: int, c_out: int) -> list[nn.Module]:
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    # Dua MaxPool 2x2 di awal; sisanya hanya memampatkan tinggi.
    WIDTH_STRIDE = 4

    def __init__(
        self,
        n_classes: int,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        hidden: int = 256,
        height: int = H,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.height = height
        self.cnn = nn.Sequential(
            *conv_block(1, c1), nn.MaxPool2d(2, 2),  # H/2,  W/2
            *conv_block(c1, c2), nn.MaxPool2d(2, 2),  # H/4,  W/4  <- stride lebar berhenti di sini
            *conv_block(c2, c3), *conv_block(c3, c3), nn.MaxPool2d((2, 1)),  # H/8
            *conv_block(c3, c4), *conv_block(c4, c4), nn.MaxPool2d((2, 1)),  # H/16
            *conv_block(c4, c4), nn.MaxPool2d((2, 1)),  # H/32 = 3
        )
        self.feat_height = height // 32
        # Kanal x tinggi diratakan, bukan dirata-rata: posisi vertikal (sandhangan
        # atas vs pasangan bawah) harus tetap terbaca.
        self.proj = nn.Linear(c4 * self.feat_height, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """(B, 1, H, W) -> logits (B, W // 4, n_classes).

        `lengths` = jumlah frame valid per sampel; kalau diberikan, LSTM tidak
        membaca frame padding.
        """
        if x.shape[2] != self.height:
            raise ValueError(f"Tinggi input {x.shape[2]}, harus {self.height}")
        f = self.cnn(x)
        b, c, h, t = f.shape
        f = self.proj(f.permute(0, 3, 1, 2).reshape(b, t, c * h))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                f, lengths.cpu().clamp(1, t), batch_first=True, enforce_sorted=False
            )
            out, _ = self.rnn(packed)
            f, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=t)
        else:
            f, _ = self.rnn(f)
        return self.head(f)


class TransformerCTC(CRNN):
    """Pembanding arsitektur: CNN dan proyeksi sama persis dengan CRNN, BiLSTM diganti encoder Transformer.

    Posisi sinusoidal (lebar baris tidak dibatasi), pre-LayerNorm, dan key padding mask dari `lengths`: frame valid
    tidak pernah memperhatikan frame padding, setara dengan packing pada LSTM. Ukuran bawaan (4 lapis, 4 kepala,
    feed-forward 768) membuat jumlah parameternya nyaris sama dengan BiLSTM 2 lapis.
    """

    def __init__(
        self,
        n_classes: int,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        hidden: int = 256,
        height: int = H,
        layers: int = 4,
        heads: int = 4,
        ff: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__(n_classes, channels, hidden, height)
        del self.rnn
        layer = nn.TransformerEncoderLayer(hidden, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)
        self.head = nn.Linear(hidden, n_classes)

    @staticmethod
    def positions(t: int, dim: int, device) -> torch.Tensor:
        pos = torch.arange(t, device=device, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10_000.0) / dim))
        table = torch.zeros(t, dim, device=device)
        table[:, 0::2] = torch.sin(pos * freq)
        table[:, 1::2] = torch.cos(pos * freq)
        return table

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[2] != self.height:
            raise ValueError(f"Tinggi input {x.shape[2]}, harus {self.height}")
        f = self.cnn(x)
        b, c, h, t = f.shape
        f = self.proj(f.permute(0, 3, 1, 2).reshape(b, t, c * h))
        f = f + self.positions(t, f.shape[-1], f.device)
        mask = None
        if lengths is not None:
            mask = torch.arange(t, device=f.device).unsqueeze(0) >= lengths.to(f.device).clamp(1, t).unsqueeze(1)
        return self.head(self.encoder(f, src_key_padding_mask=mask))


ARCHS = {"crnn": CRNN, "transformer": TransformerCTC}


def build_model(config: dict) -> CRNN:
    """Model dari `model_config` checkpoint; checkpoint lama tanpa kunci `arch` adalah CRNN."""
    arch = config.get("arch", "crnn")
    kwargs = {"channels": tuple(config["channels"]), "hidden": config["hidden"]}
    if arch == "transformer":
        kwargs |= {k: config[k] for k in ("layers", "heads", "ff") if k in config}
    return ARCHS[arch](config["n_classes"], **kwargs)
