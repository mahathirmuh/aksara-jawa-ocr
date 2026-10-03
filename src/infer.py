"""Inference standalone: citra satu baris -> Unicode aksara Jawa (urutan logis, NFC).

  python -m src.infer out/checkpoints/<run>/best.pt baris1.png [baris2.png ...]

Checkpoint membawa charset dan tabel reorder, jadi tidak butuh data/ maupun font.
"""

import argparse
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

from src.dataset import to_tensor
from src.decode import greedy_decode
from src.model import CRNN
from src.tokenizer import Tokenizer


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[CRNN, Tokenizer]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    tokenizer = Tokenizer(
        [chr(int(cp[2:], 16)) for cp in ckpt["tokenizer"]["charset"]],
        ckpt["tokenizer"]["reorder"],
    )
    config = ckpt["model_config"]
    model = CRNN(config["n_classes"], channels=tuple(config["channels"]), hidden=config["hidden"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), tokenizer


def normalize_polarity(img: Image.Image) -> Image.Image:
    """Model dilatih pada teks gelap di atas latar terang; balik kalau sebaliknya.

    Baris teks didominasi latar, jadi median piksel mewakili warna latar.
    """
    gray = img.convert("L")
    return ImageOps.invert(gray) if np.median(np.asarray(gray)) < 128 else gray


def add_margin(img: Image.Image, ratio: float) -> Image.Image:
    """Latar putih di sekeliling citra teks gelap, selebar `ratio` x tinggi citra.

    Render sintetis selalu diberi margin, sedangkan potongan pindaian nyata sering
    dipotong rapat sampai tinta menyentuh tepi.
    """
    if ratio <= 0:
        return img
    pad = max(1, round(img.height * ratio))
    out = Image.new("L", (img.width + 2 * pad, img.height + 2 * pad), 255)
    out.paste(img.convert("L"), (pad, pad))
    return out


@torch.no_grad()
def predict(
    img: Image.Image,
    model: CRNN,
    tokenizer: Tokenizer,
    device: str | torch.device = "cpu",
    pad_ratio: float = 0.0,
) -> str:
    x = to_tensor(add_margin(normalize_polarity(img), pad_ratio)).unsqueeze(0).to(device)
    logits = model(x)
    return greedy_decode(logits, torch.tensor([logits.shape[1]]), tokenizer)[0]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--pad-ratio", type=float, default=0.0, help="margin putih x tinggi citra (potongan rapat)")
    args = parser.parse_args(argv)

    model, tokenizer = load_checkpoint(args.checkpoint)
    sys.stdout.reconfigure(encoding="utf-8")
    for path in args.images:
        print(f"{path}\t{predict(Image.open(path), model, tokenizer, pad_ratio=args.pad_ratio)}")


if __name__ == "__main__":
    main()
