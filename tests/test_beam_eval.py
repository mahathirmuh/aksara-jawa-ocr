"""Test scripts/beam_eval.py: set dev sintetis mengikuti ejaan baku bila diminta, tanpa model."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))  # scripts/ bukan paket

import beam_eval  # noqa: E402
from src.dataset import TRAIN_FONTS  # noqa: E402
from src.spelling import standardize  # noqa: E402
from src.tokenizer import TOKENIZER_PATH, Tokenizer  # noqa: E402

LINES = ["ꦥꦤ꧀ꦕꦶ ꦫꦼꦒ ꦭꦼꦩꦃ", "ꦠꦿꦼꦱ꧀ꦤ ꦠꦲꦸꦤ꧀ ꧑꧙꧔꧕", "ꦏꦸꦭ ꦧꦝꦺ ꦠꦶꦤ꧀ꦢꦏ꧀"]


def test_dev_lines_follow_the_standard_spelling_option(tmp_path, monkeypatch):
    (tmp_path / "data/splits").mkdir(parents=True)
    (tmp_path / "data/splits/val.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    monkeypatch.setattr(beam_eval, "ROOT", tmp_path)
    monkeypatch.setattr(beam_eval, "line_log_probs", lambda img, model: img.size)  # tanpa model: ukuran citra saja
    tok = Tokenizer.load(TOKENIZER_PATH)
    fonts = [TRAIN_FONTS[0]]

    def texts(**kwargs):
        return [text for text, _ in beam_eval.synthetic_items(None, tok, "val", 3, fonts, "none", 0.0, seed=4242, **kwargs)]

    plain = texts()
    assert sorted(plain) == sorted(LINES)  # bawaan: tanpa ejaan baku, dev laporan lama tidak berubah
    assert texts(standard_prob=0.0) == plain
    assert sorted(texts(standard_prob=1.0)) == sorted(standardize(line) for line in LINES)
    # Citranya ikut berubah (bukan hanya labelnya).
    sizes = dict(beam_eval.synthetic_items(None, tok, "val", 3, fonts, "none", 0.0, seed=4242, standard_prob=1.0))
    assert sizes[standardize(LINES[1])] != dict(beam_eval.synthetic_items(None, tok, "val", 3, fonts, "none", 0.0, seed=4242))[LINES[1]]
