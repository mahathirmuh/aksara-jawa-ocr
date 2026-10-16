"""Test scripts/beam_eval.py: set dev sintetis mengikuti ejaan baku bila diminta, tanpa model."""

import json
import sys
from pathlib import Path

import pytest
import torch

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


def test_dev_lines_follow_the_slant_option(tmp_path, monkeypatch):
    (tmp_path / "data/splits").mkdir(parents=True)
    (tmp_path / "data/splits/val.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    monkeypatch.setattr(beam_eval, "ROOT", tmp_path)
    monkeypatch.setattr(beam_eval, "line_log_probs", lambda img, model: img.size)
    tok = Tokenizer.load(TOKENIZER_PATH)

    def sizes(**kwargs):
        return dict(beam_eval.synthetic_items(None, tok, "val", 3, [TRAIN_FONTS[0]], "none", 0.0, seed=4242, **kwargs))

    plain = sizes()
    assert sizes(slant_prob=0.0, slant_max=0.6) == plain  # peluang 0: dev laporan lama tidak berubah
    # Selalu dimiringkan sejauh 0,6: kanvas melebar sebanding dengan tingginya, label tetap.
    tilted = sizes(slant_prob=1.0, slant_max=0.6)
    assert set(tilted) == set(plain) and all(tilted[text][0] > plain[text][0] for text in plain)


def test_training_recipe_comes_from_the_checkpoint_and_its_log(tmp_path):
    (tmp_path / "fonts/extra").mkdir(parents=True)
    for font in TRAIN_FONTS:
        (tmp_path / "fonts" / font.name).write_bytes(b"")
    (tmp_path / "fonts/extra/Tambahan.ttf").write_bytes(b"")
    ckpt = tmp_path / "out/checkpoints/fase9_slant/last_snapshot.pt"
    ckpt.parent.mkdir(parents=True)
    args = {"augment": "fase5", "drop_space_prob": 0.5, "standard_prob": 0.5, "slant_prob": 0.5, "slant_max": 0.6,
            "font_fixes": True, "extra_fonts": "fonts/extra"}
    torch.save({"step": 1500, "args": args}, ckpt)
    log = ckpt.parent / "log.jsonl"
    names = [TRAIN_FONTS[1].name, "Tambahan.ttf"]

    def write_log(*rows):
        log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    # Font dari proses yang menulis checkpoint (mulai sebelum langkahnya), sesudah perbaikan font, urut seperti di log.
    write_log({"event": "start", "step": 0, "fonts": [TRAIN_FONTS[0].name]}, {"event": "start", "step": 500, "fonts": names},
              {"step": 1500, "event": "end"}, {"event": "start", "step": 1500, "fonts": ["Nanti.ttf"]})
    recipe = beam_eval.training_recipe(ckpt, tmp_path)
    assert recipe["fonts"] == [tmp_path / "fonts" / names[0], tmp_path / "fonts/extra/Tambahan.ttf"]
    assert {key: value for key, value in recipe.items() if key != "fonts"} == {
        "augment": "fase5", "drop_space_prob": 0.5, "standard_prob": 0.5, "slant_prob": 0.5, "slant_max": 0.6, "font_fixes": True}
    # Run tanpa render miring: kemiringan terbesar (bawaan argparse) tidak berarti apa-apa; run lama tanpa opsinya.
    torch.save({"step": 1500, "args": {**args, "slant_prob": 0.0}}, ckpt)
    assert beam_eval.training_recipe(ckpt, tmp_path)["slant_max"] == 0.0
    torch.save({"step": 1500, "args": {"augment": "fase5", "extra_fonts": "fonts/extra"}}, ckpt)
    old = beam_eval.training_recipe(ckpt, tmp_path)
    assert (old["drop_space_prob"], old["standard_prob"], old["slant_prob"], old["font_fixes"]) == (0.0, 0.0, 0.0, False)
    # Log tanpa nama font, atau font yang berkasnya hilang: dev tidak bisa dirender seperti training-nya.
    write_log({"event": "start", "step": 0}, {"step": 1500, "event": "end"})
    with pytest.raises(SystemExit, match="tidak mencatat font latih"):
        beam_eval.training_recipe(ckpt, tmp_path)
    write_log({"event": "start", "step": 0, "fonts": ["Hilang.ttf"]}, {"step": 1500, "event": "end"})
    with pytest.raises(SystemExit, match="font latih Hilang.ttf"):
        beam_eval.training_recipe(ckpt, tmp_path)
