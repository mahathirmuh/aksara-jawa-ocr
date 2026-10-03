"""Test penjelasan keluaran untuk web: beda suku kata, kolom citra, skor kandidat."""

import math

import numpy as np
import pytest

from src.align import ctc_logp, greedy_path, line_cer, syllable_diff, syllable_spans
from src.tokenizer import TOKENIZER_PATH, Tokenizer, logical_syllables

needs_tokenizer = pytest.mark.skipif(not TOKENIZER_PATH.exists(), reason="tokenizer belum dibangun")


def test_syllable_diff_identical_is_all_ok():
    text = "ꦲꦏꦸꦧꦺꦴꦠꦼꦤ꧀"
    segs = syllable_diff(text, text)
    assert [s["s"] for s in segs] == ["ok"] * len(logical_syllables(text))
    assert "".join(s["t"] for s in segs) == text


def test_syllable_diff_keeps_pasangan_in_one_segment():
    segs = syllable_diff("ꦏꦤ꧀ꦠꦺꦴꦤ꧀", "ꦏꦤ꧀ꦠꦺꦴꦤ꧀")
    assert "ꦤ꧀ꦠꦺꦴ" in [s["t"] for s in segs]


def test_syllable_diff_marks_sub_ins_del_and_space():
    segs = syllable_diff("ꦲꦏꦸꦧꦺꦴ", "ꦲꦏꦶ ꦧꦺꦴꦥ")
    kinds = [(s["t"], s["s"]) for s in segs]
    assert ("ꦏꦶ", "sub") in kinds and (" ", "sp") in kinds and ("ꦥ", "ins") in kinds
    missing = syllable_diff("ꦲꦏꦸꦧꦺꦴ", "ꦲꦧꦺꦴ")
    assert {"t": "ꦏꦸ", "s": "del"} in missing
    # hipotesis tetap bisa dibangun ulang dari segmen non-hapus
    assert "".join(s["t"] for s in segs if s["s"] != "del") == "ꦲꦏꦶ ꦧꦺꦴꦥ"


def test_space_is_ok_only_where_label_has_space():
    assert syllable_diff("ꦲꦏꦸ ꦧꦺꦴ", "ꦲꦏꦸ ꦧꦺꦴ") == [{"t": s, "s": "ok"} for s in ["ꦲ", "ꦏꦸ", " ", "ꦧꦺꦴ"]]
    segs = syllable_diff("ꦲꦏꦸꦧꦺꦴ", "ꦲ ꦏꦸꦧꦺꦴ")
    assert {"t": " ", "s": "sp"} in segs and [s["s"] for s in segs if s["t"] != " "] == ["ok", "ok", "ok"]


def test_line_cer():
    assert line_cer("ꦲꦏꦸ", "ꦲꦏꦸ") == 0
    assert line_cer("ꦲꦏꦸ", "ꦲꦏ") == pytest.approx(1 / 3)


def one_hot(ids, n_classes, repeat=2, blank_between=True):
    frames = []
    for k in ids:
        frames += [k] * repeat + ([0] if blank_between else [])
    lp = np.full((len(frames), n_classes), -20.0, dtype=np.float32)
    lp[np.arange(len(frames)), frames] = 0.0
    return lp


def test_greedy_path_returns_frame_spans():
    lp = one_hot([3, 3, 5], 8)
    ids, spans = greedy_path(lp)
    assert ids == [3, 3, 5]
    assert spans == [[0, 1], [3, 4], [6, 7]]


@needs_tokenizer
def test_syllable_spans_are_ordered_fractions_of_image_width():
    tok = Tokenizer.load(TOKENIZER_PATH)
    text = "ꦲꦏꦸꦧꦺꦴꦠꦼꦤ꧀"
    lp = one_hot(tok.encode(tok.to_visual(text)), tok.n_classes, repeat=3)
    width = lp.shape[0] * 4  # tinggi 96 -> lebar = frame x DOWNSAMPLE
    spans = syllable_spans(lp, tok, image_width=width, image_height=96)
    assert len(spans) == len(logical_syllables(text))
    assert all(0 <= a < b <= 1 for a, b in spans)
    assert all(spans[i][1] <= spans[i + 1][0] + 1e-9 for i in range(len(spans) - 1))


@needs_tokenizer
def test_ctc_logp_prefers_text_supported_by_pixels():
    tok = Tokenizer.load(TOKENIZER_PATH)
    text = "ꦲꦏꦸ"
    lp = one_hot(tok.encode(tok.to_visual(text)), tok.n_classes)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    assert ctc_logp(lp, text, tok) > ctc_logp(lp, "ꦲꦏꦶ", tok) + 5
    assert ctc_logp(lp, text, tok) > -1e-3
    assert ctc_logp(lp, "ꦲ" * 50, tok) == -math.inf
