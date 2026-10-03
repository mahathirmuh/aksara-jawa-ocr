"""Test LM karakter Witten-Bell — kondisi "kamus" ablasi koreksi pasca-OCR."""

import pytest

from src.charlm import EOS, CharLM

VOCAB = "xab "
LINES = ["xaxa", "xa xa", "xaxab", "ab"]


@pytest.fixture(scope="module")
def lm() -> CharLM:
    return CharLM.build(LINES, 3, VOCAB)


@pytest.mark.parametrize("history", ["", "x", "xa", "bb", "zz"])
def test_distribution_sums_to_one(lm, history):
    total = sum(lm.prob(history, ch) for ch in list(VOCAB) + [EOS])
    assert abs(total - 1.0) < 1e-9


def test_seen_continuation_beats_unseen(lm):
    assert lm.prob("x", "a") > lm.prob("x", "b") > 0
    assert lm.prob("", "x") > lm.prob("", "b")  # awal baris lebih sering x


def test_every_order_is_a_proper_distribution():
    for order in (1, 2, 5):
        model = CharLM.build(LINES, order, VOCAB)
        assert abs(sum(model.prob("xa", ch) for ch in list(VOCAB) + [EOS]) - 1.0) < 1e-9


def test_rejects_characters_outside_vocab():
    with pytest.raises(ValueError, match="U\\+"):
        CharLM.build(["xq"], 2, VOCAB)


def test_save_load_roundtrip(lm, tmp_path):
    path = tmp_path / "lm.pkl"
    lm.save(path)
    loaded = CharLM.load(path)
    assert loaded.order == lm.order and loaded.vocab == lm.vocab
    assert loaded.prob("xa", "x") == lm.prob("xa", "x")
