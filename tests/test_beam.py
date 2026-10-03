"""Test beam search awalan CTC + LM karakter."""

import random

import numpy as np

from src.beam import greedy_ids, prefix_beam_search
from src.charlm import CharLM

CHARSET = ["x", "a", "b"]  # indeks 1..3; blank = 0
C = len(CHARSET) + 1


def frames(seq, high=-0.01, low=-8.0):
    rows = np.full((len(seq), C), low, dtype=np.float32)
    rows[np.arange(len(seq)), seq] = high
    return rows


def test_repeats_need_a_blank_to_become_two_characters():
    assert prefix_beam_search(frames([2, 2]), CHARSET) == [2]
    assert prefix_beam_search(frames([2, 0, 2]), CHARSET) == [2, 2]


def test_all_blank_gives_empty_output():
    assert prefix_beam_search(frames([0, 0, 0]), CHARSET) == []


def test_matches_greedy_on_peaky_output_without_lm():
    rng = random.Random(0)
    for _ in range(20):
        seq = [rng.randrange(C) for _ in range(30)]
        lp = frames(seq)
        assert prefix_beam_search(lp, CHARSET, beam_width=8) == greedy_ids(lp)


def test_lm_flips_an_ambiguous_frame_toward_the_likely_character():
    lp = frames([1, 0, 0])
    lp[1] = [-3.0, -8.0, -0.8, -0.6]  # b sedikit lebih kuat daripada a
    assert prefix_beam_search(lp, CHARSET) == [1, 3]
    lm = CharLM.build(["xa"] * 50 + ["xb"], 2, "xab")
    assert prefix_beam_search(lp, CHARSET, lm, alpha=1.0) == [1, 2]
    # alpha 0 mematikan LM sepenuhnya
    assert prefix_beam_search(lp, CHARSET, lm, alpha=0.0) == [1, 3]
