"""Test src/metrics.py: recall, presisi, F1, akurasi karakter, dan per kelas dari penjajaran karakter."""

import random
from functools import lru_cache

import pytest
from rapidfuzz.distance import Levenshtein

from src.metrics import alignment, char_counts, char_metrics, class_metrics, glyph_class, macro_f1, rates

KA, NA, TA, DA, YA = "ꦏ", "ꦤ", "ꦠ", "ꦢ", "ꦪ"
KA_MURDA, TWO, LAYAR, WULU, TALING = "ꦑ", "꧒", "ꦂ", "ꦶ", "ꦺ"


def best_counts(ref: str, hyp: str) -> tuple[int, int, int]:
    """(biaya, kecocokan, substitusi sekelas) terbaik secara leksikografis; pembanding rekursif untuk `alignment`."""

    @lru_cache(maxsize=None)
    def best(i: int, j: int) -> tuple[int, int, int]:
        if i == len(ref) or j == len(hyp):
            return len(ref) - i + len(hyp) - j, 0, 0
        cost, neg, same = best(i + 1, j + 1)
        diag = (cost, neg - 1, same) if ref[i] == hyp[j] else (cost + 1, neg, same - (glyph_class(ref[i]) == glyph_class(hyp[j])))
        cost, neg, same = best(i + 1, j)
        delete = (cost + 1, neg, same)
        cost, neg, same = best(i, j + 1)
        return min(diag, delete, (cost + 1, neg, same))

    cost, neg, same = best(0, 0)
    return cost, -neg, -same


def test_alignment_is_minimum_cost_with_most_matches_then_same_class():
    assert [glyph_class(ch) for ch in (" ", LAYAR, TALING, "꧀", KA, TWO)] == [0, 1, 1, 1, 2, 2]
    rng = random.Random(1)
    alphabet = [KA, NA, TA, " ", KA_MURDA, TWO, LAYAR, WULU]
    for _ in range(500):
        ref = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 7)))
        hyp = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 7)))
        ops = alignment(ref, hyp)
        assert [i for tag, i, _ in ops if tag != "insert"] == list(range(len(ref)))
        assert [j for tag, _, j in ops if tag != "delete"] == list(range(len(hyp)))
        cost, matches, same_class = best_counts(ref, hyp)
        counts = char_counts(ref, hyp)
        assert counts["sub"] + counts["del"] + counts["ins"] == cost == Levenshtein.distance(ref, hyp)
        assert counts["match"] == matches
        assert sum(tag == "replace" and glyph_class(ref[i]) == glyph_class(hyp[j]) for tag, i, j in ops) == same_class
        # Identitas yang dipakai rumus di docstring modul.
        assert counts["match"] + counts["sub"] + counts["del"] == len(ref)
        assert counts["match"] + counts["sub"] + counts["ins"] == len(hyp)


def test_char_counts_examples():
    assert char_counts(KA + NA + TA, KA + NA + TA) == {"match": 3, "sub": 0, "del": 0, "ins": 0}
    assert char_counts(KA + NA + TA, KA + TA) == {"match": 2, "sub": 0, "del": 1, "ins": 0}
    assert char_counts(KA + TA, KA + NA + TA) == {"match": 2, "sub": 0, "del": 0, "ins": 1}
    assert char_counts(KA + NA, KA + TA) == {"match": 1, "sub": 1, "del": 0, "ins": 0}
    # "ꦢꦪ" vs "ꦪ ": ꦪ dihitung cocok (bukan dua substitusi seperti opcodes rapidfuzz).
    assert char_counts(DA + YA, YA + " ") == {"match": 1, "sub": 0, "del": 1, "ins": 1}
    assert char_counts("", "") == {"match": 0, "sub": 0, "del": 0, "ins": 0}


def test_rates_follow_the_formulas_and_handle_empty_sides():
    r = rates(match=8, sub=1, dele=1, ins=2)
    assert r["recall"] == pytest.approx(8 / 10)
    assert r["precision"] == pytest.approx(8 / 11)
    assert r["f1"] == pytest.approx(2 * (8 / 11) * (8 / 10) / (8 / 11 + 8 / 10))
    assert r["char_accuracy"] == pytest.approx(8 / 12)
    assert rates(0, 0, 0, 0) == {"precision": None, "recall": None, "f1": None, "char_accuracy": None}
    # Referensi ada, keluaran kosong: recall 0, presisi tidak terdefinisi, F1 tidak terdefinisi.
    assert rates(0, 0, 3, 0) == {"precision": None, "recall": 0.0, "f1": None, "char_accuracy": 0.0}
    # Keduanya ada tetapi tidak ada yang cocok: F1 = 0, bukan pembagian dengan nol.
    assert rates(0, 2, 0, 0)["f1"] == 0.0


def test_char_metrics_is_micro_over_lines_and_consistent_with_cer():
    pairs = [(KA + NA + TA, KA + NA + TA), (KA + NA + TA, KA + TA + TA), (KA, KA + NA + NA)]
    m = char_metrics(pairs)
    assert m["counts"] == {"match": 6, "sub": 1, "del": 0, "ins": 2}
    ref_len = sum(len(r) for r, _ in pairs)
    edits = sum(Levenshtein.distance(r, h) for r, h in pairs)
    assert m["counts"]["sub"] + m["counts"]["del"] + m["counts"]["ins"] == edits  # = pembilang CER
    assert m["recall"] == pytest.approx(6 / ref_len)
    assert m["precision"] == pytest.approx(6 / 9)
    assert m["char_accuracy"] == pytest.approx(6 / 9)
    assert char_metrics([]) == {"counts": {"match": 0, "sub": 0, "del": 0, "ins": 0},
                                "precision": None, "recall": None, "f1": None, "char_accuracy": None}


def test_class_metrics_assign_substitutions_to_both_classes():
    pairs = [(KA + NA + TA, KA + TA + TA),  # NA -> TA: fn NA, fp TA
             (KA + NA, KA),  # NA hilang: fn NA
             (KA, KA + WULU)]  # wulu tambahan: fp wulu
    rows = {c["char"]: c for c in class_metrics(pairs)}
    assert rows[KA] == {"char": KA, "code": "U+A98F", "name": "JAVANESE LETTER KA", "ref": 3, "hyp": 3,
                        "tp": 3, "fn": 0, "fp": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert (rows[NA]["ref"], rows[NA]["tp"], rows[NA]["fn"], rows[NA]["fp"]) == (2, 0, 2, 0)
    assert rows[NA]["recall"] == 0.0 and rows[NA]["precision"] is None and rows[NA]["f1"] is None
    assert (rows[TA]["ref"], rows[TA]["tp"], rows[TA]["fn"], rows[TA]["fp"]) == (1, 1, 0, 1)
    assert rows[TA]["precision"] == pytest.approx(0.5) and rows[TA]["recall"] == 1.0
    assert rows[TA]["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert (rows[WULU]["ref"], rows[WULU]["fp"], rows[WULU]["recall"], rows[WULU]["precision"]) == (0, 1, None, 0.0)
    # Urut dari yang paling sering di referensi; kelas yang hanya ada di keluaran di belakang.
    assert [c["char"] for c in class_metrics(pairs)] == [KA, NA, TA, WULU]
    # Jumlah tp semua kelas = match mikro; fn = sub + del; fp = sub + ins.
    micro = char_metrics(pairs)["counts"]
    assert sum(c["tp"] for c in rows.values()) == micro["match"]
    assert sum(c["fn"] for c in rows.values()) == micro["sub"] + micro["del"]
    assert sum(c["fp"] for c in rows.values()) == micro["sub"] + micro["ins"]
    # Macro-F1 hanya atas kelas yang ada di referensi dan punya F1 (wulu dan NA tidak ikut).
    assert macro_f1(rows.values()) == pytest.approx((1.0 + rows[TA]["f1"]) / 2)
    assert macro_f1([]) is None
