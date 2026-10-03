"""Test scripts/eval_rare.py: evaluasi sintetis aksara langka, dengan data buatan dan pembaca palsu (tanpa model)."""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from src.dataset import TRAIN_FONTS, SyntheticLines
from src.decode import cer
from src.text_augment import RareText
from src.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))  # scripts/ bukan paket

import eval_rare  # noqa: E402
from compare_runs import matched_mask  # noqa: E402
from eval_rare import (  # noqa: E402
    CONDITIONS,
    EvalError,
    RenderedLines,
    compare_pair,
    condition_dataset,
    estimate_full,
    line_stats,
    parse_checkpoints,
    read_as,
    sample_indices,
    summarize_run,
)

TOKENIZER_JSON = ROOT / "data" / "tokenizer.json"
NOTO = next(p for p in TRAIN_FONTS if p.name.startswith("NotoSansJavanese"))

KA, NA, TA, SA = "ꦏ", "ꦤ", "ꦠ", "ꦱ"
KA_MURDA, SA_MURDA, ADEG, WINDU, LETTER_E = "ꦑ", "ꦯ", "꧋", "꧆", "ꦌ"
ZERO, SIX = "꧐", "꧖"
RARE = (KA_MURDA, SA_MURDA, WINDU, LETTER_E, ADEG)
RARE_SET = frozenset(RARE)

# Keluaran transliterator untuk kalimat Jawa nyata (sama dengan tests/test_text_augment.py).
LINES = [
    "ꦧꦱ ꦗꦮ ꦲꦶꦏꦸ ꦧꦱ ꦏꦁ ꦢꦶꦲꦼꦁꦒꦺꦴ ꦢꦼꦤꦶꦁ ꦮꦺꦴꦁ ꦗꦮ꧉",
    "ꦏꦸꦛ ꦪꦺꦴꦒꦾꦏꦂꦠ ꦲꦤ ꦲꦶꦁ ꦠ꧀ꦭꦠꦃ ꦗꦮ ꦠꦼꦔꦃ꧈",
    "ꦮꦺꦴꦁ꧈ ꦏꦺꦴꦮꦺ꧈ ꦚꦮꦶꦗꦶ꧈ ꦏꦿꦠꦺꦴꦤ꧀ ꦥꦿꦝꦤ꧈ ꦱꦱ꧀ꦠꦿ",
    "ꦲꦶꦁ ꦱꦶꦱꦶꦃ ꦭꦺꦴꦂ ꦮꦺꦠꦤ꧀ ꧌ꦏꦶꦢꦸꦭ꧀꧍ ꦲꦤ ꦏꦭꦶ꧇",
]

needs_tokenizer = pytest.mark.skipif(not TOKENIZER_JSON.exists(), reason="tokenizer belum dibangun")


# --- masukan -----------------------------------------------------------------------------------------------


def test_sample_is_deterministic_and_small_samples_are_prefixes_of_large_ones():
    assert sample_indices(1000, 20, 0) == sample_indices(1000, 20, 0)
    assert sample_indices(1000, 20, 0) == sample_indices(1000, 500, 0)[:20]  # --lines 20 = awal --lines 500
    assert sample_indices(1000, 20, 0) != sample_indices(1000, 20, 1)
    assert len(set(sample_indices(1000, 500, 0))) == 500
    assert sorted(sample_indices(50, 0, 3)) == list(range(50))  # 0 = semua baris


def test_checkpoint_specs(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_rare, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    snapshot = tmp_path / "checkpoints" / "run_a" / "last_snapshot.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"x")
    other = tmp_path / "b.pt"
    other.write_bytes(b"x")
    parsed = parse_checkpoints(["run_a", f"lain={other}"])
    assert parsed == {"run_a": snapshot, "lain": other} and list(parsed) == ["run_a", "lain"]  # urutan = arah selisih
    with pytest.raises(EvalError, match="disebut dua kali"):
        parse_checkpoints(["run_a", f"run_a={other}"])
    with pytest.raises(EvalError, match="checkpoint tidak ada: run_b"):
        parse_checkpoints(["run_a", "run_b"])
    with pytest.raises(EvalError, match="bukan NAMA"):
        parse_checkpoints([f"={other}"])


# --- per baris ---------------------------------------------------------------------------------------------


def test_read_as_uses_the_same_alignment_as_matched_mask():
    assert read_as("abc", "abc") == ["a", "b", "c"]
    assert read_as("abc", "axc") == ["a", "x", "c"]  # substitusi
    assert read_as("abc", "ac") == ["a", "", "c"]  # hapus
    assert read_as("ac", "abc") == ["a", "c"]  # sisipan tidak punya posisi di referensi
    assert read_as("abc", "") == ["", "", ""] and read_as("", "abc") == []
    rng = random.Random(0)
    for _ in range(500):
        ref = "".join(rng.choice("abcd ") for _ in range(rng.randint(0, 12)))
        hyp = "".join(rng.choice("abcde ") for _ in range(rng.randint(0, 12)))
        assert [seen == ch for ch, seen in zip(ref, read_as(ref, hyp))] == matched_mask(ref, hyp)


def test_non_rare_errors_skip_blocks_that_read_a_rare_glyph():
    def plain(ref: str, hyp: str) -> int:
        return line_stats(ref, hyp, RARE_SET)["plain_edits"]

    token = KA + " " + WINDU + " " + NA + TA  # sisipan pada/angka berbentuk " X" sebelum spasi
    assert plain(token, KA + ZERO + "꧀꧀" + NA + TA) == 0  # satu glyph terbaca tiga karakter, menimpa spasinya
    assert plain(token, KA + ZERO + "꧀꧀" + NA + SA) == 1  # kesalahan di blok lain tetap dihitung
    assert plain(token, KA + " " + NA + TA) == 0  # sisipan " X" hilang seluruhnya
    assert plain(KA + NA, KA + KA_MURDA + NA) == 0  # sisipan aksara langka palsu: diukur presisi
    assert plain(KA + NA + TA, KA + KA_MURDA + TA) == 1  # aksara biasa terbaca aksara langka
    assert plain(KA + NA, KA + NA + TA) == 1


def test_line_stats_counts_rare_codepoints_without_the_line_opener():
    ref = ADEG + KA + KA_MURDA + NA + " " + WINDU + " " + TA
    # Pembuka terbaca; ka murda terbaca ka; windu terbaca angka nol.
    stats = line_stats(ref, ADEG + KA + KA + NA + " " + ZERO + " " + TA, RARE_SET)
    assert stats["opener"] and stats["opener_read"] and not stats["false_opener"]
    assert stats["ref"] == Counter({KA_MURDA: 1, WINDU: 1})  # pembuka tidak ikut aksara langka
    assert stats["hits"] == Counter()
    assert stats["misread"] == Counter({(KA_MURDA, KA): 1, (WINDU, ZERO): 1})
    assert stats["emitted"] == Counter({ZERO: 1})  # angka nol dilacak untuk homoglif; pembuka keluaran tidak ikut
    assert stats["edits"] == 2 and stats["chars"] == 8
    assert (stats["plain_edits"], stats["plain_chars"]) == (0, 0)  # baris berpembuka tidak ikut CER bukan-langka
    # Bacaan aksara langka (ka untuk ka murda, angka nol untuk windu) bukan kesalahan karakter biasa.
    plain = line_stats(ref[1:], KA + KA + NA + " " + ZERO + " " + TA, RARE_SET)
    assert (plain["plain_edits"], plain["plain_chars"]) == (0, 5)

    missed = line_stats(ADEG + KA + NA, KA + NA, RARE_SET)
    assert missed["opener"] and not missed["opener_read"] and missed["opener_seen"] == ""
    false = line_stats(KA + NA, " " + ADEG + KA + NA, RARE_SET)  # spasi di depan diabaikan
    assert false["false_opener"] and not false["opener"] and false["emitted"] == Counter()


def test_opener_reading_is_excluded_from_emitted_by_alignment_not_position():
    # Pembuka terbaca benar walau ada keluaran palsu di depannya: bukan keluaran aksara langka palsu.
    spurious = line_stats(ADEG + KA + NA, TA + ADEG + KA + NA, RARE_SET)
    assert spurious["opener_read"] and spurious["emitted"] == Counter() and spurious["edits"] == 1
    # Adeg-adeg tengah baris yang terbaca di awal keluaran (kata pertama hilang): benar DAN terhitung keluaran.
    early = line_stats(KA + " " + ADEG + " " + NA, ADEG + " " + NA, RARE_SET)
    assert early["hits"] == early["emitted"] == Counter({ADEG: 1})
    assert early["false_opener"]  # tetap pembuka palsu menurut posisi, sama dengan compare_runs
    # Pembuka ganda: satu bacaan pembuka, satu keluaran palsu.
    assert line_stats(ADEG + KA, ADEG + ADEG + KA, RARE_SET)["emitted"] == Counter({ADEG: 1})
    rng = random.Random(0)
    for _ in range(2000):
        ref = "".join(rng.choice((KA, NA, " ", ADEG, KA_MURDA)) for _ in range(rng.randint(1, 8)))
        ref = ADEG + ref if rng.random() < 0.5 else ref
        hyp = "".join(rng.choice((KA, NA, " ", ADEG, KA_MURDA)) for _ in range(rng.randint(0, 9)))
        stats = line_stats(ref, hyp, RARE_SET)
        assert all(stats["hits"][ch] <= min(stats["ref"][ch], stats["emitted"][ch]) for ch in RARE_SET), (ref, hyp)
        if stats["opener"]:  # satu-satunya adeg-adeg keluaran yang tidak ikut `emitted`: bacaan pembuka
            assert stats["emitted"][ADEG] == hyp.count(ADEG) - stats["opener_read"], (ref, hyp)
        else:  # paling banyak satu yang tidak ikut, dan hanya bila ada pembuka palsu
            assert hyp.count(ADEG) - stats["emitted"][ADEG] <= int(stats["false_opener"]), (ref, hyp)


# --- ringkasan ---------------------------------------------------------------------------------------------


PAIRS = [
    (ADEG + KA + KA_MURDA + NA, ADEG + KA + KA_MURDA + NA),  # semua benar
    (KA + SA_MURDA + " " + WINDU + " " + TA, KA + SA + " " + ZERO + " " + TA),  # sa murda -> sa, windu -> angka 0
    (KA + NA + " " + ZERO, KA + NA + " " + WINDU),  # angka 0 -> windu
    (TA + LETTER_E + NA, TA + SIX + NA),  # E -> angka 6
    (KA + NA, ADEG + KA + NA),  # pembuka palsu
]


def test_summary_definitions():
    stats = [line_stats(ref, hyp, RARE_SET) for ref, hyp in PAIRS]
    summary = summarize_run(PAIRS, stats, RARE)
    assert summary["cer"] == cer([r for r, _ in PAIRS], [h for _, h in PAIRS])
    assert summary["exact_line_accuracy"] == pytest.approx(1 / 5)
    # Bukan-langka per baris: berpembuka (tidak ikut), 0/4, 1/4 (angka 0 terbaca windu), 0/2, 0/2 (pembuka palsu).
    assert summary["cer_bukan_langka"] == pytest.approx(1 / 12) and summary["cer_bukan_langka_lines"] == 4

    langka, murda = summary["groups"]["langka"], summary["groups"]["murda"]
    assert (langka["ref"], langka["hits"], langka["emitted"]) == (4, 1, 2)  # pembuka baris 1 & 5 tidak ikut
    assert langka["recall"] == pytest.approx(1 / 4) and langka["precision"] == pytest.approx(1 / 2)
    assert (murda["codepoints"], murda["ref"], murda["hits"], murda["recall"]) == (2, 2, 1, 0.5)

    entries = {entry["char"]: entry for entry in summary["per_codepoint"]}
    assert [entry["char"] for entry in summary["per_codepoint"]] == list(RARE)
    assert entries[KA_MURDA]["recall"] == 1.0 and entries[KA_MURDA]["precision"] == 1.0
    assert entries[SA_MURDA]["recall"] == 0.0 and entries[SA_MURDA]["precision"] is None  # tidak pernah dikeluarkan
    assert entries[SA_MURDA]["misread"] == [{"as": SA, "n": 1}]
    assert entries[WINDU]["emitted"] == 1 and entries[WINDU]["precision"] == 0.0
    assert entries[ADEG]["ref"] == 0 and entries[ADEG]["recall"] is None  # pembuka dihitung terpisah

    assert summary["opener"] == {"codepoint": "U+A9CB", "lines": 1, "read": 1, "recall": 1.0, "misread": [],
                                 "lines_without": 4, "false": 1, "false_rate": 0.25}
    windu, letter_e, pa_murda, _ = summary["homoglyphs"]
    assert (windu["label"], letter_e["label"], pa_murda["label"]) == ("windu/0", "e/6", "pa murda/8")
    assert (windu["ref_x"], windu["ref_y"], windu["x_as_y"], windu["y_as_x"]) == (1, 1, 1, 1)
    assert (letter_e["ref_x"], letter_e["ref_y"], letter_e["x_as_y"], letter_e["y_as_x"]) == (1, 0, 1, 0)
    assert pa_murda["ref_x"] == pa_murda["x_as_y"] == 0


def test_reader_that_only_drops_rare_codepoints_has_zero_non_rare_cer():
    refs = [ref for ref, _ in PAIRS]
    for drop in (lambda text: "".join(ch for ch in text if ch not in RARE_SET),  # token jadi dua spasi
                 lambda text: "".join(ch for ch in text if ch not in RARE_SET).replace("  ", " ")):  # token hilang
        pairs = [(ref, drop(ref)) for ref in refs]
        summary = summarize_run(pairs, [line_stats(r, h, RARE_SET) for r, h in pairs], RARE)
        assert summary["cer_bukan_langka"] == 0 and summary["cer"] > 0
        assert summary["groups"]["langka"]["recall"] == 0.0 and summary["opener"]["recall"] == 0.0


def test_compare_pair_is_a_minus_b_on_the_same_lines():
    refs = [ADEG + KA + KA_MURDA + NA, KA + NA + TA, KA + SA_MURDA, ADEG + TA]
    bad = [KA + KA + NA, KA + NA + TA, KA + SA, ADEG + TA]  # baris 1: pembuka & murda hilang; baris 3: murda
    stats_a = [line_stats(ref, ref, RARE_SET) for ref in refs]
    stats_b = [line_stats(ref, hyp, RARE_SET) for ref, hyp in zip(refs, bad)]
    comparison = compare_pair(stats_a, stats_b, RARE, resamples=300, seed=0)
    m = comparison["metrics"]
    assert m["cer"]["a"] == 0 and m["cer"]["diff"] == pytest.approx(-m["cer"]["b"]) and m["cer"]["b"] > 0
    assert (m["recall_langka"]["a"], m["recall_langka"]["b"], m["recall_langka"]["diff"]) == (1.0, 0.0, 1.0)
    assert m["recall_langka"]["bootstrap"]["ci95"] == [1.0, 1.0]  # setiap resample yang sah: A semua, B nol
    assert (m["recall_pembuka"]["a"], m["recall_pembuka"]["b"]) == (1.0, 0.5)
    assert m["pembuka_palsu"]["a"] == m["pembuka_palsu"]["b"] == 0.0
    assert set(m) == {"cer", "cer_bukan_langka", "recall_langka", "presisi_langka", "recall_murda",
                      "recall_pembuka", "pembuka_palsu"}
    p = comparison["paired_lines"]
    assert (p["units"], p["a_better"], p["b_better"], p["tie"]) == (4, 2, 0, 2)
    assert comparison["paired_opener"] == {"lines": 2, "both": 1, "only_a": 1, "only_b": 0, "sign_test_p": 1.0}

    same = compare_pair(stats_a, stats_a, RARE, resamples=200, seed=0)
    assert same["metrics"]["cer"]["diff"] == 0 and same["metrics"]["cer"]["bootstrap"]["ci95"] == [0.0, 0.0]
    without_murda = compare_pair(stats_a, stats_b, (WINDU, ADEG), resamples=0, seed=0)
    assert "recall_murda" not in without_murda["metrics"]
    assert without_murda["metrics"]["cer"]["bootstrap"] is None


def test_full_run_estimate_scales_measured_times():
    blocks = {c: {"timing": {"render_ms_per_line": 10.0, "infer_ms_per_line": {"a": 100.0, "b": 300.0}}}
              for c in CONDITIONS}
    estimate = estimate_full(blocks, 2000, 3)
    # Per kondisi: 2000 x (10 + 3 x rata-rata 200) ms; checkpoint yang belum diukur memakai rata-rata.
    assert estimate["seconds"] == pytest.approx(3 * 2000 * (10 + 3 * 200) / 1000)
    assert estimate["render_s"] == pytest.approx(60.0)
    assert estimate_full({"biasa": blocks["biasa"]}, 2000, 3) is None


# --- render ------------------------------------------------------------------------------------------------


@needs_tokenizer
def test_conditions_share_text_and_render_deterministically():
    tok = Tokenizer.load(TOKENIZER_JSON)
    lines = LINES * 3
    common = RareText.from_lines(LINES, tok.charset).common_letters

    def render(condition: str) -> list[dict]:
        dataset = condition_dataset(lines, tok, NOTO, RARE, common, condition, seed=0)
        return [RenderedLines(dataset)[i] for i in range(len(lines))]

    plain, rare, augmented = render("biasa"), render("langka"), render("langka+fase5")
    assert all("image" in item and "dropped" not in item for item in plain + rare + augmented)
    assert [item["text"] for item in plain] == lines  # biasa: teks test apa adanya
    assert [item["text"] for item in augmented] == [item["text"] for item in rare]  # berpasangan antar-kondisi
    assert any(a["image"].tobytes() != b["image"].tobytes() for a, b in zip(rare, augmented))
    starts = [item["text"].startswith(ADEG) for item in rare]
    assert any(starts) and not all(starts)  # pembuka p=0,5
    inserted = [any(ch in RARE_SET for ch in item["text"].removeprefix(ADEG)) for item in rare]
    assert sum(inserted) >= len(lines) - 2  # sisip p=1; sisipan bisa gagal bila tidak ada tempat yang sah
    again = render("langka+fase5")
    assert all(a["image"].tobytes() == b["image"].tobytes() for a, b in zip(augmented, again))  # deterministik


# --- skrip utuh --------------------------------------------------------------------------------------------


@needs_tokenizer
def test_main_with_fake_readers_writes_report(tmp_path, monkeypatch, capsys):
    tok = Tokenizer.load(TOKENIZER_JSON)
    original = SyntheticLines.sample

    def sample_with_label(self, idx):
        image, text = original(self, idx)
        image.info["label"] = text  # pembaca palsu membaca label yang ditempel pada citra
        return image, text

    def drop(text: str) -> str:
        return "".join(ch for ch in text if ch not in RARE_SET)

    readers = {"sempurna": lambda text: text, "buta": drop,
               "angka": lambda text: drop(text.replace(WINDU, ZERO).replace(LETTER_E, SIX))}
    monkeypatch.setattr(SyntheticLines, "sample", sample_with_label)
    monkeypatch.setattr(eval_rare, "load_checkpoint", lambda path: (Path(path).stem, tok))
    monkeypatch.setattr(eval_rare, "predict",
                        lambda image, model, tokenizer, **_: readers[model](image.info.get("label", "")))
    monkeypatch.setattr(eval_rare, "read_split", lambda name: LINES * 2)
    monkeypatch.setattr(eval_rare, "load_rare",
                        lambda cache: {"params": {"uji": True}, "codepoints": [{"char": ch} for ch in RARE]})
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    monkeypatch.setattr(eval_rare, "EVAL_DIR", eval_dir)
    paths = {name: tmp_path / f"{name}.pt" for name in readers}
    for name, path in paths.items():
        torch.save({"step": 7, "args": {"rare_insert_prob": 0.3 if name == "sempurna" else 0.0}}, path)
    (eval_dir / "g1.json").write_text(json.dumps({  # laporan G1 resmi palsu untuk satu checkpoint
        "goal": "G1", "augment": "none", "checkpoint": str(paths["sempurna"]),
        "results": {NOTO.name: {"lines": 100, "cer": 0.001}}}), encoding="utf-8")
    out = tmp_path / "out"

    eval_rare.main(["--lines", "6", "--checkpoints", *(f"{name}={path}" for name, path in paths.items()),
                    "--font", str(NOTO), "--out", str(out), "--bootstrap", "200"])
    report = json.loads((out / "rare_synthetic.json").read_text(encoding="utf-8"))
    assert list(report["conditions"]) == list(CONDITIONS)
    info = report["checkpoints"]["sempurna"]
    assert info["step"] == 7 and info["rare_insert_prob"] == 0.3
    assert report["settings"]["lines"] == 6 and report["rare"]["missing_in_font"] == []
    for condition, block in report["conditions"].items():
        results = block["results"]
        assert block["lines"] == 6 and results["sempurna"]["cer"] == 0, condition
        # Hanya membuang aksara langka, atau membacanya sebagai angka: tidak ada kesalahan pada karakter biasa.
        assert results["buta"]["cer_bukan_langka"] == results["angka"]["cer_bukan_langka"] == 0
        windu = results["angka"]["homoglyphs"][0]
        assert windu["x_as_y"] == windu["ref_x"] and results["sempurna"]["homoglyphs"][0]["x_as_y"] == 0
        assert [(c["a"], c["b"]) for c in block["comparisons"]] == [("sempurna", "buta"), ("sempurna", "angka"),
                                                                   ("buta", "angka")]
    langka = report["conditions"]["langka"]
    assert langka["rare_in_reference"] > 0 and 0 < langka["lines_with_opener"] < 6
    assert langka["results"]["sempurna"]["groups"]["langka"]["recall"] == 1.0
    assert langka["results"]["buta"]["groups"]["langka"]["recall"] == 0.0
    assert langka["results"]["buta"]["opener"]["recall"] == 0.0
    first = langka["comparisons"][0]["metrics"]
    assert first["recall_langka"]["diff"] == 1.0 and first["recall_pembuka"]["diff"] == 1.0
    assert report["conditions"]["biasa"]["lines_with_opener"] == 0
    g1_file = (eval_dir / "g1.json").resolve().as_posix()
    assert report["sanity_g1"] == {"sempurna": {"file": g1_file, "lines": 100, "cer": 0.001},
                                   "buta": None, "angka": None}
    assert report["timing"]["estimate_full"]["checkpoints"] == 3

    lines_file = (out / "rare_synthetic_lines.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in lines_file.splitlines()]
    assert len(records) == 6 * len(CONDITIONS)
    assert all(record["hyp"]["sempurna"] == record["reference"] for record in records)
    markdown = (out / "rare_synthetic.md").read_text(encoding="utf-8")
    assert "# Evaluasi sintetis aksara langka" in markdown and "## Kondisi `langka+fase5`" in markdown
    assert "| `sempurna` - `buta` | recall langka | 100.00% | 0.00% | +100.00 pt |" in markdown
    printed = capsys.readouterr().out
    assert "perkiraan jalan penuh 2000 baris x 3 kondisi x 3 checkpoint" in printed
    assert "cek kewajaran sempurna: biasa 0.00% (6 baris) vs G1 resmi 0.10% (100 baris" in printed
