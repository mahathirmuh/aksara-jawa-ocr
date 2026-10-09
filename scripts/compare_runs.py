"""Bandingkan dua pipeline pada baris nyata yang SAMA (kontrak data out/results/, skema 1).

  .venv/Scripts/python scripts/compare_runs.py crnn_fase6_rare crnn_fase6_ctrl
  .venv/Scripts/python scripts/compare_runs.py A B [--results DIR] [--bootstrap 10000] [--seed 0] [--out DIR]

Semua selisih = A - B. CER: negatif berarti A lebih baik. Recall/presisi: positif berarti A lebih baik.

  CER            mikro (total edit / total panjang referensi) dari fungsi resmi src.evaluate.summarize, termasuk
                 varian "spasi dibuang dari keluaran" (cer_hyp_no_space). Dicocokkan dengan manifest.json (beda =
                 skrip berhenti) dan dengan laporan src.evaluate out/eval/<run>_G3_full.json bila ada (beda =
                 peringatan). Juga CER pada baris yang referensinya tanpa aksara langka: efek samping pada teks umum.
  Selang 95%     bootstrap berpasangan, persentil 2,5-97,5: resampling baris, dan resampling halaman (source_id)
                 karena baris sehalaman berkorelasi.
  Berpasangan    jumlah baris (dan halaman) A lebih baik / B lebih baik / sama, uji tanda eksak dua sisi.
  Aksara langka  recall dan presisi per codepoint dari penjajaran karakter referensi-hipotesis ("equal" = benar;
                 biaya Levenshtein minimum, lalu kecocokan terbanyak, lalu substitusi sekelas terbanyak) untuk
                 himpunan yang disisipkan training fase6_rare (RareText.from_lines, 44 codepoint).
  Adeg-adeg      U+A9CB di awal baris referensi: berapa baris, dan pada berapa baris karakter itu terbaca benar.
  Homoglif       aksara/pada langka vs angka Jawa yang glyph-nya (nyaris) identik di sebagian font training: berapa
                 kali yang langka di referensi dijajarkan sebagai substitusi angka (dan sebaliknya), dan berapa kali
                 keluaran memuatnya pada baris yang referensinya tanpa karakter itu.

Keluaran: <out>/<A>_vs_<B>.json dan .md, ringkasan di layar, dan <out>/rare_codepoints.json (himpunan aksara
langka, dihitung sekali dari split train). NusaAksara adalah test set: pakai untuk melaporkan, bukan memilih model.
"""

import argparse
import json
import math
import random
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.align import line_cer  # noqa: E402
from src.evaluate import summarize  # noqa: E402
from src.metrics import alignment, glyph_class  # noqa: E402, F401  (penjajaran; dipindah ke src/metrics.py)
from src.text_augment import RARE_MAX_LINE_FRACTION, RareText  # noqa: E402
from src.tokenizer import TOKENIZER_PATH, Tokenizer, nfc  # noqa: E402
from src.train import SPLITS_DIR, read_split  # noqa: E402

SCHEMA = 1
RESULTS_DIR = ROOT / "out" / "results"
OUT_DIR = ROOT / "out" / "compare"
RARE_CACHE = "rare_codepoints.json"
# Argumen run fase6_rare (out/checkpoints/fase6_rare/log.jsonl); log training: "aksara langka: 44 codepoint".
RARE_TRAIN_LINES, TRAIN_SEED, RARE_EXPECTED = 100_000, 0, 44
ADEG_ADEG = "\uA9CB"  # pembuka baris yang disisipkan --rare-opener-prob (src.text_augment.LINE_OPENERS)
# (aksara, angka) yang glyph-nya identik atau nyaris identik di sebagian font training (tinjauan validitas
# 2026-10-02): pada windu ~ nol, E ~ enam, pa murda ~ delapan, nga lelet ~ dua. Kebingungannya dihitung terpisah.
HOMOGLYPHS = (("\uA9C6", "\uA9D0"), ("\uA98C", "\uA9D6"), ("\uA9A6", "\uA9D8"), ("\uA98A", "\uA9D2"))
HOMOGLYPH_SUBS = frozenset(HOMOGLYPHS) | frozenset((digit, letter) for letter, digit in HOMOGLYPHS)
BOOT_CHUNK = 1000  # resample per potong, supaya memori tidak tumbuh dengan --bootstrap
TOLERANCE = 1e-9


class CompareError(ValueError):
    """Kontrak data tidak lengkap atau tidak konsisten; pesannya menjelaskan apa yang salah."""


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")


def cp_label(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def relative(path: Path) -> str:
    """Path relatif terhadap root repo bila di dalamnya (untuk laporan)."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


# --- data ------------------------------------------------------------------------------------------------


def paired_lines(lines: Sequence[dict], predictions: Sequence[dict], a: str, b: str,
                 common_only: bool = False) -> list[dict]:
    """Baris yang dibandingkan: [{id, source_id, reference, a, b}], semua string NFC.

    Perbandingan berpasangan hanya sah bila kedua pipeline membaca baris yang sama, jadi baris yang hilang di
    salah satu pipeline adalah galat, kecuali `common_only` (mis. uji buta VLM yang hanya 50 baris). Baris tanpa
    source_id menjadi klaster sendiri.
    """
    ids = [line["id"] for line in lines]
    if len(set(ids)) != len(ids):
        raise CompareError("lines.jsonl memuat id baris ganda")
    texts: dict[str, dict[str, dict]] = {a: {}, b: {}}
    for pred in predictions:
        if pred["pipeline"] not in texts:
            continue
        if pred["line"] in texts[pred["pipeline"]]:
            raise CompareError(f"predictions.jsonl: prediksi ganda untuk baris {pred['line']}, pipeline "
                               f"'{pred['pipeline']}'")
        texts[pred["pipeline"]][pred["line"]] = pred
    available = sorted({pred["pipeline"] for pred in predictions})
    for key in (a, b):
        if not texts[key]:
            raise CompareError(f"Pipeline '{key}' tidak punya prediksi di predictions.jsonl "
                               f"(tersedia: {', '.join(available) or '-'})")
        unknown = sorted(set(texts[key]) - set(ids))
        if unknown:
            raise CompareError(f"Pipeline '{key}': {len(unknown)} prediksi untuk baris yang tidak ada di lines.jsonl "
                               f"(mis. {', '.join(unknown[:3])}); kedua berkas bukan dari ekspor yang sama?")
    missing = {key: [i for i in ids if i not in texts[key]] for key in (a, b)}
    if not common_only and any(missing.values()):
        detail = "; ".join(f"'{key}' tidak punya {len(gone)} dari {len(ids)} baris (mis. {', '.join(gone[:3])})"
                           for key, gone in missing.items() if gone)
        raise CompareError(f"Perbandingan berpasangan butuh kedua pipeline pada baris yang sama: {detail}. "
                           "Pakai --common-only untuk membandingkan hanya baris yang dimiliki keduanya.")

    rows = []
    for line in lines:
        if line["id"] not in texts[a] or line["id"] not in texts[b]:
            continue
        row = {"id": line["id"], "source_id": line.get("source_id") or line["id"], "reference": nfc(line["reference"])}
        for side, key in (("a", a), ("b", b)):
            pred = texts[key][line["id"]]
            row[side] = nfc(pred["text"])
            # CER tersimpan dihitung ekspor dari label yang sama; kalau beda, kedua berkas tidak sejalan.
            if pred.get("cer") is not None and abs(pred["cer"] - line_cer(row["reference"], row[side])) > TOLERANCE:
                raise CompareError(f"Baris {line['id']}, pipeline '{key}': CER tersimpan {pred['cer']:.6f} berbeda "
                                   f"dari hitungan ulang {line_cer(row['reference'], row[side]):.6f}; lines.jsonl "
                                   "dan predictions.jsonl bukan dari ekspor yang sama?")
        rows.append(row)
    if not rows:
        raise CompareError(f"Tidak ada baris yang dimiliki '{a}' dan '{b}' sekaligus")
    return rows


# --- per baris -------------------------------------------------------------------------------------------


def mask_from(ops: Sequence[tuple[str, int | None, int | None]], length: int) -> list[bool]:
    """Topeng "terbaca benar" per karakter referensi dari hasil `alignment`."""
    mask = [False] * length
    for tag, i, _ in ops:
        if tag == "equal":
            mask[i] = True
    return mask


def matched_mask(reference: str, hypothesis: str) -> list[bool]:
    """True untuk tiap karakter referensi yang dijajarkan "equal" dengan hipotesis, yaitu terbaca benar.

    Substitusi dan hapus = False. Sisipan tidak punya posisi di referensi: ia hanya menambah jumlah keluaran,
    jadi menurunkan presisi, bukan recall. Penjajaran = `alignment` (biaya minimum, lalu kecocokan terbanyak, lalu
    substitusi sekelas terbanyak), sama untuk kedua pipeline.
    """
    return mask_from(alignment(reference, hypothesis), len(reference))


def score(reference: str, hypothesis: str, codepoints: frozenset[str]) -> dict:
    """Angka satu baris untuk satu pipeline (string NFC): jarak edit dan jumlah per codepoint di `codepoints`."""
    ops = alignment(reference, hypothesis)
    edits = Levenshtein.distance(reference, hypothesis)
    if sum(tag != "equal" for tag, _, _ in ops) != edits:  # penjajaran wajib berbiaya minimum
        raise CompareError(f"Penjajaran {reference!r} vs {hypothesis!r} tidak berbiaya minimum ({edits} edit)")
    mask = mask_from(ops, len(reference))
    homoglyph_subs = Counter()  # substitusi aksara <-> angka homoglif, (referensi, keluaran), penjajaran yang sama
    for tag, i, j in ops:
        if tag == "replace" and (reference[i], hypothesis[j]) in HOMOGLYPH_SUBS:
            homoglyph_subs[reference[i], hypothesis[j]] += 1
    return {
        "edits": edits,
        # Sama dengan cer_hyp_no_space di src/evaluate.py: spasi dibuang dari keluaran saja, label apa adanya.
        # NFC sesudah spasi dibuang, seperti summarize: "pangkon + spasi + cecak telu" bertukar urutan begitu
        # spasinya hilang, dan tanpa ini jumlah per baris tidak mereproduksi angka resmi.
        "edits_no_space": Levenshtein.distance(reference, nfc(hypothesis.replace(" ", ""))),
        "hits": Counter(ch for ch, ok in zip(reference, mask) if ok and ch in codepoints),
        "emitted": Counter(ch for ch in hypothesis if ch in codepoints),
        "first_ok": bool(mask) and mask[0],
        "homoglyph_subs": homoglyph_subs,
    }


def ratio(numerator: float, denominator: float) -> float | None:
    """None bila penyebut 0: codepoint yang tidak muncul tidak punya recall/presisi, bukan 0%."""
    return numerator / denominator if denominator else None


# --- statistik -------------------------------------------------------------------------------------------


def sign_test(first: int, second: int) -> float:
    """p dua sisi uji tanda eksak (binomial, p = 0,5) untuk `first` vs `second` pasangan tak seri."""
    n = first + second
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(first, second) + 1))
    return min(1.0, 2 * tail / 2**n)


def paired_counts(values_a: Sequence[float], values_b: Sequence[float]) -> dict:
    """Satuan dengan nilai A lebih kecil (A lebih baik), B lebih kecil, atau sama; untuk jarak edit."""
    a_better = sum(x < y for x, y in zip(values_a, values_b, strict=True))
    b_better = sum(x > y for x, y in zip(values_a, values_b, strict=True))
    return {"units": len(values_a), "a_better": a_better, "b_better": b_better,
            "tie": len(values_a) - a_better - b_better, "sign_test_p": sign_test(a_better, b_better)}


def bootstrap_diffs(stats: dict[str, tuple], resamples: int, seed: int,
                    clusters: Sequence | None = None) -> dict[str, dict | None]:
    """Selang persentil 95% untuk selisih rasio-jumlah  sum(num_a)/sum(den_a) - sum(num_b)/sum(den_b).

    `stats`: nama -> (num_a, den_a, num_b, den_b), satu angka per baris. Berpasangan: tiap resample memakai
    baris yang sama untuk A dan B, dan semua statistik memakai resample yang sama. Satuan resampling = baris,
    atau klaster bila `clusters` (satu label per baris) diberikan: semua baris seklaster ikut bersama.
    Resample yang penyebutnya 0 dibuang dan terlihat dari `valid`; None bila tidak ada yang tersisa.
    """
    names = list(stats)
    if resamples <= 0 or not names or not len(stats[names[0]][0]):
        return {name: None for name in names}
    values = np.stack([np.asarray(column, dtype=np.float64) for name in names for column in stats[name]], axis=1)
    if clusters is not None:
        _, inverse = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
        grouped = np.zeros((int(inverse.max()) + 1, values.shape[1]))
        np.add.at(grouped, inverse, values)
        values = grouped
    n = values.shape[0]
    rng = np.random.default_rng(seed)
    diffs = np.empty((resamples, len(names)))
    for start in range(0, resamples, BOOT_CHUNK):
        m = min(BOOT_CHUNK, resamples - start)
        # Baris ke-r: berapa kali tiap satuan terambil pada resample r (n pengambilan dengan pengembalian).
        weights = rng.multinomial(n, np.full(n, 1.0 / n), size=m)
        sums = weights @ values
        with np.errstate(divide="ignore", invalid="ignore"):
            diffs[start : start + m] = sums[:, 0::4] / sums[:, 1::4] - sums[:, 2::4] / sums[:, 3::4]

    out: dict[str, dict | None] = {}
    for k, name in enumerate(names):
        d = diffs[:, k]
        d = d[np.isfinite(d)]
        if d.size == 0:
            out[name] = None
            continue
        low, high = np.percentile(d, [2.5, 97.5])
        out[name] = {"ci95": [float(low), float(high)], "se": float(d.std(ddof=1)) if d.size > 1 else 0.0,
                     "valid": int(d.size)}
    return out


def compare(rows: Sequence[dict], rare: Sequence[str], resamples: int = 10_000, seed: int = 0) -> dict:
    """Semua statistik A vs B. `rows` dari paired_lines (NFC); `rare` = codepoint aksara langka."""
    if not rows:
        raise CompareError("Tidak ada baris untuk dibandingkan")
    rare_set = frozenset(rare)
    refs = [row["reference"] for row in rows]
    clusters = [row["source_id"] for row in rows]
    ref_len = [len(ref) for ref in refs]
    ref_rare = [Counter(ch for ch in ref if ch in rare_set) for ref in refs]
    scores = {side: [score(row["reference"], row[side], rare_set) for row in rows] for side in "ab"}

    # Angka resmi dari fungsi jalur resmi; jumlah per baris (bahan bootstrap) harus mereproduksinya.
    official = {side: summarize([(row["reference"], row[side]) for row in rows]) for side in "ab"}
    for side in "ab":
        for field, name in (("edits", "cer"), ("edits_no_space", "cer_hyp_no_space")):
            total = sum(s[field] for s in scores[side]) / max(1, sum(ref_len))
            if abs(total - official[side][name]) > TOLERANCE:
                raise CompareError(f"Jumlah edit per baris ({total:.6%}) tidak mereproduksi "
                                   f"src.evaluate.summarize ({official[side][name]:.6%}) untuk {name} {side.upper()}")

    def column(side: str, field: str) -> list[int]:
        return [s[field] for s in scores[side]]

    def members_per_line(counters: Sequence[Counter], members: frozenset[str]) -> list[int]:
        return [sum(n for ch, n in counter.items() if ch in members) for counter in counters]

    # Baris yang referensinya tanpa aksara langka: efek samping perlakuan pada teks umum. Baris lain diberi bobot 0,
    # bukan dibuang, supaya bootstrap meresampling satuan yang sama dengan statistik lain (jumlah baris tanpa
    # aksara langka ikut bervariasi antar-resample, seperti seharusnya untuk subpopulasi).
    plain = [not counter for counter in ref_rare]

    def only_plain(values: Sequence[int]) -> list[int]:
        return [value if keep else 0 for value, keep in zip(values, plain)]

    stats = {
        "cer": (column("a", "edits"), ref_len, column("b", "edits"), ref_len),
        "cer_hyp_no_space": (column("a", "edits_no_space"), ref_len, column("b", "edits_no_space"), ref_len),
        "cer_without_rare": (only_plain(column("a", "edits")), only_plain(ref_len),
                             only_plain(column("b", "edits")), only_plain(ref_len)),
    }
    murda = frozenset(ch for ch in rare_set if "MURDA" in unicodedata.name(ch, ""))
    groups: dict[str, dict] = {}
    for name, label, members in (("langka", "semua aksara langka", rare_set), ("murda", "aksara murda", murda)):
        if not members:
            continue
        in_ref = members_per_line(ref_rare, members)
        hit = {side: members_per_line(column(side, "hits"), members) for side in "ab"}
        emitted = {side: members_per_line(column(side, "emitted"), members) for side in "ab"}
        stats[f"recall:{name}"] = (hit["a"], in_ref, hit["b"], in_ref)
        stats[f"precision:{name}"] = (hit["a"], emitted["a"], hit["b"], emitted["b"])
        groups[name] = {"label": label, "codepoints": len(members), "ref": sum(in_ref),
                        "hits_a": sum(hit["a"]), "hits_b": sum(hit["b"]),
                        "hyp_a": sum(emitted["a"]), "hyp_b": sum(emitted["b"])}

    # Adeg-adeg pembuka baris: satu karakter per baris, jadi dihitung per baris. Recall pembuka dan tingkat pembuka
    # palsu ikut dibootstrap karena keduanya titik akhir utama eksperimen fase6 (aturan analisis di CLAUDE.md).
    starts = [ref.startswith(ADEG_ADEG) for ref in refs]
    read = {side: column(side, "first_ok") for side in "ab"}
    hyp_starts = {side: [row[side].lstrip(" ").startswith(ADEG_ADEG) for row in rows] for side in "ab"}
    with_opener = [int(s) for s in starts]
    without_opener = [int(not s) for s in starts]
    opener_hits = {side: [int(s and bool(r)) for s, r in zip(starts, read[side])] for side in "ab"}
    false_starts = {side: [int(h and not s) for s, h in zip(starts, hyp_starts[side])] for side in "ab"}
    stats["recall:pembuka"] = (opener_hits["a"], with_opener, opener_hits["b"], with_opener)
    stats["false_start"] = (false_starts["a"], without_opener, false_starts["b"], without_opener)

    by_line = bootstrap_diffs(stats, resamples, seed)
    by_cluster = bootstrap_diffs(stats, resamples, seed, clusters)

    def metric(name: str, a: float | None = None, b: float | None = None) -> dict:
        num_a, den_a, num_b, den_b = (sum(values) for values in stats[name])
        a = ratio(num_a, den_a) if a is None else a
        b = ratio(num_b, den_b) if b is None else b
        return {"a": a, "b": b, "diff": None if a is None or b is None else a - b,
                "bootstrap_lines": by_line[name], "bootstrap_clusters": by_cluster[name]}

    for name, group in groups.items():
        group["recall"] = metric(f"recall:{name}")
        group["precision"] = metric(f"precision:{name}")

    ref_total = sum(ref_rare, Counter())
    hits = {side: sum(column(side, "hits"), Counter()) for side in "ab"}
    emitted = {side: sum(column(side, "emitted"), Counter()) for side in "ab"}
    table = []
    for ch in sorted(rare_set):
        entry = {"codepoint": cp_label(ch), "char": ch, "name": unicodedata.name(ch, "?"), "ref": ref_total[ch]}
        for side in "ab":
            entry[f"hits_{side}"] = hits[side][ch]
            entry[f"recall_{side}"] = ratio(hits[side][ch], ref_total[ch])
            entry[f"hyp_{side}"] = emitted[side][ch]
            entry[f"precision_{side}"] = ratio(hits[side][ch], emitted[side][ch])
        table.append(entry)

    both = sum(s and x and y for s, x, y in zip(starts, read["a"], read["b"]))
    only_a = sum(s and x and not y for s, x, y in zip(starts, read["a"], read["b"]))
    only_b = sum(s and y and not x for s, x, y in zip(starts, read["a"], read["b"]))

    homoglyphs = []
    for letter, digit in HOMOGLYPHS:
        entry = {"letter": cp_label(letter), "letter_char": letter, "letter_name": unicodedata.name(letter, "?"),
                 "digit": cp_label(digit), "digit_char": digit, "digit_name": unicodedata.name(digit, "?"),
                 "ref_letter": sum(ref.count(letter) for ref in refs),
                 "ref_digit": sum(ref.count(digit) for ref in refs)}
        for side in "ab":
            entry[f"letter_as_digit_{side}"] = sum(s["homoglyph_subs"][letter, digit] for s in scores[side])
            entry[f"digit_as_letter_{side}"] = sum(s["homoglyph_subs"][digit, letter] for s in scores[side])
            # Keluaran memuat aksara itu padahal referensi barisnya tidak: jumlah karakter dan jumlah baris.
            false = [row[side].count(letter) for row in rows if letter not in row["reference"]]
            entry[f"false_letter_{side}"] = sum(false)
            entry[f"false_letter_lines_{side}"] = sum(1 for count in false if count)
        homoglyphs.append(entry)

    page_edits: dict[str, list[int]] = {}
    for cluster, x, y in zip(clusters, column("a", "edits"), column("b", "edits")):
        totals = page_edits.setdefault(cluster, [0, 0])
        totals[0] += x
        totals[1] += y
    sizes = Counter(clusters).values()

    return {
        "lines": len(rows),
        "clusters": len(page_edits),
        "cluster_sizes": {"min": min(sizes), "median": median(sizes), "max": max(sizes)},
        "reference_chars": sum(ref_len),
        "cer": metric("cer", official["a"]["cer"], official["b"]["cer"]),
        "cer_hyp_no_space": metric("cer_hyp_no_space", official["a"]["cer_hyp_no_space"],
                                   official["b"]["cer_hyp_no_space"]),
        "cer_without_rare": {"lines": sum(plain), "reference_chars": sum(only_plain(ref_len)),
                             **metric("cer_without_rare")},
        "exact_line_accuracy": {side: official[side]["exact_line_accuracy"] for side in "ab"},
        # Referensi sama untuk A dan B, jadi urutan jarak edit = urutan CER baris.
        "paired_lines": paired_counts(column("a", "edits"), column("b", "edits")),
        "paired_clusters": paired_counts([t[0] for t in page_edits.values()], [t[1] for t in page_edits.values()]),
        "rare": {
            "codepoints": len(rare_set),
            "in_reference": sum(1 for entry in table if entry["ref"]),
            "groups": groups,
            "per_codepoint": sorted((entry for entry in table if entry["ref"]), key=lambda entry: -entry["ref"]),
            "only_in_hypotheses": [e for e in table if not e["ref"] and (e["hyp_a"] or e["hyp_b"])],
            "absent": [e["codepoint"] for e in table if not e["ref"] and not e["hyp_a"] and not e["hyp_b"]],
        },
        "line_initial_adeg_adeg": {
            "codepoint": cp_label(ADEG_ADEG), "lines": sum(starts),
            "read_a": both + only_a, "read_b": both + only_b,
            "both": both, "only_a": only_a, "only_b": only_b, "neither": sum(starts) - both - only_a - only_b,
            "sign_test_p": sign_test(only_a, only_b),
            "recall": metric("recall:pembuka"),
            # Keluaran berawalan adeg-adeg padahal referensi tidak: sisi presisi dari pembuka baris.
            "lines_without": len(rows) - sum(starts),
            "false_start_a": sum(h and not s for s, h in zip(starts, hyp_starts["a"])),
            "false_start_b": sum(h and not s for s, h in zip(starts, hyp_starts["b"])),
            "false_start_rate": metric("false_start"),
        },
        "homoglyphs": homoglyphs,
    }


# --- himpunan aksara langka ------------------------------------------------------------------------------


def rare_params() -> dict:
    """Penentu himpunan aksara langka; cache hanya dipakai bila semuanya masih sama."""
    split = SPLITS_DIR / "train.txt"
    return {
        "split": relative(split), "split_bytes": split.stat().st_size if split.exists() else None,
        "tokenizer": relative(TOKENIZER_PATH),
        "tokenizer_bytes": TOKENIZER_PATH.stat().st_size if TOKENIZER_PATH.exists() else None,
        "train_lines": RARE_TRAIN_LINES, "seed": TRAIN_SEED, "max_line_fraction": RARE_MAX_LINE_FRACTION,
    }


def rare_from_training() -> list[dict]:
    """Aksara langka persis seperti src/train.py::build_datasets pada run fase6_rare (mahal: baca split train)."""
    lines = read_split("train")
    random.Random(TRAIN_SEED).shuffle(lines)
    lines = lines[:RARE_TRAIN_LINES]
    rare = RareText.from_lines(lines, Tokenizer.load(TOKENIZER_PATH).charset).rare
    containing = Counter(ch for line in lines for ch in set(line))
    return [{"codepoint": cp_label(ch), "char": ch, "name": unicodedata.name(ch, "?"),
             "train_lines_with": containing[ch]} for ch in rare]


def load_rare(cache: Path, refresh: bool = False, expected: int = RARE_EXPECTED) -> dict:
    """Himpunan aksara langka dari cache; dihitung lalu disimpan bila cache belum ada atau penentunya berubah."""
    params = rare_params()
    data = read_json(cache) if cache.exists() and not refresh else {}
    stale = data.get("params") != params
    if stale:
        print(f"menghitung himpunan aksara langka dari split train (sekali; disimpan di {relative(cache)})", flush=True)
        codepoints = rare_from_training()
        data = {"source": "RareText.from_lines (src/text_augment.py), cara yang sama dengan "
                          "src/train.py::build_datasets: split train, shuffle random.Random(seed), train_lines "
                          "baris pertama",
                "params": params, "count": len(codepoints), "codepoints": codepoints}
    if len(data["codepoints"]) != expected:
        raise CompareError(f"Himpunan aksara langka berisi {len(data['codepoints'])} codepoint, sedangkan log "
                           f"training fase6_rare mencatat {expected}; split train atau tokenizer berubah?")
    if stale:
        write_json(cache, data)
    return data


# --- cek angka -------------------------------------------------------------------------------------------


def check_manifest(manifest: dict, key: str, total_lines: int, value: dict) -> dict | None:
    """Angka manifest untuk seluruh baris harus sama dengan hitungan ulang; beda = galat."""
    if value["lines"] != total_lines:
        return None  # subset: manifest tidak punya angka pembanding
    found = None
    for m in manifest.get("metrics", []):
        if m.get("pipeline") != key or m.get("lines") != total_lines or "cer_no_space" not in m:
            continue
        for name, mine in (("cer", value["cer"]), ("cer_no_space", value["cer_hyp_no_space"])):
            if abs(m[name] - mine) > TOLERANCE:
                raise CompareError(f"'{key}': {name} {mine:.6%} berbeda dari manifest.json {m[name]:.6%} (scope "
                                   f"{m.get('scope')}); lines.jsonl, predictions.jsonl, dan manifest.json bukan "
                                   "dari ekspor yang sama?")
        found = {"scope": m.get("scope"), "match": True}
    return found


def check_official(eval_dir: Path, key: str, value: dict) -> dict | None:
    """Bandingkan dengan laporan src.evaluate <eval>/<run>_G3_full.json bila ada (run = kunci tanpa 'crnn_')."""
    path = eval_dir / f"{key.removeprefix('crnn_')}_G3_full.json"
    block = read_json(path).get("results", {}).get("semua") if path.exists() else None
    if not block:
        return None
    out = {"file": relative(path), "lines": block["lines"], "cer": block["cer"],
           "cer_hyp_no_space": block["cer_hyp_no_space"]}
    if block["lines"] != value["lines"]:
        return {**out, "match": None}
    return {**out, "match": abs(block["cer"] - value["cer"]) <= TOLERANCE
            and abs(block["cer_hyp_no_space"] - value["cer_hyp_no_space"]) <= TOLERANCE}


# --- laporan ---------------------------------------------------------------------------------------------


def pct(value) -> str:
    return "-" if value is None else f"{value:.2%}"


def points(value) -> str:
    return "-" if value is None else f"{value * 100:+.2f} pt"


def interval(boot: dict | None) -> str:
    return "-" if not boot else f"[{boot['ci95'][0] * 100:+.2f}, {boot['ci95'][1] * 100:+.2f}] pt"


def check_text(checks: dict) -> str:
    parts = []
    for side in "ab":
        manifest, official = checks[side]["manifest"], checks[side]["official"]
        text = f"manifest {'cocok' if manifest else 'tanpa pembanding'}"
        if official is None:
            text += ", laporan src.evaluate tidak ada"
        elif official["match"] is None:
            text += f", `{official['file']}` berisi {official['lines']} baris (tidak dibandingkan)"
        else:
            text += f", `{official['file']}` {'cocok' if official['match'] else 'BERBEDA'}"
        parts.append(f"{side.upper()}: {text}")
    return "; ".join(parts)


def metric_cells(m: dict) -> str:
    return (f"{pct(m['a'])} | {pct(m['b'])} | {points(m['diff'])} | {interval(m['bootstrap_lines'])} | "
            f"{interval(m['bootstrap_clusters'])}")


def short_name(name: str) -> str:
    return name.removeprefix("JAVANESE ")


def homoglyph_header(e: dict) -> str:
    return (f"{e['letter']} {e['letter_char']} {short_name(e['letter_name'])} ~ "
            f"{e['digit']} {e['digit_char']} {short_name(e['digit_name'])}")


def render_markdown(report: dict) -> str:
    a, b, rare, opener = report["a"], report["b"], report["rare"], report["line_initial_adeg_adeg"]
    sizes, boot, plain = report["cluster_sizes"], report["bootstrap"], report["cer_without_rare"]
    datasets = ", ".join(f"{name} {n}" for name, n in report["datasets"].items())
    out = [
        f"# Perbandingan pipeline: `{a['key']}` (A) vs `{b['key']}` (B)",
        "",
        f"- A = {a['label']}: {a['config']}",
        f"- B = {b['label']}: {b['config']}",
        f"- Data: `{report['results']['dir']}` (ekspor {report['results']['generated']}"
        f"{', TERBATAS --limit' if report['results']['limited'] else ''}), {report['lines']} baris "
        f"nyata ({datasets}; {report['subset']}), {report['reference_chars']} karakter referensi, "
        f"{report['clusters']} halaman (`source_id`; {sizes['min']}-{sizes['max']} baris per halaman, "
        f"median {sizes['median']:g}).",
        "- Selisih = A - B. CER: negatif berarti A lebih baik. Recall/presisi: positif berarti A lebih baik.",
        f"- SK 95% = bootstrap berpasangan, {boot['resamples']} resample, seed {boot['seed']}, persentil 2,5-97,5. "
        "\"baris\" meresampling baris; \"halaman\" meresampling halaman beserta semua barisnya. Baris sehalaman "
        "berkorelasi, jadi SK halaman yang dipakai untuk menyimpulkan.",
        f"- Cek angka: {check_text(report['checks'])}.",
        "",
        "## CER",
        "",
        "| metrik | A | B | A - B | SK 95% (baris) | SK 95% (halaman) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| CER (definisi jalur resmi) | {metric_cells(report['cer'])} |",
        f"| CER, spasi dibuang dari keluaran (bukan angka gerbang) | {metric_cells(report['cer_hyp_no_space'])} |",
        f"| CER, hanya {plain['lines']} baris yang referensinya tanpa aksara langka ({plain['reference_chars']} "
        f"karakter; efek samping pada teks umum) | {metric_cells(plain)} |",
        f"| baris persis sama | {pct(report['exact_line_accuracy']['a'])} | "
        f"{pct(report['exact_line_accuracy']['b'])} | | | |",
        "",
        "## Berpasangan",
        "",
        "| satuan | jumlah | A lebih baik | B lebih baik | sama | uji tanda dua sisi |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, p in (("baris", report["paired_lines"]), ("halaman", report["paired_clusters"])):
        out.append(f"| {label} | {p['units']} | {p['a_better']} | {p['b_better']} | {p['tie']} | "
                   f"p = {p['sign_test_p']:.3g} |")
    out += [
        "",
        "Lebih baik = jarak edit lebih kecil pada baris itu (per halaman: jumlah edit sehalaman). Uji tanda "
        "membuang yang sama; uji per baris menganggap baris saling bebas.",
        "",
        "## Aksara langka",
        "",
        f"Himpunan = {rare['codepoints']} codepoint yang disisipkan training fase6_rare (`{rare['source']['cache']}`: "
        f"muncul di < {rare['source']['params']['max_line_fraction']:.1%} dari "
        f"{rare['source']['params']['train_lines']} baris train). {rare['in_reference']} di antaranya muncul di "
        "referensi; yang tidak muncul tidak bisa dinilai dan tidak dihitung sebagai 0%. Benar = dijajarkan "
        "\"equal\" pada penjajaran karakter referensi-keluaran (biaya Levenshtein minimum, lalu kecocokan "
        "terbanyak, lalu substitusi sekelas terbanyak); recall = benar / jumlah di referensi, presisi = benar / "
        "jumlah di keluaran (\"-\" = penyebut 0).",
        "",
        "| kelompok | ukuran | A | B | A - B | SK 95% (baris) | SK 95% (halaman) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in rare["groups"].values():
        out.append(f"| {group['label']} ({group['codepoints']} codepoint) | recall, {group['ref']} di referensi | "
                   f"{metric_cells(group['recall'])} |")
        out.append(f"| | presisi, {group['hyp_a']} / {group['hyp_b']} di keluaran A / B | "
                   f"{metric_cells(group['precision'])} |")
    codepoint_header = [
        "| codepoint | nama Unicode | referensi | recall A | recall B | keluaran A | presisi A | keluaran B | "
        "presisi B |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def codepoint_row(e: dict) -> str:
        return (f"| {e['codepoint']} {e['char']} | {e['name'].removeprefix('JAVANESE ')} | {e['ref']} | "
                f"{pct(e['recall_a'])} | {pct(e['recall_b'])} | {e['hyp_a']} | {pct(e['precision_a'])} | "
                f"{e['hyp_b']} | {pct(e['precision_b'])} |")

    out += ["", *codepoint_header, *[codepoint_row(e) for e in rare["per_codepoint"]]]
    if rare["only_in_hypotheses"]:
        out += ["", "Hanya di keluaran (tidak ada di referensi, jadi setiap kemunculannya salah):", "",
                *codepoint_header, *[codepoint_row(e) for e in rare["only_in_hypotheses"]]]
    out += [
        "",
        f"Tidak muncul di referensi maupun keluaran ({len(rare['absent'])}): {', '.join(rare['absent']) or '-'}.",
        "",
        f"## Adeg-adeg ({opener['codepoint']}) di awal baris",
        "",
        f"{opener['lines']} baris referensi berawalan adeg-adeg. Karakter pertama itu terbaca benar pada "
        f"**{opener['read_a']}** baris oleh A dan **{opener['read_b']}** baris oleh B (keduanya {opener['both']}, "
        f"A saja {opener['only_a']}, B saja {opener['only_b']}, tidak keduanya {opener['neither']}; uji tanda dua "
        f"sisi pada yang berbeda p = {opener['sign_test_p']:.3g}). Keluaran berawalan adeg-adeg padahal referensi "
        f"tidak: A {opener['false_start_a']}, B {opener['false_start_b']} dari {opener['lines_without']} baris.",
        "",
        "| metrik | A | B | A - B | SK 95% (baris) | SK 95% (halaman) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| recall pembuka ({opener['lines']} baris) | {metric_cells(opener['recall'])} |",
        f"| pembuka palsu ({opener['lines_without']} baris; negatif = A lebih sedikit) | "
        f"{metric_cells(opener['false_start_rate'])} |",
        "",
        "## Homoglif aksara-angka",
        "",
        "Pasangan yang glyph-nya identik atau nyaris identik di sebagian font training, dihitung terpisah dari CER. "
        "X -> Y = X di referensi dijajarkan sebagai substitusi Y di keluaran (penjajaran yang sama dengan recall); "
        "X palsu = jumlah X di keluaran pada baris yang referensinya tanpa X (dalam kurung: jumlah baris). Satu "
        f"substitusi = 1 edit = {1 / max(1, report['reference_chars']):.4%} CER pada {report['reference_chars']} "
        "karakter referensi.",
        "",
        "| aksara X ~ angka Y | di referensi X / Y | X -> Y (A / B) | Y -> X (A / B) | X palsu A | X palsu B |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {homoglyph_header(e)} | {e['ref_letter']} / {e['ref_digit']} | {e['letter_as_digit_a']} / "
          f"{e['letter_as_digit_b']} | {e['digit_as_letter_a']} / {e['digit_as_letter_b']} | {e['false_letter_a']} "
          f"({e['false_letter_lines_a']} baris) | {e['false_letter_b']} ({e['false_letter_lines_b']} baris) |"
          for e in report["homoglyphs"]],
        "",
        "## Batasan",
        "",
        "- Satu run per pipeline: SK hanya memuat keragaman sampel baris/halaman, bukan keragaman antar-run "
        "training (seed, urutan batch, render acak).",
        "- Penjajaran berbiaya minimum tidak selalu tunggal. Dipilih yang kecocokannya terbanyak (opcodes rapidfuzz "
        "saja kehilangan 1-2% karakter benar pada keluaran nyata: aksara hilang + spasi tambahan terbaca sebagai dua "
        "substitusi), lalu yang substitusinya paling banyak sekelas (aksara/angka/pada dengan sesamanya, sandhangan "
        "dengan sandhangan, spasi dengan spasi); tanpa itu homoglif yang diikuti spasi tambahan atau sandhangan yang "
        "hilang tidak tercatat. Bila masih seri, recall per codepoint yang jumlahnya kecil bisa bergeser satu-dua "
        "karakter.",
    ]
    if "nusaaksara" in report["datasets"]:
        out += [
            "- NusaAksara adalah test set; membandingkan banyak run di sini lalu memilih yang terbaik membuat "
            "angkanya bias optimistis.",
            "- Halaman 0-86 NusaAksara berasal dari satu majalah, jadi halaman pun tidak sepenuhnya saling bebas: "
            "SK halaman menggambarkan keragaman di dalam sumber ini, bukan cetakan lain.",
        ]
    return "\n".join(out) + "\n"


def summary_lines(report: dict) -> list[str]:
    a, b, rare, opener = report["a"], report["b"], report["rare"], report["line_initial_adeg_adeg"]

    def row(label: str, m: dict) -> str:
        return (f"{label:34} A {pct(m['a']):>8}  B {pct(m['b']):>8}  A-B {points(m['diff']):>10}  "
                f"SK95 baris {interval(m['bootstrap_lines'])}  halaman {interval(m['bootstrap_clusters'])}")

    out = [
        f"A = {a['key']}  |  B = {b['key']}  |  {report['lines']} baris ({report['subset']}), "
        f"{report['clusters']} halaman; selisih = A - B",
        row("CER", report["cer"]),
        row("CER spasi dibuang dari keluaran", report["cer_hyp_no_space"]),
        row(f"CER baris tanpa aksara langka (n={report['cer_without_rare']['lines']})", report["cer_without_rare"]),
    ]
    for label, p in (("baris", report["paired_lines"]), ("halaman", report["paired_clusters"])):
        out.append(f"{label}: A lebih baik {p['a_better']}, B lebih baik {p['b_better']}, sama {p['tie']}; "
                   f"uji tanda dua sisi p = {p['sign_test_p']:.3g}")
    out.append(f"aksara langka: {rare['codepoints']} codepoint, {rare['in_reference']} muncul di referensi")
    for group in rare["groups"].values():
        out.append(row(f"recall {group['label']} (n={group['ref']})", group["recall"]))
        out.append(row(f"presisi (keluaran {group['hyp_a']}/{group['hyp_b']})", group["precision"]))
    for e in rare["per_codepoint"]:
        out.append(f"  {e['codepoint']} {e['name'].removeprefix('JAVANESE '):24} ref {e['ref']:4}  recall A "
                   f"{pct(e['recall_a']):>7} B {pct(e['recall_b']):>7}  presisi A {pct(e['precision_a']):>7} "
                   f"({e['hyp_a']}) B {pct(e['precision_b']):>7} ({e['hyp_b']})")
    for e in rare["only_in_hypotheses"]:
        out.append(f"  {e['codepoint']} {e['name'].removeprefix('JAVANESE '):24} hanya di keluaran: A {e['hyp_a']}, "
                   f"B {e['hyp_b']}")
    out.append(f"adeg-adeg di awal baris: {opener['lines']} baris; terbaca A {opener['read_a']}, B "
               f"{opener['read_b']} (A saja {opener['only_a']}, B saja {opener['only_b']}, p = "
               f"{opener['sign_test_p']:.3g}); awalan adeg-adeg palsu A {opener['false_start_a']}, B "
               f"{opener['false_start_b']} dari {opener['lines_without']} baris")
    out.append(row(f"recall pembuka (n={opener['lines']})", opener["recall"]))
    out.append(row(f"pembuka palsu (n={opener['lines_without']})", opener["false_start_rate"]))
    for e in report["homoglyphs"]:
        out.append(f"homoglif {homoglyph_header(e)}: di referensi {e['ref_letter']}/{e['ref_digit']}; X->Y A "
                   f"{e['letter_as_digit_a']} B {e['letter_as_digit_b']}; Y->X A {e['digit_as_letter_a']} B "
                   f"{e['digit_as_letter_b']}; X palsu A {e['false_letter_a']} ({e['false_letter_lines_a']} baris) "
                   f"B {e['false_letter_b']} ({e['false_letter_lines_b']} baris)")
    out.append(f"cek angka: {check_text(report['checks'])}")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", metavar="A", help="kunci pipeline di manifest.json, mis. crnn_fase6_rare")
    parser.add_argument("b", metavar="B", help="kunci pipeline pembanding, mis. crnn_fase6_ctrl")
    parser.add_argument("--results", default=str(RESULTS_DIR),
                        help="direktori kontrak data dari scripts/export_results.py (default out/results)")
    parser.add_argument("--bootstrap", type=int, default=10_000, help="jumlah resample; 0 = tanpa selang kepercayaan")
    parser.add_argument("--seed", type=int, default=0, help="seed bootstrap")
    parser.add_argument("--out", default=str(OUT_DIR), help="direktori keluaran (default out/compare)")
    parser.add_argument("--common-only", action="store_true",
                        help="bandingkan hanya baris yang dimiliki kedua pipeline (mis. uji buta 50 baris)")
    parser.add_argument("--refresh-rare", action="store_true", help="hitung ulang himpunan aksara langka")
    args = parser.parse_args(argv)
    if args.a == args.b:
        parser.error("A dan B harus pipeline yang berbeda")
    if args.bootstrap < 0:
        parser.error("--bootstrap tidak boleh negatif")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    started = time.time()
    results_dir, out_dir = Path(args.results), Path(args.out)

    try:
        for name in ("manifest.json", "lines.jsonl", "predictions.jsonl"):
            if not (results_dir / name).exists():
                raise CompareError(f"{results_dir / name} tidak ada; jalankan scripts/export_results.py dulu")
        manifest = read_json(results_dir / "manifest.json")
        if manifest.get("schema") != SCHEMA:
            raise CompareError(f"manifest.json: skema {manifest.get('schema')}, skrip ini untuk skema {SCHEMA}")
        pipelines = {p["key"]: p for p in manifest.get("pipelines", [])}
        for key in (args.a, args.b):
            if key not in pipelines:
                raise CompareError(f"Pipeline '{key}' tidak ada di manifest.json (tersedia: {', '.join(pipelines)})")
        lines = read_jsonl(results_dir / "lines.jsonl")
        rows = paired_lines(lines, read_jsonl(results_dir / "predictions.jsonl"), args.a, args.b, args.common_only)
        rare = load_rare(out_dir / RARE_CACHE, args.refresh_rare)
        stats = compare(rows, [entry["char"] for entry in rare["codepoints"]], args.bootstrap, args.seed)

        checks = {}
        for side, key in (("a", args.a), ("b", args.b)):
            value = {"lines": stats["lines"], "cer": stats["cer"][side],
                     "cer_hyp_no_space": stats["cer_hyp_no_space"][side]}
            checks[side] = {"manifest": check_manifest(manifest, key, len(lines), value),
                            "official": check_official(results_dir.parent / "eval", key, value)}
    except CompareError as error:
        raise SystemExit(f"galat: {error}") from None

    compared = {row["id"] for row in rows}
    stats["rare"]["source"] = {"cache": relative(out_dir / RARE_CACHE), "params": rare["params"]}
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "results": {"dir": relative(results_dir), "schema": manifest["schema"],
                    "generated": manifest.get("generated"), "limited": bool(manifest.get("limited"))},
        "a": {k: pipelines[args.a].get(k) for k in ("key", "label", "config")},
        "b": {k: pipelines[args.b].get(k) for k in ("key", "label", "config")},
        "subset": "semua baris" if len(rows) == len(lines) else f"hanya {len(rows)} dari {len(lines)} baris",
        "datasets": dict(Counter(line.get("dataset", "?") for line in lines if line["id"] in compared)),
        "lines_without_source_id": sum(1 for line in lines if line["id"] in compared and not line.get("source_id")),
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "interval": "persentil 2,5-97,5"},
        "checks": checks,
        **stats,
    }
    stem = f"{args.a}_vs_{args.b}"
    write_json(out_dir / f"{stem}.json", report)
    (out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")

    print("\n".join(summary_lines(report)))
    for side, key in (("a", args.a), ("b", args.b)):
        official = checks[side]["official"]
        if official and official["match"] is False:
            print(f"PERINGATAN: '{key}' CER {stats['cer'][side]:.6%} berbeda dari laporan resmi {official['file']} "
                  f"({official['cer']:.6%}); ekspor dan evaluasi resmi memakai checkpoint yang berbeda?")
    if report["lines_without_source_id"]:
        print(f"PERINGATAN: {report['lines_without_source_id']} baris tanpa source_id dihitung sebagai halaman sendiri")
    if report["results"]["limited"]:
        print("PERINGATAN: kontrak data hasil ekspor terbatas (--limit), bukan seluruh baris nyata")
    print(f"hasil: {relative(out_dir / stem)}.json, .md ({time.time() - started:.0f} dtk)")


if __name__ == "__main__":
    main()
