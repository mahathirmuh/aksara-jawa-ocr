"""Evaluasi sintetis tertarget aksara langka: citra yang SAMA untuk setiap checkpoint, tanpa NusaAksara.

  .venv/Scripts/python scripts/eval_rare.py                            # 2000 baris x 3 kondisi x 3 checkpoint
  .venv/Scripts/python scripts/eval_rare.py --lines 20 --checkpoints fase6_rare fase5_fonts
  .venv/Scripts/python scripts/eval_rare.py --checkpoints A=path/a.pt B=path/b.pt [--conditions biasa langka]
      [--seed 0] [--workers 2] [--threads 0] [--bootstrap 10000] [--out DIR]

Titik akhir utama eksperimen fase6_rare vs fase6_ctrl. Efek langsung aksara langka pada G3 paling banyak ~2,4 poin
dan derau antar-run belum terukur; di sini setiap checkpoint membaca citra yang sama (berpasangan) dan setiap
codepoint langka muncul ~N/44 kali per kondisi.

  Baris      sampel acak split test: indeks diacak dengan --seed, ambil N pertama (sampel kecil = awal sampel besar).
             Bukan train/val, bukan NusaAksara.
  Render     SyntheticLines(deterministic=True, seed=--seed), satu font held-out (javatext.ttf, tidak pernah merender
             data training), ukuran 56-72 px seperti G1, spasi tidak dibuang. Baris yang melanggar T >= 1,5L ditolak
             seperti G1. Seed sama untuk semua kondisi: per indeks, teks dasar dan ukuran sama.
  Kondisi    biasa         teks test apa adanya, bersih: efek samping pada teks umum (mirip G1, sampel lain).
             langka        RareText dengan 44 codepoint yang sama dengan training fase6_rare (compare_runs.load_rare):
                           sisip p=1 (paling banyak satu per baris), pembuka adeg-adeg p=0,5, supaya setengah baris
                           mengukur recall pembuka dan setengah lagi pembuka palsu. Bersih.
             langka+fase5  teks dan ukuran sama persis dengan langka, ditambah augmentasi preset fase5 (deterministik
                           per indeks).
  Inferensi  src.infer.predict satu baris: greedy, to_tensor, urutan logis, NFC, pad_ratio 0 (jalur G3 resmi; G1
             resmi memakai jalur batch src.evaluate).

Metrik per checkpoint x kondisi (NFC, urutan logis):
  CER              mikro dari src.evaluate.summarize: total edit / total panjang referensi.
  CER bukan-langka kesalahan di luar aksara langka / jumlah karakter bukan-langka referensi. Langkah penjajaran
                   tak-sama yang bersebelahan membentuk blok; blok yang memuat substitusi/hapus aksara langka referensi
                   = bacaan glyph itu (bisa beberapa karakter, termasuk spasi sisipan " X"), sudah diukur recall, jadi
                   tidak dihitung. Sisipan aksara langka (keluaran palsu) juga tidak, sudah diukur presisi. Hanya baris
                   tanpa pembuka: pembuka yang terbaca taling bisa pindah ke belakang aksara pertama (urutan logis).
                   Efek samping pada teks umum yang paling bersih tetap CER kondisi biasa.
  Langka           benar = "equal" pada penjajaran compare_runs.alignment (biaya Levenshtein minimum, lalu kecocokan
                   terbanyak; sama dengan compare_runs.matched_mask). Recall = benar / jumlah di referensi, presisi =
                   benar / jumlah di keluaran; kelompok semua aksara langka dan aksara murda, dan per codepoint beserta
                   bacaan salah tersering. Adeg-adeg di awal baris TIDAK termasuk: dengan p=0,5 jumlahnya ~N/2 dan
                   akan menenggelamkan codepoint lain, jadi dilaporkan sendiri (Pembuka).
  Pembuka          baris referensi berawalan adeg-adeg yang karakter pertamanya terbaca benar. Palsu: keluaran
                   berawalan adeg-adeg (spasi di depan diabaikan) pada baris yang referensinya tanpa pembuka.
  Homoglif         substitusi X -> Y dan Y -> X pada penjajaran yang sama: pada windu / angka 0, E / angka 6,
                   pa murda / angka 8, nga lelet / angka 2.
Selisih A - B untuk setiap pasangan checkpoint (A = yang lebih dulu di --checkpoints): bootstrap berpasangan per baris
(compare_runs.bootstrap_diffs, persentil 2,5-97,5) dan uji tanda per baris.

Keluaran: <out>/rare_synthetic.json, .md, rare_synthetic_lines.jsonl (teks per baris), ringkasan di layar termasuk
ms/baris dan perkiraan waktu jalan penuh. Cek kewajaran: CER kondisi biasa dibandingkan dengan laporan G1 resmi
(out/eval) untuk checkpoint yang sama.
"""

import argparse
import json
import random
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))  # compare_runs; scripts/ bukan paket

from compare_runs import (  # noqa: E402
    ADEG_ADEG,
    HOMOGLYPHS,
    OUT_DIR,
    RARE_CACHE,
    TOLERANCE,
    CompareError,
    alignment,
    bootstrap_diffs,
    cp_label,
    interval,
    load_rare,
    mask_from,
    paired_counts,
    pct,
    points,
    ratio,
    read_json,
    relative,
    sign_test,
    write_json,
)
from src.augment import build_augment  # noqa: E402
from src.dataset import RENDER_SIZE_RANGE, SyntheticLines, is_trainable, to_tensor  # noqa: E402
from src.evaluate import summarize  # noqa: E402
from src.infer import load_checkpoint, predict  # noqa: E402
from src.render import EmptyRender, RenderClipped  # noqa: E402
from src.text_augment import RareText, font_codepoints  # noqa: E402
from src.tokenizer import TOKENIZER_PATH, WINDOWS_JAVANESE_TEXT, Tokenizer, nfc  # noqa: E402
from src.train import read_split  # noqa: E402

DEFAULT_LINES = 2000
CHECKPOINT_DIR = ROOT / "out" / "checkpoints"
SNAPSHOT = "last_snapshot.pt"  # salinan last.pt di akhir run (scripts/fase6_common.sh::evaluate)
# Urutan = arah selisih: pasangan (A, B) mengikuti urutan ini, jadi fase6_rare - fase6_ctrl lebih dulu.
DEFAULT_CHECKPOINTS = ("fase6_rare", "fase6_ctrl", "fase5_fonts")
EVAL_DIR = ROOT / "out" / "eval"
STEM = "rare_synthetic"
CONDITIONS = {
    "biasa": {"insert_prob": 0.0, "opener_prob": 0.0, "augment": "none",
              "purpose": "Teks test apa adanya, bersih: efek samping pada teks umum (mirip G1, sampel lain)"},
    "langka": {"insert_prob": 1.0, "opener_prob": 0.5, "augment": "none",
               "purpose": "Paling banyak satu aksara langka per baris, setengah baris berpembuka adeg-adeg, bersih"},
    "langka+fase5": {"insert_prob": 1.0, "opener_prob": 0.5, "augment": "fase5",
                     "purpose": "Teks dan ukuran sama persis dengan `langka`, ditambah augmentasi preset fase5"},
}
# Pasangan (aksara, angka) dari compare_runs: pada windu / 0, E / 6, pa murda / 8, nga lelet / 2.
HOMOGLYPH_CHARS = frozenset(ch for pair in HOMOGLYPHS for ch in pair)
METRICS = {  # kunci -> label; CER, CER bukan-langka, dan pembuka palsu: lebih kecil = lebih baik
    "cer": "CER",
    "cer_bukan_langka": "CER bukan-langka",
    "recall_langka": "recall langka",
    "presisi_langka": "presisi langka",
    "recall_murda": "recall murda",
    "recall_pembuka": "pembuka terbaca",
    "pembuka_palsu": "pembuka palsu",
}
TOP_MISREADS = 3
PROGRESS_EVERY = 100
WARMUP_SIZE = (384, 96)  # citra polos untuk satu forward pemanasan per model sebelum waktu diukur


class EvalError(ValueError):
    """Masukan tidak lengkap atau tidak konsisten; pesannya menjelaskan apa yang salah."""


# --- masukan ---------------------------------------------------------------------------------------------


def sample_indices(total: int, n: int, seed: int) -> list[int]:
    """Indeks baris acak tetap: urutan acak yang sama untuk setiap N, jadi sampel kecil = awal sampel besar.

    random.sample tidak menjamin itu (algoritmanya berganti menurut N). n <= 0 = semua baris.
    """
    order = list(range(total))
    random.Random(seed).shuffle(order)
    return order if n <= 0 else order[:n]


def parse_checkpoints(specs: Sequence[str]) -> dict[str, Path]:
    """'NAMA=PATH' atau 'NAMA' (= out/checkpoints/<NAMA>/last_snapshot.pt) -> {nama: path}; semua berkas harus ada."""
    out: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = (part.strip() for part in spec.partition("="))
        if not name or (sep and not path):
            raise EvalError(f"--checkpoints: {spec!r} bukan NAMA atau NAMA=PATH")
        if name in out:
            raise EvalError(f"--checkpoints: nama '{name}' disebut dua kali")
        out[name] = Path(path) if sep else CHECKPOINT_DIR / name / SNAPSHOT
    missing = [f"{name} ({relative(path)})" for name, path in out.items() if not path.is_file()]
    if missing:
        raise EvalError(f"checkpoint tidak ada: {', '.join(missing)}; training belum selesai? Pilih dengan "
                        "--checkpoints")
    return out


def checkpoint_info(path: Path) -> dict:
    """Langkah dan argumen training yang membedakan run, untuk laporan."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args") or {}
    stat = path.stat()
    return {"path": relative(path), "step": ckpt.get("step"), "bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            **{key: args.get(key) for key in ("init", "augment", "extra_fonts", "drop_space_prob",
                                               "rare_insert_prob", "rare_opener_prob")}}


def load_models(checkpoints: dict[str, Path], tokenizer: Tokenizer) -> dict[str, tuple]:
    """(model CPU, tokenizer) per checkpoint; charset-nya harus sama dengan tokenizer yang membuat label render."""
    models = {}
    for name, path in checkpoints.items():
        model, ckpt_tokenizer = load_checkpoint(path)
        if list(ckpt_tokenizer.charset) != list(tokenizer.charset):
            raise EvalError(f"charset checkpoint '{name}' berbeda dari {relative(TOKENIZER_PATH)}")
        models[name] = (model, ckpt_tokenizer)
    return models


# --- render dan baca -------------------------------------------------------------------------------------


def condition_dataset(lines: Sequence[str], tokenizer: Tokenizer, font: Path, rare: Sequence[str],
                      common_letters: Sequence[str], condition: str, seed: int) -> SyntheticLines:
    """SyntheticLines satu kondisi.

    Seed sama untuk semua kondisi: font dan ukuran diambil sebelum sisipan, jadi ukurannya sama per indeks di setiap
    kondisi, dan langka & langka+fase5 bahkan teksnya sama persis (augmentasi mengambil angka acak sesudah sisipan).
    """
    spec = CONDITIONS[condition]
    rare_text = RareText(tuple(rare), tuple(common_letters), spec["insert_prob"], spec["opener_prob"])
    return SyntheticLines(lines, tokenizer, [font], augment=build_augment(spec["augment"]), deterministic=True,
                          seed=seed, rare_text=rare_text if rare_text.active else None)


class RenderedLines(Dataset):
    """Item i = citra dan teks SyntheticLines.sample(i), atau alasan baris dilewati seperti G1 (gagal render, T < 1,5L).

    Dibaca lewat DataLoader, jadi --workers merender sementara proses utama menjalankan model.
    """

    def __init__(self, synthetic: SyntheticLines):
        self.synthetic = synthetic

    def __len__(self) -> int:
        return len(self.synthetic)

    def __getitem__(self, idx: int) -> dict:
        started = time.perf_counter()
        item: dict = {"i": idx}
        try:
            image, text = self.synthetic.sample(idx)
        except (RenderClipped, EmptyRender) as error:
            item["dropped"] = f"gagal render: {error}"
        else:
            tokenizer = self.synthetic.tokenizer
            item["text"] = nfc(text)
            if is_trainable(len(tokenizer.encode(tokenizer.to_visual(text))), to_tensor(image).shape[-1]):
                item["image"] = image
            else:
                item["dropped"] = "T < 1,5L"
        item["ms"] = (time.perf_counter() - started) * 1000
        return item


def _unchanged(item):
    """collate_fn DataLoader tanpa batch: item apa adanya (fungsi modul supaya bisa di-pickle ke worker)."""
    return item


def warm_up(models: dict[str, tuple]) -> None:
    """Satu forward per model sebelum waktu diukur: forward pertama memuat kernel dan mengalokasikan memori."""
    blank = Image.new("L", WARMUP_SIZE, 255)
    for model, tokenizer in models.values():
        predict(blank, model, tokenizer, pad_ratio=0.0)


def read_condition(dataset: SyntheticLines, models: dict[str, tuple], workers: int, label: str) -> dict:
    """Render setiap indeks sekali dan baca citranya dengan setiap checkpoint (citra sama = berpasangan)."""
    loader = DataLoader(RenderedLines(dataset), batch_size=None, num_workers=workers, collate_fn=_unchanged)
    rows, dropped = [], []
    render_ms, infer_ms = 0.0, dict.fromkeys(models, 0.0)
    started, total = time.time(), len(dataset)
    for done, item in enumerate(loader, 1):
        render_ms += item["ms"]
        if "dropped" in item:
            dropped.append({"i": item["i"], "reason": item["dropped"], "text": item.get("text")})
        else:
            hyps = {}
            for name, (model, tokenizer) in models.items():
                tick = time.perf_counter()
                hyps[name] = nfc(predict(item["image"], model, tokenizer, pad_ratio=0.0))
                infer_ms[name] += (time.perf_counter() - tick) * 1000
            rows.append({"i": item["i"], "reference": item["text"], "hyp": hyps})
        if done % PROGRESS_EVERY == 0 and done < total:
            elapsed = time.time() - started
            remaining = elapsed / done * (total - done) / 60
            print(f"  [{label}] {done}/{total} baris, {elapsed:.0f} dtk, sisa ~{remaining:.1f} mnt", flush=True)
    return {"rows": rows, "dropped": dropped,
            "timing": {"rendered": total, "render_ms_per_line": render_ms / max(1, total),
                       "infer_ms_per_line": {name: ms / max(1, len(rows)) for name, ms in infer_ms.items()},
                       "wall_s": time.time() - started}}


# --- per baris -------------------------------------------------------------------------------------------


def plain_errors(reference: str, hypothesis: str, ops: Sequence[tuple], rare: frozenset[str]) -> int:
    """Edit pada karakter biasa: langkah penjajaran tak-sama di luar blok yang menyentuh aksara langka referensi.

    Blok = deretan langkah tak-sama yang bersebelahan. Blok yang memuat substitusi/hapus aksara langka referensi adalah
    bacaan glyph itu (satu glyph bisa terbaca beberapa karakter dan ikut menimpa spasi sisipan " X"), sudah diukur
    recall, jadi tidak dihitung. Di blok lain setiap langkah = satu edit, kecuali sisipan aksara langka (keluaran palsu,
    diukur presisi). Karakter biasa yang terbaca sebagai aksara langka tetap dihitung.
    """
    edits = block = 0
    touches = False
    for tag, i, j in [*ops, ("equal", None, None)]:  # langkah penutup mengakhiri blok terakhir
        if tag == "equal":
            edits += 0 if touches else block
            block, touches = 0, False
        elif i is not None and reference[i] in rare:
            touches = True
        elif not (tag == "insert" and hypothesis[j] in rare):
            block += 1
    return edits


def read_as(reference: str, hypothesis: str, ops: Sequence[tuple] | None = None) -> list[str]:
    """Untuk tiap karakter referensi: karakter keluaran yang dijajarkan dengannya, "" bila terhapus.

    Penjajaran = compare_runs.alignment (biaya minimum, lalu kecocokan terbanyak), yang juga dipakai matched_mask,
    jadi read_as(r, h)[i] == r[i] tepat bila matched_mask(r, h)[i]. `ops` = hasil alignment yang sudah dihitung
    (pure Python, ~2 ms per baris: cukup sekali per baris).
    """
    seen = [""] * len(reference)
    for tag, i, j in alignment(reference, hypothesis) if ops is None else ops:
        if tag in ("equal", "replace"):
            seen[i] = hypothesis[j]
    return seen


def homoglyph_label(letter: str, digit: str) -> str:
    """Label singkat untuk layar, mis. 'windu/0', 'pa murda/8'."""
    name = unicodedata.name(letter, "?").removeprefix("JAVANESE ").removeprefix("LETTER ").removeprefix("PADA ")
    return f"{name.lower()}/{unicodedata.digit(digit, '?')}"


def opener_position(text: str) -> int | None:
    """Indeks adeg-adeg pembuka baris (spasi di depan diabaikan, seperti compare_runs); None bila tidak ada."""
    body = text.lstrip(" ")
    return len(text) - len(body) if body.startswith(ADEG_ADEG) else None


def line_stats(reference: str, hypothesis: str, rare: frozenset[str]) -> dict:
    """Angka satu baris (string NFC) untuk satu checkpoint.

    Counter `ref`/`hits`/`emitted`/`misread` mencakup codepoint langka dan pasangan homoglif, tanpa adeg-adeg pembuka
    (indeks 0 referensi dan bacaannya di keluaran): pembuka dihitung lewat `opener*`. Topeng "terbaca benar" =
    compare_runs.matched_mask, dari satu penjajaran yang juga memberi bacaan salah (read_as).
    """
    ops = alignment(reference, hypothesis)
    edits = Levenshtein.distance(reference, hypothesis)
    if sum(tag != "equal" for tag, _, _ in ops) != edits:  # penjajaran wajib berbiaya minimum
        raise EvalError(f"penjajaran {reference!r} vs {hypothesis!r} tidak berbiaya minimum ({edits} edit)")
    mask = mask_from(ops, len(reference))
    seen = read_as(reference, hypothesis, ops)
    opener = reference.startswith(ADEG_ADEG)
    hyp_opener = opener_position(hypothesis)
    # Keluaran yang tidak ikut `emitted` ditentukan penjajaran, bukan posisi saja: pada baris berpembuka, karakter yang
    # dijajarkan "equal" dengan pembuka referensi (walau ada keluaran palsu di depannya); pada baris tanpa pembuka,
    # adeg-adeg di awal keluaran yang tidak dijajarkan dengan adeg-adeg referensi. Dengan posisi saja, bacaan pembuka
    # yang benar ("ꦠ꧋…" untuk "꧋…") terhitung keluaran palsu, dan adeg-adeg tengah baris yang terbaca di awal
    # keluaran terhitung benar tanpa terhitung keluaran (presisi > 100%).
    matched = {j: i for tag, i, j in ops if tag == "equal"}
    if opener:
        skip = next((j for j, i in matched.items() if i == 0), None)
    else:
        skip = hyp_opener if hyp_opener not in matched else None
    tracked = rare | HOMOGLYPH_CHARS
    positions = [i for i, ch in enumerate(reference) if ch in tracked and not (opener and i == 0)]
    # Baris berpembuka tidak ikut CER bukan-langka: pembuka yang terbaca taling (pre-base) pindah ke belakang aksara
    # pertama pada urutan logis bila suku katanya dikenal tokenizer, jadi tampil sebagai sisipan terpisah dari bloknya,
    # yang hanya merugikan model yang tidak bisa membaca pembuka.
    return {
        "chars": len(reference),
        "edits": edits,
        "plain_chars": 0 if opener else sum(ch not in rare for ch in reference),
        "plain_edits": 0 if opener else plain_errors(reference, hypothesis, ops, rare),
        "ref": Counter(reference[i] for i in positions),
        "hits": Counter(reference[i] for i in positions if mask[i]),
        "emitted": Counter(ch for j, ch in enumerate(hypothesis) if ch in tracked and j != skip),
        "misread": Counter((reference[i], seen[i]) for i in positions if not mask[i]),
        "opener": opener,
        "opener_read": opener and mask[0],
        "opener_seen": seen[0] if opener else None,
        "false_opener": not opener and hyp_opener is not None,
    }


def per_line(stats: Sequence[dict], field: str, members: frozenset[str] | None = None) -> list[int]:
    """Satu angka per baris: field skalar, atau jumlah Counter `field` untuk karakter di `members`."""
    if members is None:
        return [int(s[field]) for s in stats]
    return [sum(n for ch, n in s[field].items() if ch in members) for s in stats]


def rare_groups(rare: Sequence[str]) -> list[tuple[str, str, frozenset[str]]]:
    """(kunci, label, anggota): semua aksara langka, dan aksara murda seperti compare_runs; kelompok kosong dibuang."""
    rare_set = frozenset(rare)
    murda = frozenset(ch for ch in rare_set if "MURDA" in unicodedata.name(ch, ""))
    return [group for group in (("langka", "semua aksara langka", rare_set), ("murda", "aksara murda", murda))
            if group[2]]


# --- ringkasan -------------------------------------------------------------------------------------------


def misreads_of(misread: Counter, ch: str) -> list[dict]:
    """Bacaan salah tersering untuk karakter referensi `ch`: [{"as": karakter keluaran ("" = hilang), "n"}]."""
    found = Counter({seen: n for (ref, seen), n in misread.items() if ref == ch})
    return [{"as": seen, "n": n} for seen, n in found.most_common(TOP_MISREADS)]


def summarize_run(pairs: Sequence[tuple[str, str]], stats: Sequence[dict], rare: Sequence[str]) -> dict:
    """Ringkasan satu checkpoint pada satu kondisi. `pairs` = (referensi, keluaran) NFC, `stats` = line_stats-nya."""
    official = summarize(pairs)
    edits, chars = sum(per_line(stats, "edits")), sum(per_line(stats, "chars"))
    if abs(edits / max(1, chars) - official["cer"]) > TOLERANCE:
        raise EvalError(f"jumlah edit per baris ({edits / max(1, chars):.6%}) tidak mereproduksi "
                        f"src.evaluate.summarize ({official['cer']:.6%})")
    ref, hits, emitted, misread = (sum((s[field] for s in stats), Counter())
                                   for field in ("ref", "hits", "emitted", "misread"))
    groups = {}
    for key, label, members in rare_groups(rare):
        n_ref, n_hits, n_emitted = (sum(counter[ch] for ch in members) for counter in (ref, hits, emitted))
        groups[key] = {"label": label, "codepoints": len(members), "ref": n_ref, "hits": n_hits, "emitted": n_emitted,
                       "recall": ratio(n_hits, n_ref), "precision": ratio(n_hits, n_emitted)}
    openers, opener_read = sum(per_line(stats, "opener")), sum(per_line(stats, "opener_read"))
    false_openers = sum(per_line(stats, "false_opener"))
    opener_seen = Counter(s["opener_seen"] for s in stats if s["opener"] and not s["opener_read"])
    return {
        "lines": len(pairs),
        "cer": official["cer"],
        "cer_bukan_langka": ratio(sum(per_line(stats, "plain_edits")), sum(per_line(stats, "plain_chars"))),
        "cer_bukan_langka_lines": len(stats) - openers,
        "exact_line_accuracy": official["exact_line_accuracy"],
        "groups": groups,
        "per_codepoint": [{"codepoint": cp_label(ch), "char": ch, "name": unicodedata.name(ch, "?"), "ref": ref[ch],
                           "hits": hits[ch], "recall": ratio(hits[ch], ref[ch]), "emitted": emitted[ch],
                           "precision": ratio(hits[ch], emitted[ch]), "misread": misreads_of(misread, ch)}
                          for ch in rare],
        "opener": {"codepoint": cp_label(ADEG_ADEG), "lines": openers, "read": opener_read,
                   "recall": ratio(opener_read, openers),
                   "misread": [{"as": seen, "n": n} for seen, n in opener_seen.most_common(TOP_MISREADS)],
                   "lines_without": len(stats) - openers, "false": false_openers,
                   "false_rate": ratio(false_openers, len(stats) - openers)},
        "homoglyphs": [{"label": homoglyph_label(x, y), "x": cp_label(x), "y": cp_label(y), "x_char": x, "y_char": y,
                        "ref_x": ref[x], "ref_y": ref[y], "hits_x": hits[x], "hits_y": hits[y],
                        "x_as_y": misread[(x, y)], "y_as_x": misread[(y, x)]} for x, y in HOMOGLYPHS],
        "worst": official["worst"][:10],
    }


def compare_pair(stats_a: Sequence[dict], stats_b: Sequence[dict], rare: Sequence[str], resamples: int,
                 seed: int) -> dict:
    """Selisih A - B pada baris dan citra yang sama: rasio-jumlah, bootstrap berpasangan per baris, uji tanda."""
    groups = {key: members for key, _, members in rare_groups(rare)}
    opener = per_line(stats_a, "opener")
    no_opener = [1 - flag for flag in opener]
    columns = {
        "cer": ("edits", "chars", None),
        "cer_bukan_langka": ("plain_edits", "plain_chars", None),
        "recall_langka": ("hits", "ref", "langka"),
        "presisi_langka": ("hits", "emitted", "langka"),
        "recall_murda": ("hits", "ref", "murda"),
    }
    stats: dict[str, tuple] = {}
    for key, (num, den, group) in columns.items():
        if group is not None and group not in groups:
            continue
        members = groups.get(group)
        stats[key] = (per_line(stats_a, num, members), per_line(stats_a, den, members),
                      per_line(stats_b, num, members), per_line(stats_b, den, members))
    stats["recall_pembuka"] = (per_line(stats_a, "opener_read"), opener, per_line(stats_b, "opener_read"), opener)
    stats["pembuka_palsu"] = (per_line(stats_a, "false_opener"), no_opener,
                              per_line(stats_b, "false_opener"), no_opener)

    boot = bootstrap_diffs(stats, resamples, seed)
    metrics = {}
    for key, (num_a, den_a, num_b, den_b) in stats.items():
        a, b = ratio(sum(num_a), sum(den_a)), ratio(sum(num_b), sum(den_b))
        metrics[key] = {"a": a, "b": b, "diff": None if a is None or b is None else a - b, "bootstrap": boot[key]}
    read_a, read_b = per_line(stats_a, "opener_read"), per_line(stats_b, "opener_read")
    only_a = sum(x and not y for x, y in zip(read_a, read_b))
    only_b = sum(y and not x for x, y in zip(read_a, read_b))
    return {
        "metrics": metrics,
        # Referensi sama untuk A dan B, jadi urutan jarak edit = urutan CER baris.
        "paired_lines": paired_counts(per_line(stats_a, "edits"), per_line(stats_b, "edits")),
        "paired_opener": {"lines": sum(opener), "both": sum(x and y for x, y in zip(read_a, read_b)),
                          "only_a": only_a, "only_b": only_b, "sign_test_p": sign_test(only_a, only_b)},
    }


def evaluate_condition(condition: str, read: dict, rare: Sequence[str], names: Sequence[str], resamples: int,
                       seed: int) -> dict:
    """Semua angka satu kondisi dari hasil read_condition."""
    rows = read["rows"]
    if not rows:
        raise EvalError(f"kondisi {condition}: tidak ada baris yang terbaca (semua ditolak atau gagal render)")
    rare_set = frozenset(rare)
    stats = {name: [line_stats(row["reference"], row["hyp"][name], rare_set) for row in rows] for name in names}
    base = stats[names[0]]  # referensi sama untuk semua checkpoint
    return {
        **CONDITIONS[condition],
        "lines": len(rows),
        "dropped": read["dropped"],
        "reference_chars": sum(per_line(base, "chars")),
        "rare_in_reference": sum(per_line(base, "ref", rare_set)),
        "lines_with_rare": sum(1 for s in base if any(ch in rare_set for ch in s["ref"])),
        "lines_with_opener": sum(per_line(base, "opener")),
        "timing": read["timing"],
        "results": {name: summarize_run([(row["reference"], row["hyp"][name]) for row in rows], stats[name], rare)
                    for name in names},
        "comparisons": [{"a": a, "b": b, **compare_pair(stats[a], stats[b], rare, resamples, seed)}
                        for a, b in combinations(names, 2)],
    }


def official_g1(eval_dir: Path, checkpoint: Path, font_name: str) -> dict | None:
    """Laporan G1 resmi src.evaluate (tanpa augmentasi, font sama) untuk checkpoint ini; yang barisnya terbanyak."""
    target = checkpoint.resolve()
    best = None
    for path in sorted(eval_dir.glob("*.json")):
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("goal") != "G1" or data.get("augment") != "none":
            continue
        stored = Path(str(data.get("checkpoint", "")))
        block = (data.get("results") or {}).get(font_name)
        if not block or (stored if stored.is_absolute() else ROOT / stored).resolve() != target:
            continue
        if best is None or block["lines"] > best["lines"]:
            best = {"file": relative(path), "lines": block["lines"], "cer": block["cer"]}
    return best


def estimate_full(conditions: dict, lines: int, n_checkpoints: int) -> dict | None:
    """Detik untuk `lines` baris x semua kondisi x `n_checkpoints`, berurutan (render + inferensi per checkpoint).

    Inferensi per baris = rata-rata checkpoint yang diukur (arsitekturnya sama). None bila ada kondisi yang tidak
    diukur.
    """
    if set(conditions) != set(CONDITIONS):
        return None
    per_condition = {}
    for name, block in conditions.items():
        timing = block["timing"]
        infer = sum(timing["infer_ms_per_line"].values()) / max(1, len(timing["infer_ms_per_line"]))
        per_condition[name] = {"render_s": lines * timing["render_ms_per_line"] / 1000,
                               "infer_s": lines * n_checkpoints * infer / 1000}
    return {"lines": lines, "conditions": len(conditions), "checkpoints": n_checkpoints,
            "render_s": sum(p["render_s"] for p in per_condition.values()),
            "infer_s": sum(p["infer_s"] for p in per_condition.values()),
            "seconds": sum(p["render_s"] + p["infer_s"] for p in per_condition.values()),
            "per_condition": per_condition}


# --- laporan ---------------------------------------------------------------------------------------------


def share(value: float | None, num: int, den: int) -> str:
    return f"{pct(value)} ({num}/{den})"


def group_cell(group: dict | None, field: str) -> str:
    if group is None:
        return "-"
    return share(group[field], group["hits"], group["ref" if field == "recall" else "emitted"])


def char_label(ch: str) -> str:
    return f"{cp_label(ch)} {ch}" if ch else "hilang"


def misread_text(entries: Sequence[dict], limit: int = TOP_MISREADS) -> str:
    """Mis. 'U+A9D0 ꧐ ×3, hilang ×1'."""
    return ", ".join(f"{char_label(entry['as'])} ×{entry['n']}" for entry in entries[:limit]) or "-"


def short_name(name: str) -> str:
    return name.removeprefix("JAVANESE ")


def condition_markdown(condition: str, block: dict, names: Sequence[str]) -> list[str]:
    reasons = Counter(entry["reason"].split(":")[0] for entry in block["dropped"])
    dropped = ", ".join(f"{reason} {n}" for reason, n in reasons.items()) or "tidak ada"
    results = block["results"]
    out = [
        "",
        f"## Kondisi `{condition}`",
        "",
        f"{block['purpose']}. Sisip p={block['insert_prob']}, pembuka p={block['opener_prob']}, augmentasi "
        f"`{block['augment']}`. {block['lines']} baris dibaca (dilewati: {dropped}), {block['reference_chars']} "
        f"karakter referensi, {block['rare_in_reference']} aksara langka di {block['lines_with_rare']} baris (tanpa "
        f"pembuka), {block['lines_with_opener']} baris berpembuka adeg-adeg (CER bukan-langka dari "
        f"{block['lines'] - block['lines_with_opener']} baris tanpa pembuka).",
        "",
        "| checkpoint | CER | CER bukan-langka | recall langka | presisi langka | recall murda | pembuka terbaca | "
        "pembuka palsu | baris persis |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        r, opener = results[name], results[name]["opener"]
        out.append(f"| `{name}` | {pct(r['cer'])} | {pct(r['cer_bukan_langka'])} | "
                   f"{group_cell(r['groups'].get('langka'), 'recall')} | "
                   f"{group_cell(r['groups'].get('langka'), 'precision')} | "
                   f"{group_cell(r['groups'].get('murda'), 'recall')} | "
                   f"{share(opener['recall'], opener['read'], opener['lines'])} | "
                   f"{share(opener['false_rate'], opener['false'], opener['lines_without'])} | "
                   f"{pct(r['exact_line_accuracy'])} |")
    if any(results[name]["opener"]["misread"] for name in names):
        out += ["", "Pembuka yang tidak terbaca, dibaca sebagai: " + "; ".join(
            f"`{name}` {misread_text(results[name]['opener']['misread'])}" for name in names) + "."]

    if block["comparisons"]:
        out += ["", "| A - B | metrik | A | B | A - B | SK 95% |", "|---|---|---:|---:|---:|---:|"]
        for comp in block["comparisons"]:
            for key, m in comp["metrics"].items():
                out.append(f"| `{comp['a']}` - `{comp['b']}` | {METRICS[key]} | {pct(m['a'])} | {pct(m['b'])} | "
                           f"{points(m['diff'])} | {interval(m['bootstrap'])} |")
        out.append("")
        for comp in block["comparisons"]:
            p, opener = comp["paired_lines"], comp["paired_opener"]
            out.append(f"- `{comp['a']}` vs `{comp['b']}`: per baris (jarak edit) A lebih baik {p['a_better']}, B "
                       f"lebih baik {p['b_better']}, sama {p['tie']} (uji tanda p = {p['sign_test_p']:.3g}); pembuka "
                       f"terbaca keduanya {opener['both']}, A saja {opener['only_a']}, B saja {opener['only_b']} dari "
                       f"{opener['lines']} baris (p = {opener['sign_test_p']:.3g}).")

    first = results[names[0]]["per_codepoint"]
    shown = [k for k, entry in enumerate(first)
             if entry["ref"] or any(results[name]["per_codepoint"][k]["emitted"] for name in names)]
    if shown:
        out += [
            "",
            "Per codepoint (baris yang muncul di referensi atau keluaran; n = jumlah di referensi; presisi dengan "
            "jumlah di keluaran; salah baca = dua bacaan salah tersering):",
            "",
            "| codepoint | nama | n | " + " | ".join(f"recall `{name}`" for name in names) + " | "
            + " | ".join(f"presisi `{name}`" for name in names) + " | "
            + " | ".join(f"salah baca `{name}`" for name in names) + " |",
            "|---|---|---:|" + "---:|" * (2 * len(names)) + "---|" * len(names),
        ]
        for k in shown:
            entries = [results[name]["per_codepoint"][k] for name in names]
            e = entries[0]
            out.append(f"| {e['codepoint']} {e['char']} | {short_name(e['name'])} | {e['ref']} | "
                       + " | ".join(pct(x["recall"]) for x in entries) + " | "
                       + " | ".join(f"{pct(x['precision'])} ({x['emitted']})" for x in entries) + " | "
                       + " | ".join(misread_text(x["misread"], 2) for x in entries) + " |")
    out += [
        "",
        "Homoglif (substitusi pada penjajaran; X->Y = X di referensi terbaca Y):",
        "",
        "| X / Y | di referensi X / Y | " + " | ".join(f"`{name}` X->Y / Y->X (X benar)" for name in names) + " |",
        "|---|---:|" + "---:|" * len(names),
    ]
    for k, (x, y) in enumerate(HOMOGLYPHS):
        entries = [results[name]["homoglyphs"][k] for name in names]
        e = entries[0]
        pair = f"{e['x']} {x} {short_name(unicodedata.name(x))} / {e['y']} {y} {short_name(unicodedata.name(y))}"
        out.append(f"| {pair} | {e['ref_x']} / {e['ref_y']} | "
                   + " | ".join(f"{h['x_as_y']} / {h['y_as_x']} ({h['hits_x']})" for h in entries) + " |")
    return out


def render_markdown(report: dict) -> str:
    settings, rare, timing = report["settings"], report["rare"], report["timing"]
    names = list(report["checkpoints"])
    boot = settings["bootstrap"]
    low, high = settings["size_range"]
    coverage = ("semua ada di cmap font" if not rare["missing_in_font"]
                else "TIDAK ada di cmap font (tidak pernah disisipkan): " + ", ".join(rare["missing_in_font"]))
    out = [
        "# Evaluasi sintetis aksara langka",
        "",
        "Citra yang SAMA untuk setiap checkpoint (berpasangan), tanpa NusaAksara: titik akhir utama fase6_rare vs "
        "fase6_ctrl.",
        "",
        f"- Baris: {settings['lines']} dari {settings['split_lines']} baris split test (`{settings['split']}`), sampel "
        f"acak seed {settings['seed']} (indeks diacak, N pertama). Bukan train/val, bukan NusaAksara.",
        f"- Render: `SyntheticLines(deterministic=True, seed={settings['seed']})`, satu font `{settings['font']}` "
        f"(held-out: tidak pernah merender data training), ukuran {low}-{high} px, spasi tidak dibuang; baris yang "
        "melanggar T >= 1,5L ditolak seperti G1. Seed sama untuk semua kondisi: per indeks, teks dasar dan ukuran "
        "sama.",
        f"- Aksara langka: {rare['count']} codepoint yang sama dengan training fase6_rare (`{rare['cache']}`); "
        f"{coverage}. Sisipan aksara menggantikan salah satu dari {len(rare['common_letters'])} aksara tersering "
        "split test (himpunannya sama dengan training).",
        f"- Inferensi: {settings['inference']}; {settings['threads']} thread torch.",
        "- Checkpoint (urutan = arah selisih A - B):",
        *[f"  - `{name}`: `{info['path']}` (langkah {info['step']}; sisip aksara langka "
          f"p={info['rare_insert_prob'] or 0}, pembuka p={info['rare_opener_prob'] or 0})"
          for name, info in report["checkpoints"].items()],
        f"- SK 95% = bootstrap berpasangan per baris ({boot['resamples']} resample, seed {boot['seed']}, persentil "
        "2,5-97,5). Setiap baris dirender sendiri, jadi baris = satuan resampling.",
        "",
        "Definisi (NFC, urutan logis):",
        "",
        "- CER: mikro, fungsi resmi `src.evaluate.summarize`.",
        "- CER bukan-langka: kesalahan di luar aksara langka / jumlah karakter bukan-langka referensi. Langkah "
        "penjajaran tak-sama yang bersebelahan membentuk blok; blok yang memuat substitusi/hapus aksara langka "
        "referensi adalah bacaan glyph itu (satu glyph bisa terbaca beberapa karakter dan menimpa spasi sisipan "
        "\" X\"), sudah diukur recall, jadi tidak dihitung; sisipan aksara langka (keluaran palsu) juga tidak, sudah "
        "diukur presisi. Karakter biasa yang terbaca sebagai aksara langka tetap dihitung. Hanya baris tanpa pembuka "
        "adeg-adeg: pembuka yang terbaca taling (pre-base) bisa pindah ke belakang aksara pertama pada urutan logis, "
        "sehingga tampil sebagai sisipan terpisah yang hanya merugikan model yang tidak bisa membaca pembuka. Efek "
        "samping pada teks umum yang paling bersih tetap CER kondisi `biasa`.",
        "- Recall/presisi langka: benar = \"equal\" pada penjajaran `compare_runs.alignment` (biaya Levenshtein "
        "minimum, lalu kecocokan terbanyak; sama dengan `matched_mask`); recall = benar / jumlah di referensi, presisi "
        "= benar / jumlah di keluaran (\"-\" = penyebut 0). Adeg-adeg di awal baris tidak termasuk: dilaporkan sendiri "
        "sebagai pembuka. Homoglif dan salah baca memakai penjajaran yang sama.",
        "- Pembuka terbaca: baris referensi berawalan adeg-adeg yang karakter pertamanya terbaca benar. Pembuka palsu: "
        "keluaran berawalan adeg-adeg (spasi di depan diabaikan) pada baris yang referensinya tanpa pembuka.",
        "- Selisih = A - B. CER, CER bukan-langka, pembuka palsu: negatif = A lebih baik. Recall, presisi, pembuka "
        "terbaca: positif = A lebih baik.",
    ]
    for condition, block in report["conditions"].items():
        out += condition_markdown(condition, block, names)

    out += [
        "",
        "## Biaya",
        "",
        f"Diukur pada run ini ({settings['lines']} baris, {settings['threads']} thread torch, --workers "
        f"{settings['workers']}); proses lain (training) mungkin sedang berjalan, jadi anggap kasar.",
        "",
        "| kondisi | render ms/baris | " + " | ".join(f"inferensi `{name}` ms/baris" for name in names)
        + " | waktu dinding |",
        "|---|---:|" + "---:|" * len(names) + "---:|",
    ]
    for condition, block in report["conditions"].items():
        t = block["timing"]
        out.append(f"| `{condition}` | {t['render_ms_per_line']:.0f} | "
                   + " | ".join(f"{t['infer_ms_per_line'][name]:.0f}" for name in names)
                   + f" | {t['wall_s']:.0f} dtk |")
    estimate = timing["estimate_full"]
    if estimate:
        out += ["", f"Perkiraan jalan penuh {estimate['lines']} baris x {estimate['conditions']} kondisi x "
                    f"{estimate['checkpoints']} checkpoint, berurutan (--workers 0), {settings['threads']} thread: "
                    f"~{estimate['seconds'] / 60:.0f} menit (render {estimate['render_s'] / 60:.0f}, inferensi "
                    f"{estimate['infer_s'] / 60:.0f}). Dengan --workers render sebagian besar tumpang tindih dengan "
                    "inferensi."]

    out += ["", "## Cek kewajaran: kondisi biasa vs G1 resmi", ""]
    biasa = report["conditions"].get("biasa")
    if biasa is None:
        out.append("Kondisi `biasa` tidak dijalankan.")
    else:
        for name, official in report["sanity_g1"].items():
            mine = f"`{name}`: {pct(biasa['results'][name]['cer'])} pada {biasa['lines']} baris"
            out.append(f"- {mine}; G1 resmi `{official['file']}` {pct(official['cer'])} pada {official['lines']} baris."
                       if official else f"- {mine}; laporan G1 resmi untuk checkpoint ini tidak ada.")
        out += ["", "G1 resmi = N baris PERTAMA split test, seed render 12345, jalur batch `src.evaluate`; di sini "
                    "sampel acak, seed lain, inferensi satu baris. Selisih sebesar derau sampel wajar; selisih jauh di "
                    "atasnya berarti jalurnya perlu diperiksa."]
    inserted = max((block["rare_in_reference"] for block in report["conditions"].values()), default=0)
    per_codepoint = inserted / max(1, rare["count"])
    out += [
        "",
        "## Batasan",
        "",
        "- Satu font held-out dan teks sintetis: angka ini mengukur pengenalan bentuk glyph, bukan cetakan nyata. G3 "
        "tetap angka laporan untuk data nyata.",
        "- Sisipan tidak linguistis (aksara langka di posisi acak), sama dengan cara training fase6_rare "
        "menyisipkannya; fase6_ctrl dan fase5_fonts tidak pernah melihat distribusi sisipan itu. Itu memang yang "
        "diukur (efek intervensi), tapi recall di sini bukan recall pada teks Jawa sungguhan.",
        f"- Per codepoint rata-rata hanya {per_codepoint:.1f} kemunculan per kondisi ({inserted} aksara langka untuk "
        f"{rare['count']} codepoint); SK hanya dihitung untuk kelompok.",
        "- Satu run per checkpoint: SK hanya memuat keragaman baris, bukan keragaman antar-run training (seed, urutan "
        "batch).",
        "- Penjajaran bisa seri walau biaya minimum dan kecocokan terbanyak; pilihannya deterministik, tapi jumlah per "
        "codepoint yang kecil bisa bergeser satu-dua karakter dengan aturan pemutus lain.",
    ]
    return "\n".join(out) + "\n"


def summary_lines(report: dict) -> list[str]:
    settings, timing = report["settings"], report["timing"]
    names = list(report["checkpoints"])
    width = max(len(name) for name in names)
    out = [f"{settings['lines']} baris test (seed {settings['seed']}), font {Path(settings['font']).name}, "
           f"{settings['threads']} thread; checkpoint {', '.join(names)}; selisih = A - B"]
    for condition, block in report["conditions"].items():
        out.append(f"== {condition}: {block['lines']} baris (dilewati {len(block['dropped'])}), "
                   f"{block['rare_in_reference']} aksara langka di {block['lines_with_rare']} baris, "
                   f"{block['lines_with_opener']} berpembuka")
        for name in names:
            r, opener = block["results"][name], block["results"][name]["opener"]
            out.append(f"  {name:{width}}  CER {pct(r['cer']):>7}  bukan-langka {pct(r['cer_bukan_langka']):>7}  "
                       f"recall langka {group_cell(r['groups'].get('langka'), 'recall'):>17}  "
                       f"presisi {group_cell(r['groups'].get('langka'), 'precision'):>17}  "
                       f"murda {group_cell(r['groups'].get('murda'), 'recall'):>15}  "
                       f"pembuka {share(opener['recall'], opener['read'], opener['lines']):>15}  "
                       f"palsu {share(opener['false_rate'], opener['false'], opener['lines_without'])}")
        for comp in block["comparisons"]:
            m = comp["metrics"]
            out.append(f"  {comp['a']} - {comp['b']}: " + "; ".join(
                f"{METRICS[key]} {points(m[key]['diff'])} SK95 {interval(m[key]['bootstrap'])}"
                for key in ("cer", "cer_bukan_langka", "recall_langka", "recall_pembuka", "pembuka_palsu") if key in m)
                + f"; baris A/B/sama {comp['paired_lines']['a_better']}/{comp['paired_lines']['b_better']}/"
                f"{comp['paired_lines']['tie']}")
        out.append("  homoglif X->Y/Y->X: " + "; ".join(
            f"{name} " + ", ".join(f"{h['label']} {h['x_as_y']}/{h['y_as_x']}"
                                   for h in block["results"][name]["homoglyphs"])
            for name in names))
    out.append("waktu per baris: " + "; ".join(
        f"{condition} render {block['timing']['render_ms_per_line']:.0f} ms, inferensi "
        + ", ".join(f"{name} {block['timing']['infer_ms_per_line'][name]:.0f} ms" for name in names)
        for condition, block in report["conditions"].items()))
    estimate = timing["estimate_full"]
    if estimate:
        out.append(f"perkiraan jalan penuh {estimate['lines']} baris x {estimate['conditions']} kondisi x "
                   f"{estimate['checkpoints']} checkpoint (berurutan, {settings['threads']} thread): "
                   f"~{estimate['seconds'] / 60:.0f} menit (render {estimate['render_s'] / 60:.0f}, inferensi "
                   f"{estimate['infer_s'] / 60:.0f})")
    biasa = report["conditions"].get("biasa")
    if biasa is not None:
        for name, official in report["sanity_g1"].items():
            out.append(f"cek kewajaran {name}: biasa {pct(biasa['results'][name]['cer'])} ({biasa['lines']} baris) vs "
                       + (f"G1 resmi {pct(official['cer'])} ({official['lines']} baris, {official['file']})"
                          if official else "laporan G1 resmi tidak ada"))
    return out


# --- skrip -----------------------------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lines", type=int, default=DEFAULT_LINES, help="baris split test; 0 = semua")
    parser.add_argument("--seed", type=int, default=0, help="seed sampel baris, render, dan bootstrap")
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS), metavar="NAMA[=PATH]",
                        help="NAMA=PATH, atau NAMA = out/checkpoints/NAMA/last_snapshot.pt; urutan = arah selisih "
                             f"(default {' '.join(DEFAULT_CHECKPOINTS)})")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--workers", type=int, default=0, help="proses render paralel (DataLoader); 0 = proses utama")
    parser.add_argument("--threads", type=int, default=0, help="thread torch untuk inferensi; 0 = bawaan torch")
    parser.add_argument("--bootstrap", type=int, default=10_000, help="jumlah resample; 0 = tanpa selang kepercayaan")
    parser.add_argument("--font", default=str(WINDOWS_JAVANESE_TEXT),
                        help="font render (default javatext, held-out); ganti hanya untuk uji")
    parser.add_argument("--out", default=str(OUT_DIR), help="direktori keluaran (default out/compare)")
    args = parser.parse_args(argv)
    for option in ("lines", "workers", "threads", "bootstrap"):
        if getattr(args, option) < 0:
            parser.error(f"--{option} tidak boleh negatif")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.threads:
        torch.set_num_threads(args.threads)
    started = time.time()
    font, out_dir = Path(args.font), Path(args.out)
    conditions = [condition for condition in CONDITIONS if condition in args.conditions]

    try:
        if not font.is_file():
            raise EvalError(f"font tidak ada: {font}")
        checkpoints = parse_checkpoints(args.checkpoints)
        tokenizer = Tokenizer.load(TOKENIZER_PATH)
        # Cache kanonis compare_runs (bukan --out): himpunan yang sama dengan training fase6_rare.
        rare_data = load_rare(OUT_DIR / RARE_CACHE)
        rare = [entry["char"] for entry in rare_data["codepoints"]]
        covered = font_codepoints(str(font))
        missing = [cp_label(ch) for ch in rare if ord(ch) not in covered]
        test_lines = read_split("test")
        order = sample_indices(len(test_lines), args.lines, args.seed)
        lines = [test_lines[i] for i in order]
        # Hanya keanggotaan yang dipakai insert_codepoint; 20 aksara tersering split test = himpunan training.
        common = RareText.from_lines(test_lines, tokenizer.charset).common_letters
        info = {name: checkpoint_info(path) for name, path in checkpoints.items()}
        models = load_models(checkpoints, tokenizer)
    except (EvalError, CompareError) as error:
        raise SystemExit(f"galat: {error}") from None

    print(f"eval_rare: {len(lines)} baris test (seed {args.seed}), font {font.name}, kondisi {', '.join(conditions)}, "
          f"checkpoint {', '.join(checkpoints)}; {torch.get_num_threads()} thread torch, workers {args.workers}",
          flush=True)
    if missing:
        print(f"PERINGATAN: font tidak punya glyph {', '.join(missing)}; codepoint itu tidak pernah disisipkan")
    warm_up(models)
    results, records = {}, []
    try:
        for condition in conditions:
            dataset = condition_dataset(lines, tokenizer, font, rare, common, condition, args.seed)
            read = read_condition(dataset, models, args.workers, condition)
            for entry in read["dropped"]:
                entry["split_index"] = order[entry["i"]]
            results[condition] = evaluate_condition(condition, read, rare, list(checkpoints), args.bootstrap, args.seed)
            records += [{"condition": condition, "i": row["i"], "split_index": order[row["i"]],
                         "reference": row["reference"], "hyp": row["hyp"]} for row in read["rows"]]
            records += [{"condition": condition, **entry} for entry in read["dropped"]]
            print(f"  [{condition}] selesai: {len(read['rows'])} baris, {read['timing']['wall_s']:.0f} dtk", flush=True)
    except EvalError as error:
        raise SystemExit(f"galat: {error}") from None

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "split": relative(ROOT / "data" / "splits" / "test.txt"), "split_lines": len(test_lines),
            "lines": len(lines), "seed": args.seed,
            "sampling": "indeks split test diacak random.Random(seed).shuffle, N pertama",
            "font": relative(font), "size_range": list(RENDER_SIZE_RANGE),
            "inference": "src.infer.predict satu baris (greedy, to_tensor, urutan logis, NFC, pad_ratio 0), CPU",
            "threads": torch.get_num_threads(), "workers": args.workers,
            "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "interval": "persentil 2,5-97,5",
                          "unit": "baris"},
        },
        "rare": {"cache": relative(OUT_DIR / RARE_CACHE), "params": rare_data.get("params"), "count": len(rare),
                 "codepoints": [cp_label(ch) for ch in rare],
                 "murda": [cp_label(ch) for ch in rare if "MURDA" in unicodedata.name(ch, "")],
                 "missing_in_font": missing, "common_letters": [cp_label(ch) for ch in common],
                 "common_letters_source": "RareText.from_lines(split test).common_letters"},
        "checkpoints": info,
        "conditions": results,
        "timing": {"estimate_full": estimate_full(results, DEFAULT_LINES, len(DEFAULT_CHECKPOINTS)),
                   "total_wall_s": time.time() - started},
        "sanity_g1": {name: official_g1(EVAL_DIR, path, font.name) for name, path in checkpoints.items()},
    }
    write_json(out_dir / f"{STEM}.json", report)
    (out_dir / f"{STEM}.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")
    with (out_dir / f"{STEM}_lines.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)

    print("\n".join(summary_lines(report)))
    print(f"hasil: {relative(out_dir / STEM)}.json, .md, _lines.jsonl ({time.time() - started:.0f} dtk)")


if __name__ == "__main__":
    main()
