"""Evaluasi sintetis dosis-respons jarak antar suku kata: citra yang SAMA untuk setiap checkpoint, tanpa NusaAksara.

  .venv/Scripts/python scripts/eval_spacing.py                    # 200 baris x 2 bentuk x 6 jarak x checkpoint yang ada
  .venv/Scripts/python scripts/eval_spacing.py --lines 8 --tracking 0 0.3 --checkpoints fase6_ctrl fase5_fonts --out DIR
  .venv/Scripts/python scripts/eval_spacing.py --checkpoints A=path/a.pt B [--fonts F ...] [--tracking EM ...]
      [--lines 200] [--seed 0] [--threads 0] [--bootstrap 10000] [--fresh] [--out DIR]

Titik akhir utama eksperimen tracking fase 7 (aturan analisis fase 7 di CLAUDE.md), dibuat sebelum hasil run fase7
dibaca. Cetakan Jawa tanpa spasi diratakan dengan merenggangkan aksara, dan model yang hanya melihat jarak bawaan font
membaca renggang itu sebagai spasi kata (fase6_ctrl pada 745 baris nyata: 2.168 spasi keluaran, 110 di referensi). Di
sini gejala itu diukur sebagai fungsi jarak pada citra sintetis bersih; test set tidak disentuh.

  Baris      sampel acak split val: indeks diacak dengan --seed, ambil N pertama (sampel kecil = awal sampel besar).
             Bukan train, bukan NusaAksara.
  Bentuk     tanpa-spasi  spasi dibuang dari teks sebelum render, seperti cetakan ("C + pangkon + C" antar kata
                          menjadi pasangan; sama dengan --drop-space-prob saat training).
             berspasi     teks apa adanya.
  Render     src.render.render_line(teks, font, 64, tracking=t): bersih, deterministik, tanpa augmentasi. t (em) =
             jarak tambahan sesudah tiap suku kata ortografis termasuk spasi, jadi celah antar-aksara bertambah t dan
             celah kata bertambah 2t (sama dengan augmentasi training fase7). Bawaan 0 0.1 0.2 0.3 0.45 0.6: 0-0,3 em
             = rentang latih fase7_track (--track-max 0.3), di atasnya ekstrapolasi.
  Font       bawaan javatext.ttf: tidak merender data training, tetapi sekeluarga dengan font training CarakanJawa,
             jadi bukan font yang benar-benar baru. --fonts mengganti daftarnya (sebut javatext lagi untuk menambah).
  Ditolak    baris yang salah satu bentuknya tidak round-trip tokenizer, memuat karakter tanpa glyph di font, gagal
             dirender, atau melanggar T >= 1,5L (seperti G1) pada salah satu jarak dibuang dari SEMUA kondisi font
             itu: setiap jarak memakai baris yang sama.
  Inferensi  src.infer.predict satu baris: greedy, to_tensor, urutan logis, NFC, pad_ratio 0 (jalur G3 resmi).
  Checkpoint NAMA = out/checkpoints/NAMA/last_snapshot.pt, atau NAMA=PATH. Yang berkasnya tidak ada dilewati dengan
             pesan (run fase7 selesai satu per satu); galat hanya bila tidak ada satu pun.

Metrik per checkpoint x font x bentuk x jarak (NFC, urutan logis):
  Spasi palsu   per 100 batas suku kata: spasi keluaran yang tidak berpasangan dengan spasi referensi, dibagi jumlah
                batas antar suku kata ortografis referensi yang bukan spasi (src.tokenizer.logical_syllables). Semua
                spasi palsu dihitung di mana pun letaknya, jadi angkanya bisa melebihi 100.
                Berpasangan = berada di celah yang sama: karakter bukan-spasi referensi dan keluaran dijajarkan dengan
                compare_runs.alignment (biaya Levenshtein minimum, lalu kecocokan terbanyak, lalu substitusi sekelas),
                dan spasi keluaran dipasangkan dengan spasi referensi hanya bila celah keduanya bertemu pada
                penjajaran itu. Penjajaran string utuh tidak dipakai untuk spasi: spasi yang bergeser satu aksara seri
                dengan aksara itu, dan pemutus serinya menganggap spasi geser-kanan terbaca (lihat line_stats).
  Recall spasi  bentuk berspasi saja: spasi referensi yang berpasangan / jumlah spasi referensi.
  CER           mikro dari src.evaluate.summarize: literal, dan dengan spasi dibuang dari keluaran (cer_hyp_no_space),
                sebagai kontrol bahwa bentuk aksara tetap terbaca. Pada bentuk berspasi spasi referensi yang dibuang
                ikut terhitung salah, jadi di sana kontrolnya CER tak peka spasi (spasi dibuang dari kedua sisi); pada
                bentuk tanpa-spasi keduanya sama.
Selisih A - B untuk setiap pasangan checkpoint (A = yang lebih dulu di --checkpoints; pasangan pertama = dua yang
pertama ada) pada tiap jarak: bootstrap berpasangan per baris (compare_runs.bootstrap_diffs, persentil 2,5-97,5), satu
set resample untuk semua jarak.

Keluaran: <out>/spacing_synthetic.json, .md, spacing_synthetic_lines.jsonl (teks per citra), dan ringkasan di layar
termasuk ms per citra dan perkiraan waktu jalan penuh. Setiap bacaan langsung ditulis ke
<out>/spacing_synthetic_cache.jsonl (kunci = isi font, isi checkpoint, teks, jarak; hanya dipakai bila versi pustaka
dan kode src/ dari render sampai decode masih sama), jadi jalan panjang yang terputus cukup dijalankan ulang: bacaan
yang sudah ada tidak dihitung lagi, juga saat --lines diperbesar atau checkpoint baru menyusul. --fresh mengabaikan
cache. Ringkasan tiap kondisi yang selesai ditulis ke <out>/spacing_synthetic_partial.jsonl dan dihapus saat laporan
akhir tertulis.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import PIL
import torch
from PIL import Image, features
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))  # compare_runs, eval_rare; scripts/ bukan paket

from compare_runs import (  # noqa: E402
    OUT_DIR,
    TOLERANCE,
    alignment,
    bootstrap_diffs,
    cp_label,
    interval,
    pct,
    points,
    ratio,
    relative,
    write_json,
)
from eval_rare import EvalError, sample_indices  # noqa: E402
from src.dataset import MIN_SPACE_RATIO, is_trainable, space_ratio, to_tensor  # noqa: E402
from src.evaluate import summarize  # noqa: E402
from src.infer import load_checkpoint, predict  # noqa: E402
from src.render import DEFAULT_PAD, DEFAULT_SIZE, EmptyRender, RenderClipped, render_line  # noqa: E402
from src.text_augment import font_codepoints  # noqa: E402
from src.tokenizer import TOKENIZER_PATH, WINDOWS_JAVANESE_TEXT, Tokenizer, logical_syllables, nfc  # noqa: E402
from src.train import read_split  # noqa: E402

DEFAULT_LINES = 200
PLANNED_MIN_LINES = 300  # aturan analisis fase 7 (CLAUDE.md): ">= 300 baris, berspasi & tanpa spasi, 0-0,6 em"
DEFAULT_TRACKING = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6)  # em
TRAIN_TRACK_MAX = 0.3  # --track-max rantai fase7 (scripts/run_fase7.sh): di atasnya = ekstrapolasi
RENDER_SIZE = DEFAULT_SIZE  # 64 px, tetap: satu-satunya yang berubah antar-kondisi adalah bentuk dan jarak
CHECKPOINT_DIR = ROOT / "out" / "checkpoints"
SNAPSHOT = "last_snapshot.pt"  # salinan last.pt di akhir run (scripts/fase6_common.sh::evaluate)
# Urutan = arah selisih: pasangan (A, B) mengikuti urutan ini, jadi fase7_track - fase7_ctrl (efek jarak saja,
# jumlah langkah sama) lebih dulu.
DEFAULT_CHECKPOINTS = ("fase7_track", "fase7_ctrl", "fase7_track_rare", "fase6_ctrl", "fase5_fonts")
STEM = "spacing_synthetic"
SPACE = " "
NO_SPACE, SPACED = "tanpa-spasi", "berspasi"
FORMS = {
    NO_SPACE: "spasi dibuang dari teks sebelum render, seperti cetakan (\"C + pangkon + C\" antar kata menjadi "
              "pasangan)",
    SPACED: "teks apa adanya",
}
# kunci -> (pembilang, penyebut, label) dari angka line_stats. Spasi palsu & CER: lebih kecil = lebih baik.
PAIR_METRICS = {
    "false_per_boundary": ("false_spaces", "boundaries", "spasi palsu per 100 batas"),
    "space_recall": ("matched_spaces", "ref_spaces", "recall spasi"),
    "cer_space_insensitive": ("glyph_edits", "glyph_chars", "CER tak peka spasi"),
}
PAIR_ROWS = (  # (metrik, bentuk) dalam urutan tabel selisih; recall spasi tidak ada pada bentuk tanpa-spasi
    ("false_per_boundary", NO_SPACE), ("false_per_boundary", SPACED), ("space_recall", SPACED),
    ("cer_space_insensitive", NO_SPACE), ("cer_space_insensitive", SPACED),
)
# Dasar terukur sebelum skrip ini ada (CLAUDE.md, 2026-10-03): model tanpa jarak, 8 baris javatext tanpa spasi.
BASELINE_NOTE = "0 / 54-64 / 91-93 spasi palsu per 100 batas pada 0 / 0,15 / 0,30 em"
CACHE_VERSION = 1  # naikkan bila isi atau makna catatan cache berubah: bacaan lama tidak dipakai lagi
# Berkas src/ yang menentukan jalur citra -> teks (render, to_tensor, model, decode, urutan logis); isinya ikut
# menjadi bagian kunci cache lewat environment().
CODE_FILES = ("render.py", "dataset.py", "model.py", "infer.py", "decode.py", "tokenizer.py")
PROGRESS_SECONDS = 30
WARMUP_SIZE = (384, 96)  # citra polos untuk satu forward pemanasan per model sebelum waktu diukur


# --- masukan ---------------------------------------------------------------------------------------------


def file_sha(path: Path) -> str:
    """12 heksadesimal pertama SHA-1 isi berkas: identitas font dan checkpoint di cache bacaan."""
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha1").hexdigest()[:12]


def resolve_checkpoints(specs: Sequence[str]) -> tuple[dict[str, Path], list[dict]]:
    """'NAMA=PATH' atau 'NAMA' (= out/checkpoints/<NAMA>/last_snapshot.pt) -> ({nama: path} yang ada, yang dilewati).

    Urutan dipertahankan (= arah selisih). Berkas yang tidak ada dilewati, bukan galat: run fase7 selesai satu per
    satu dan skrip dijalankan dengan daftar yang sama. Galat hanya bila tidak ada satu pun yang ada.
    """
    found: dict[str, Path] = {}
    skipped: list[dict] = []
    seen: set[str] = set()
    for spec in specs:
        name, sep, path = (part.strip() for part in spec.partition("="))
        if not name or (sep and not path):
            raise EvalError(f"--checkpoints: {spec!r} bukan NAMA atau NAMA=PATH")
        if name in seen:
            raise EvalError(f"--checkpoints: nama '{name}' disebut dua kali")
        seen.add(name)
        target = Path(path) if sep else CHECKPOINT_DIR / name / SNAPSHOT
        if target.is_file():
            found[name] = target
        else:
            skipped.append({"name": name, "path": relative(target)})
    if not found:
        missing = ", ".join(f"{entry['name']} ({entry['path']})" for entry in skipped)
        raise EvalError(f"tidak ada satu pun checkpoint yang berkasnya ada: {missing}; training belum selesai? Pilih "
                        "dengan --checkpoints")
    return found, skipped


def checkpoint_info(path: Path) -> dict:
    """Langkah dan argumen training yang membedakan run, untuk laporan; `sha1` = identitas di cache bacaan."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt.get("args") or {}
    stat = path.stat()
    return {"path": relative(path), "sha1": file_sha(path), "step": ckpt.get("step"), "bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            **{key: args.get(key) for key in ("init", "augment", "extra_fonts", "drop_space_prob", "track_prob",
                                               "track_max", "rare_insert_prob", "rare_opener_prob")}}


def load_models(checkpoints: dict[str, Path], tokenizer: Tokenizer) -> dict[str, tuple]:
    """(model CPU, tokenizer) per checkpoint.

    Charset dan tabel reorder-nya harus sama dengan tokenizer yang memeriksa label: baris dipilih menurut round-trip
    tokenizer itu, sedangkan keluaran dikembalikan ke urutan logis oleh tokenizer checkpoint.
    """
    models = {}
    for name, path in checkpoints.items():
        model, ckpt_tokenizer = load_checkpoint(path)
        if list(ckpt_tokenizer.charset) != list(tokenizer.charset) or ckpt_tokenizer.reorder != tokenizer.reorder:
            raise EvalError(f"charset atau tabel reorder checkpoint '{name}' berbeda dari {relative(TOKENIZER_PATH)}")
        models[name] = (model, ckpt_tokenizer)
    return models


def environment() -> dict:
    """Penentu citra dan keluaran selain font, teks, dan checkpoint; bacaan cache hanya dipakai bila semuanya sama.

    Versi pustaka (.venv dan .venv-xpu berbeda Pillow/HarfBuzz/torch, dan Pillow lain bisa merender berbeda:
    CLAUDE.md) dan isi berkas src/ dari render sampai decode, supaya perubahan kode tidak diam-diam memakai bacaan lama.
    """
    code = hashlib.sha1()
    for name in CODE_FILES:
        code.update((ROOT / "src" / name).read_bytes())
    return {"pillow": PIL.__version__, "raqm": features.version("raqm"), "harfbuzz": features.version("harfbuzz"),
            "torch": torch.__version__, "code": code.hexdigest()[:12]}


def font_note(font: Path) -> str:
    """Hubungan font dengan data training, untuk laporan."""
    if font.name.lower() == WINDOWS_JAVANESE_TEXT.name.lower():
        note = ("tidak merender data training, tetapi sekeluarga dengan font training CarakanJawa (advance sama pada "
                "72 dari 73 aksara/angka/pada, IoU glyph median 0,86; CLAUDE.md): bukan font yang benar-benar baru")
    elif font.resolve().parent == ROOT / "fonts":
        note = "font training (font inti)"
    elif font.resolve().parent == ROOT / "fonts" / "extra":
        note = "font training (font tambahan, run --extra-fonts fonts/extra)"
    else:
        note = ("bukan font training (di luar fonts/ dan fonts/extra/); kekerabatannya dengan font training tidak "
                "diperiksa")
    gap = space_ratio(str(font))
    if gap < MIN_SPACE_RATIO:
        note += (f"; celah kata {gap:.2f} x lebar ka < {MIN_SPACE_RATIO}: training selalu membuang spasi font ini, "
                 "jadi bentuk berspasi tidak bermakna")
    return note


# --- cache bacaan ----------------------------------------------------------------------------------------


class ReadingCache:
    """Bacaan per citra, ditulis ke jsonl begitu selesai, supaya jalan panjang yang terputus bisa dilanjutkan.

    Kunci = isi, bukan nama: (SHA font, ukuran, pad, jarak, teks) -> {SHA checkpoint: teks keluaran}, hanya dari
    catatan dengan versi cache dan lingkungan (`environment()`: versi pustaka dan kode src/) yang sama. Render dan
    inferensi deterministik, jadi bacaan yang sama berlaku untuk run mana pun yang memakai citra dan checkpoint itu:
    --lines lebih besar, jarak tambahan, atau checkpoint yang baru selesai. Baris rusak (proses mati saat menulis)
    dilewati.
    """

    def __init__(self, path: Path, env: dict, fresh: bool = False):
        self.path, self.env = Path(path), env
        self.readings: dict[tuple, dict[str, str]] = {}
        self.loaded = self.foreign = self.broken = 0
        complete = True
        if not fresh and self.path.exists():
            complete = self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w" if fresh else "a", encoding="utf-8", newline="\n")
        if not complete:  # baris terakhir terpotong: jangan sambung bacaan baru ke baris itu
            self._file.write("\n")

    @staticmethod
    def key(font_sha: str, size: int, pad: int, tracking: float, text: str) -> tuple:
        return (font_sha, int(size), int(pad), float(tracking), text)

    def _load(self) -> bool:
        """Muat bacaan yang cocok; False bila berkas tidak berakhir dengan baris utuh."""
        raw = self.path.read_bytes()
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = self.key(record["font"], record["size"], record["pad"], record["tracking"], record["text"])
                usable = record["v"] == CACHE_VERSION and record["env"] == self.env
                readings = {str(sha): str(text) for sha, text in record["hyp"].items()}
            except (ValueError, KeyError, TypeError, AttributeError):
                self.broken += 1
                continue
            if usable:
                self.readings.setdefault(key, {}).update(readings)
                self.loaded += len(readings)
            else:
                self.foreign += len(readings)
        return not raw or raw.endswith(b"\n")

    def get(self, key: tuple, checkpoint_sha: str) -> str | None:
        """Teks keluaran yang tersimpan ("" = keluaran kosong yang sah), None bila belum ada."""
        return self.readings.get(key, {}).get(checkpoint_sha)

    def put(self, key: tuple, readings: dict[str, str]) -> None:
        """Simpan bacaan baru satu citra ({SHA checkpoint: teks}) dan tulis ke berkas saat itu juga."""
        self.readings.setdefault(key, {}).update(readings)
        font_sha, size, pad, tracking, text = key
        record = {"v": CACHE_VERSION, "env": self.env, "font": font_sha, "size": size, "pad": pad,
                  "tracking": tracking, "text": text, "hyp": readings}
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class Progress:
    """Kemajuan menurut bacaan yang benar-benar dihitung: bacaan dari cache tidak memakan waktu."""

    def __init__(self, total: int):
        self.total, self.done = total, 0
        self.started = self.last = time.time()

    def advance(self, readings: int, label: str, line: int, lines: int) -> None:
        self.done += readings
        now = time.time()
        if now - self.last < PROGRESS_SECONDS:
            return
        self.last = now
        elapsed = now - self.started
        remaining = elapsed / max(1, self.done) * max(0, self.total - self.done)
        print(f"  [{label}] baris {line}/{lines}; bacaan {self.done}/{self.total}, {elapsed / 60:.1f} mnt, sisa "
              f"~{remaining / 60:.0f} mnt (kasar: citra melebar dengan jarak)", flush=True)


# --- baris dan render ------------------------------------------------------------------------------------


def rejection(forms: dict[str, str], font: Path, tokenizer: Tokenizer, charset: frozenset[str],
              glyphless: frozenset[str], trackings: Sequence[float]) -> str | None:
    """Alasan baris ini tidak bisa dipakai, atau None. Semua bentuk dan jarak diperiksa.

    `glyphless` = karakter charset yang tidak ada di cmap font: font tetap merender kotak .notdef tanpa galat, jadi
    labelnya tidak cocok dengan citra (CarakanJawa tidak punya aksara O).
    """
    for form, text in forms.items():
        if not text:
            return f"kosong ({form})"
        # Label yang tidak kembali utuh lewat urutan visual tidak bisa dihasilkan model mana pun (bentuk tanpa-spasi:
        # 153 dari 51.756 baris val = 0,3%, pola pasangan yang tidak ada di tabel reorder; bentuk berspasi: 0); G1
        # resmi pun menilai label hasil round-trip.
        if any(ch not in charset for ch in text) or tokenizer.to_logical(tokenizer.to_visual(text)) != text:
            return f"tidak round-trip tokenizer ({form})"
        absent = sorted({ch for ch in text if ch in glyphless})
        if absent:
            return f"glyph tidak ada di font ({', '.join(cp_label(ch) for ch in absent)})"
        targets = len(tokenizer.encode(tokenizer.to_visual(text)))
        for tracking in trackings:
            try:
                image = render_line(text, font, RENDER_SIZE, tracking=tracking)
            except (RenderClipped, EmptyRender) as error:
                return f"gagal render ({form}, {tracking:g} em): {error}"
            if not is_trainable(targets, to_tensor(image).shape[-1]):
                return f"T < 1,5L ({form}, {tracking:g} em)"
    return None


def select_lines(texts: Sequence[str], order: Sequence[int], font: Path, tokenizer: Tokenizer,
                 trackings: Sequence[float]) -> tuple[list[dict], list[dict]]:
    """Baris yang dipakai di SEMUA kondisi font ini, dan yang ditolak beserta alasannya.

    `texts[i]` = baris split ke-`order[i]`. Satu himpunan baris untuk semua bentuk dan jarak, supaya kurva
    dosis-respons tidak tercampur perubahan sampel: baris ditolak seluruhnya bila salah satu bentuknya tidak
    round-trip tokenizer, memuat karakter yang glyph-nya tidak ada di font, gagal dirender, atau melanggar
    T >= 1,5L (ditolak seperti G1) pada salah satu jarak.
    """
    charset = frozenset(tokenizer.charset)
    covered = font_codepoints(str(font))
    # Spasi tidak diperiksa: GumregahNew tidak punya glyph spasi tetapi celah katanya dibentuk shaping (font_note).
    glyphless = frozenset(ch for ch in charset if ch != SPACE and ord(ch) not in covered)
    kept, dropped = [], []
    for i, text in enumerate(texts):
        text = nfc(text)
        forms = {NO_SPACE: text.replace(SPACE, ""), SPACED: text}
        reason = rejection(forms, font, tokenizer, charset, glyphless, trackings)
        if reason is None:
            kept.append({"i": i, "split_index": order[i], "text": forms})
        else:
            dropped.append({"i": i, "split_index": order[i], "reason": reason, "text": text})
    return kept, dropped


def warm_up(models: dict[str, tuple]) -> None:
    """Satu forward per model sebelum waktu diukur: forward pertama memuat kernel dan mengalokasikan memori."""
    blank = Image.new("L", WARMUP_SIZE, 255)
    for model, tokenizer in models.values():
        predict(blank, model, tokenizer, pad_ratio=0.0)


def pending_readings(lines: Sequence[dict], font_sha: str, trackings: Sequence[float], digests: dict[str, str],
                     cache: ReadingCache) -> int:
    """Jumlah bacaan (citra x checkpoint) yang belum ada di cache untuk semua kondisi font ini."""
    return sum(cache.get(cache.key(font_sha, RENDER_SIZE, DEFAULT_PAD, tracking, line["text"][form]), sha) is None
               for form in FORMS for tracking in trackings for line in lines for sha in digests.values())


def read_condition(lines: Sequence[dict], font: Path, font_sha: str, form: str, tracking: float,
                   models: dict[str, tuple], digests: dict[str, str], cache: ReadingCache, progress: Progress,
                   label: str) -> dict:
    """Render tiap baris sekali dan baca citranya dengan setiap checkpoint (citra sama = berpasangan).

    Bacaan yang sudah ada di cache dipakai; citra hanya dirender bila ada checkpoint yang belum membacanya. Waktu
    hanya diukur pada render dan bacaan yang benar-benar dijalankan.
    """
    rows = []
    render_ms, rendered = 0.0, 0
    infer_ms, computed = dict.fromkeys(models, 0.0), dict.fromkeys(models, 0)
    started = time.time()
    for line in lines:
        text = line["text"][form]
        key = cache.key(font_sha, RENDER_SIZE, DEFAULT_PAD, tracking, text)
        hyps = {name: cache.get(key, digests[name]) for name in models}
        missing = [name for name in models if hyps[name] is None]
        if missing:
            tick = time.perf_counter()
            image = render_line(text, font, RENDER_SIZE, tracking=tracking)
            render_ms += (time.perf_counter() - tick) * 1000
            rendered += 1
            fresh = {}
            for name in missing:
                model, tokenizer = models[name]
                tick = time.perf_counter()
                hyps[name] = fresh[digests[name]] = nfc(predict(image, model, tokenizer, pad_ratio=0.0))
                infer_ms[name] += (time.perf_counter() - tick) * 1000
                computed[name] += 1
            cache.put(key, fresh)
            progress.advance(len(missing), label, len(rows) + 1, len(lines))
        rows.append({"i": line["i"], "split_index": line["split_index"], "reference": text, "hyp": hyps})
    return {"rows": rows,
            "timing": {"form": form, "tracking": tracking, "lines": len(rows), "rendered": rendered,
                       "render_ms": render_ms / rendered if rendered else None,
                       "infer_ms": {name: infer_ms[name] / computed[name] if computed[name] else None
                                    for name in models},
                       "computed": computed, "wall_s": time.time() - started}}


# --- per baris -------------------------------------------------------------------------------------------


def boundary_count(text: str) -> int:
    """Batas antar suku kata ortografis yang bukan spasi: pasangan suku kata bersebelahan yang keduanya bukan spasi.

    Itulah tempat spasi palsu bisa muncul di antara aksara. Spasi adalah "suku kata" sendiri di logical_syllables,
    jadi batas di kiri dan kanan spasi tidak dihitung; rantai pasangan (pangkon + aksara) tetap satu suku kata.
    """
    syllables = logical_syllables(text)
    return sum(1 for left, right in zip(syllables, syllables[1:]) if not left.isspace() and not right.isspace())


def split_spaces(text: str) -> tuple[str, list[int]]:
    """Teks -> (karakter bukan-spasi, jumlah spasi di tiap celah).

    Celah k = tepat sebelum karakter bukan-spasi ke-k; celah terakhir = ujung baris. Jumlah celah = karakter + 1.
    """
    glyphs, gaps = [], [0]
    for ch in text:
        if ch == SPACE:
            gaps[-1] += 1
        else:
            glyphs.append(ch)
            gaps.append(0)
    return "".join(glyphs), gaps


def line_stats(reference: str, hypothesis: str) -> dict:
    """Angka satu baris (string NFC) untuk satu checkpoint.

    Spasi adalah isi celah di antara dua karakter bukan-spasi. Karakter bukan-spasi referensi dan keluaran dijajarkan
    dengan compare_runs.alignment (biaya Levenshtein minimum, lalu kecocokan terbanyak, lalu substitusi sekelas).
    Spasi keluaran "asli" bila celahnya bertemu celah referensi yang berisi spasi pada penjajaran itu (satu spasi
    referensi memasangkan paling banyak satu spasi keluaran); spasi keluaran lain palsu, di mana pun letaknya.

    Penjajaran string utuh TIDAK dipakai untuk memasangkan spasi. Untuk "ka spasi na ta" vs "ka na spasi ta" ia seri
    antara mencocokkan na dan mencocokkan spasi, dan pemutus serinya memilih spasi bila spasi keluaran bergeser satu
    aksara ke kanan (tidak bila ke kiri). Terukur 2026-10-03 pada 8 baris berspasi, 0,3 em, fase6_ctrl: 26 dari 38
    spasi "terbaca" menurut penjajaran string utuh, 17 dari 38 bila celahnya harus sama.

    Penjajaran hanya dihitung bila kedua sisi memuat spasi: tanpa spasi di salah satu sisi tidak ada yang berpasangan
    (penjajarannya pure Python, ~2 ms per baris). Spasi dipasangkan pada teks keluaran apa adanya; jarak edit yang
    spasinya dibuang dihitung sesudah NFC, seperti src.evaluate.summarize (lihat komentar sebelum `return`).
    """
    ref_glyphs, ref_gaps = split_spaces(reference)
    hyp_glyphs, hyp_gaps = split_spaces(hypothesis)
    ref_spaces, hyp_spaces = sum(ref_gaps), sum(hyp_gaps)
    matched = 0
    if ref_spaces and hyp_spaces:
        ops = alignment(ref_glyphs, hyp_glyphs)
        cost = Levenshtein.distance(ref_glyphs, hyp_glyphs)
        if sum(tag != "equal" for tag, _, _ in ops) != cost:  # penjajaran wajib berbiaya minimum
            raise EvalError(f"penjajaran {ref_glyphs!r} vs {hyp_glyphs!r} tidak berbiaya minimum ({cost} edit)")
        # Lintasan penjajaran: di tiap simpul (i, j), celah referensi i bertemu celah keluaran j. Karakter keluaran
        # sisipan membuat dua celah keluaran bertemu satu celah referensi; karakter referensi yang hilang sebaliknya.
        i = j = 0
        met = [(0, 0)]
        for tag, _, _ in ops:
            if tag != "insert":
                i += 1
            if tag != "delete":
                j += 1
            met.append((i, j))
        for i, j in met:
            pairs = min(ref_gaps[i], hyp_gaps[j])
            matched += pairs
            ref_gaps[i] -= pairs
            hyp_gaps[j] -= pairs
    # NFC sesudah spasi dibuang, seperti src.evaluate.summarize: "pangkon + spasi + cecak telu" adalah NFC, tetapi
    # tanpa spasinya pangkon (kelas gabung 9) bertemu cecak telu (7) dan NFC menukar keduanya. Tanpa normalisasi ini
    # jumlah per baris tidak mereproduksi angka resmi dan summarize_condition menghentikan seluruh evaluasi.
    ref_bare, hyp_bare = nfc(ref_glyphs), nfc(hyp_glyphs)
    return {
        "chars": len(reference),
        "edits": Levenshtein.distance(reference, hypothesis),
        # Sama dengan cer_hyp_no_space di src/evaluate.py: spasi dibuang dari keluaran saja, label apa adanya.
        "edits_no_space": Levenshtein.distance(reference, hyp_bare),
        # Tak peka spasi: spasi dibuang dari kedua sisi, jadi hanya bentuk aksara yang dinilai.
        "glyph_chars": len(ref_bare),
        "glyph_edits": Levenshtein.distance(ref_bare, hyp_bare),
        "boundaries": boundary_count(reference),
        "ref_spaces": ref_spaces,
        "hyp_spaces": hyp_spaces,
        "matched_spaces": matched,
        "false_spaces": hyp_spaces - matched,
    }


# --- ringkasan -------------------------------------------------------------------------------------------


def summarize_condition(pairs: Sequence[tuple[str, str]], stats: Sequence[dict]) -> dict:
    """Ringkasan satu checkpoint pada satu kondisi. `pairs` = (referensi, keluaran) NFC, `stats` = line_stats-nya."""
    official = summarize(pairs)
    total = {field: sum(s[field] for s in stats) for field in stats[0]} if stats else {}
    for field, name in (("edits", "cer"), ("edits_no_space", "cer_hyp_no_space")):
        mine = total.get(field, 0) / max(1, total.get("chars", 0))
        if abs(mine - official[name]) > TOLERANCE:
            raise EvalError(f"jumlah edit per baris ({mine:.6%}) tidak mereproduksi src.evaluate.summarize "
                            f"({official[name]:.6%}) untuk {name}")
    return {
        "lines": len(pairs),
        "reference_chars": total.get("chars", 0),
        "boundaries": total.get("boundaries", 0),
        "ref_spaces": total.get("ref_spaces", 0),
        "hyp_spaces": total.get("hyp_spaces", 0),
        "matched_spaces": total.get("matched_spaces", 0),
        "false_spaces": total.get("false_spaces", 0),
        # Rasio (x 100 = spasi palsu per 100 batas suku kata); None bila penyebut 0.
        "false_per_boundary": ratio(total.get("false_spaces", 0), total.get("boundaries", 0)),
        "space_recall": ratio(total.get("matched_spaces", 0), total.get("ref_spaces", 0)),
        "cer": official["cer"],
        "cer_hyp_no_space": official["cer_hyp_no_space"],
        "cer_space_insensitive": ratio(total.get("glyph_edits", 0), total.get("glyph_chars", 0)),
        "exact_line_accuracy": official["exact_line_accuracy"],
    }


def compare_pair(stats_a: dict[tuple, Sequence[dict]], stats_b: dict[tuple, Sequence[dict]], resamples: int,
                 seed: int) -> list[dict]:
    """Selisih A - B per (bentuk, jarak) pada baris dan citra yang sama: rasio-jumlah dan bootstrap per baris.

    `stats_x[(bentuk, jarak)]` = line_stats per baris, baris yang sama dan urutan yang sama di setiap kondisi, jadi
    satu set resample dipakai untuk semua jarak. Metrik yang penyebutnya 0 di referensi (recall spasi pada bentuk
    tanpa-spasi) tidak dihitung.
    """
    columns: dict[tuple, tuple] = {}
    for condition, lines_a in stats_a.items():
        lines_b = stats_b[condition]
        for key, (num, den, _) in PAIR_METRICS.items():
            denominator = [s[den] for s in lines_a]  # dari referensi: sama untuk A dan B
            if any(denominator):
                columns[condition, key] = ([s[num] for s in lines_a], denominator, [s[num] for s in lines_b],
                                           denominator)
    boot = bootstrap_diffs(columns, resamples, seed)
    rows = []
    for form, tracking in stats_a:
        metrics = {}
        for key in PAIR_METRICS:
            if ((form, tracking), key) not in columns:
                continue
            num_a, den, num_b, _ = columns[(form, tracking), key]
            a, b = ratio(sum(num_a), sum(den)), ratio(sum(num_b), sum(den))
            metrics[key] = {"a": a, "b": b, "diff": None if a is None or b is None else a - b,
                            "bootstrap": boot[(form, tracking), key]}
        rows.append({"form": form, "tracking": tracking, "metrics": metrics})
    return rows


def reference_totals(texts: Sequence[str]) -> dict:
    """Jumlah di referensi satu bentuk: penyebut metrik, sama untuk setiap checkpoint dan jarak."""
    return {"chars": sum(len(text) for text in texts), "boundaries": sum(boundary_count(text) for text in texts),
            "spaces": sum(text.count(SPACE) for text in texts)}


def sanity_checks(fonts: dict[str, dict], checkpoints: dict[str, dict]) -> list[dict]:
    """Kontrol positif: checkpoint yang dilatih tanpa jarak harus mereproduksi gejalanya pada bentuk tanpa-spasi.

    Per font dan checkpoint: angka pada jarak terkecil dan pada 0,3 em (None bila jarak itu tidak dijalankan).
    """
    out = []
    for font_name, block in fonts.items():
        for name, info in checkpoints.items():
            if info.get("track_prob"):
                continue
            rows = {row["tracking"]: row for row in block["results"][name][NO_SPACE]}
            low, high = rows[min(rows)], rows.get(TRAIN_TRACK_MAX)
            out.append({"font": font_name, "checkpoint": name,
                        **{side: None if row is None else
                           {key: row[key] for key in ("tracking", "false_per_boundary", "cer_hyp_no_space")}
                           for side, row in (("low", low), ("high", high))}})
    return out


def mean_timing(timings: Sequence[dict], names: Sequence[str]) -> dict:
    """ms per citra: rata-rata semua kondisi, berbobot jumlah citra yang benar-benar dirender/dibaca (None = nol)."""
    rendered = sum(t["rendered"] for t in timings)
    render = sum(t["render_ms"] * t["rendered"] for t in timings if t["render_ms"] is not None)
    infer = {}
    for name in names:
        computed = sum(t["computed"][name] for t in timings)
        total = sum(t["infer_ms"][name] * t["computed"][name] for t in timings if t["infer_ms"][name] is not None)
        infer[name] = total / computed if computed else None
    return {"render_ms": render / rendered if rendered else None, "infer_ms": infer}


def fit_ms(samples: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """(a, b) untuk ms = a + b x jarak dari titik (jarak, ms), kuadrat terkecil; satu jarak saja = konstan."""
    if len({tracking for tracking, _ in samples}) < 2:
        return sum(ms for _, ms in samples) / len(samples), 0.0
    slope, intercept = np.polyfit([tracking for tracking, _ in samples], [ms for _, ms in samples], 1)
    return float(intercept), float(slope)


def estimate_full(measured: Sequence[dict], lines: int, trackings: Sequence[float], n_checkpoints: int) -> dict | None:
    """Detik untuk jalan penuh dengan cache kosong: `lines` baris x 2 bentuk x `trackings` x 1 font x `n_checkpoints`.

    `measured` = waktu per kondisi dari read_condition (ms per citra yang benar-benar dirender/dibaca; inferensi =
    rata-rata checkpoint yang diukur, arsitekturnya sama). Citra melebar linear dengan jarak, jadi ms per citra pada
    jarak yang tidak diukur diperkirakan dengan garis lurus per bentuk; `flat` = ada bentuk yang hanya terukur pada
    satu jarak, sehingga ms per citranya dianggap sama di semua jarak. None bila ada bentuk yang tidak terukur.
    """
    render_s = infer_s = 0.0
    flat = False
    for form in FORMS:
        render, infer = [], []
        for timing in measured:
            if timing["form"] != form:
                continue
            if timing["render_ms"] is not None:
                render.append((timing["tracking"], timing["render_ms"]))
            read = [ms for ms in timing["infer_ms"].values() if ms is not None]
            if read:
                infer.append((timing["tracking"], sum(read) / len(read)))
        if not render or not infer:
            return None
        flat = flat or len({tracking for tracking, _ in infer}) < 2
        (render_a, render_b), (infer_a, infer_b) = fit_ms(render), fit_ms(infer)
        for tracking in trackings:
            render_s += lines * max(0.0, render_a + render_b * tracking) / 1000
            infer_s += lines * n_checkpoints * max(0.0, infer_a + infer_b * tracking) / 1000
    seen = sorted({timing["tracking"] for timing in measured if timing["render_ms"] is not None})
    return {"lines": lines, "forms": len(FORMS), "trackings": list(trackings), "fonts": 1,
            "checkpoints": n_checkpoints, "render_s": render_s, "infer_s": infer_s, "seconds": render_s + infer_s,
            "measured_trackings": seen, "flat": flat,
            "extrapolated": max(trackings) > seen[-1] or min(trackings) < seen[0]}


# --- laporan ---------------------------------------------------------------------------------------------


def em(tracking: float) -> str:
    """Jarak untuk tabel, mis. '0.3'; bertanda * bila di luar rentang latih fase7."""
    return f"{tracking:g}{'*' if tracking > TRAIN_TRACK_MAX else ''}"


def per100(value: float | None) -> str:
    """Rasio per batas -> 'per 100 batas' (bukan persen: bisa melebihi 100)."""
    return "-" if value is None else f"{value * 100:.1f}"


def training_tracking(info: dict) -> str:
    if info.get("track_prob"):
        return f"jarak saat training p={info['track_prob']:g}, 0-{info.get('track_max') or 0:g} em"
    return "tanpa jarak saat training"


def dropped_text(dropped: Sequence[dict]) -> str:
    """Mis. 'tidak round-trip tokenizer 1, T < 1,5L 2' (alasan tanpa rincian bentuk/jarak)."""
    reasons: dict[str, int] = {}
    for entry in dropped:
        reason = entry["reason"].split(" (")[0]
        reasons[reason] = reasons.get(reason, 0) + 1
    return ", ".join(f"{reason} {n}" for reason, n in reasons.items()) or "tidak ada"


def used_lines(fonts: dict[str, dict]) -> int:
    """Baris yang benar-benar dinilai, sesudah penolakan; dengan beberapa font: font yang barisnya paling sedikit.

    Angka inilah yang dibandingkan dengan ">= 300 baris" aturan analisis, bukan --lines: baris yang ditolak tidak
    dibaca (sampel seed 0: baris ke-199 tidak round-trip tanpa spasi, jadi --lines 300 memakai 299 baris).
    """
    return min(block["lines"] for block in fonts.values())


def matrix(block: dict, names: Sequence[str], form: str, field: str, show) -> list[list[str]]:
    """Baris = jarak, kolom = checkpoint: [jarak, nilai per checkpoint...]."""
    results = block["results"]
    return [[em(row["tracking"]), *(show(results[name][form][k][field]) for name in names)]
            for k, row in enumerate(results[names[0]][form])]


def checkpoint_table(results: dict) -> list[str]:
    out = [
        "| jarak (em) | tanpa-spasi: spasi palsu /100 batas | CER | CER spasi dibuang | berspasi: spasi palsu /100 "
        "batas | recall spasi | CER | CER spasi dibuang | CER tak peka spasi |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bare, spaced in zip(results[NO_SPACE], results[SPACED], strict=True):
        out.append(f"| {em(bare['tracking'])} | {per100(bare['false_per_boundary'])} | {pct(bare['cer'])} | "
                   f"{pct(bare['cer_hyp_no_space'])} | {per100(spaced['false_per_boundary'])} | "
                   f"{pct(spaced['space_recall'])} | {pct(spaced['cer'])} | {pct(spaced['cer_hyp_no_space'])} | "
                   f"{pct(spaced['cer_space_insensitive'])} |")
    return out


def pair_table(comparison: dict) -> list[str]:
    out = ["| metrik | bentuk | jarak (em) | A | B | A - B | SK 95% |", "|---|---|---:|---:|---:|---:|---:|"]
    for key, form in PAIR_ROWS:
        show = per100 if key == "false_per_boundary" else pct
        for row in comparison["rows"]:
            if row["form"] == form and key in row["metrics"]:
                m = row["metrics"][key]
                out.append(f"| {PAIR_METRICS[key][2]} | {form} | {em(row['tracking'])} | {show(m['a'])} | "
                           f"{show(m['b'])} | {points(m['diff'])} | {interval(m['bootstrap'])} |")
    return out


def sanity_text(entry: dict) -> str:
    low, high = entry["low"], entry["high"]
    text = (f"`{entry['checkpoint']}`, {entry['font']}, tanpa-spasi: {low['tracking']:g} em -> "
            f"{per100(low['false_per_boundary'])} spasi palsu per 100 batas (CER spasi dibuang "
            f"{pct(low['cer_hyp_no_space'])})")
    if high is None:
        return text + f"; jarak {TRAIN_TRACK_MAX:g} em tidak dijalankan"
    if high["tracking"] == low["tracking"]:
        return text
    return text + (f"; {high['tracking']:g} em -> {per100(high['false_per_boundary'])} (CER spasi dibuang "
                   f"{pct(high['cer_hyp_no_space'])})")


def estimate_text(estimate: dict | None, threads: int) -> str:
    if estimate is None:
        return ("perkiraan jalan penuh: tidak ada (ada bentuk yang semua bacaannya dari cache, jadi waktunya tidak "
                "terukur)")
    distances = " ".join(f"{tracking:g}" for tracking in estimate["trackings"])
    measured = " ".join(f"{tracking:g}" for tracking in estimate["measured_trackings"])
    if estimate["flat"]:
        basis = f"diukur hanya pada {measured} em: ms per citra dianggap sama di semua jarak, padahal citra melebar"
    else:
        basis = f"diukur pada {measured} em; jarak lain dari garis lurus ms per citra terhadap jarak"
    return (f"perkiraan jalan penuh dengan cache kosong ({estimate['lines']} baris x {estimate['forms']} bentuk x "
            f"{len(estimate['trackings'])} jarak [{distances} em] x {estimate['fonts']} font x "
            f"{estimate['checkpoints']} checkpoint, {threads} thread): ~{estimate['seconds'] / 60:.0f} menit (render "
            f"{estimate['render_s'] / 60:.1f}, inferensi {estimate['infer_s'] / 60:.0f} = "
            f"~{estimate['infer_s'] / 60 / max(1, estimate['checkpoints']):.0f} menit per checkpoint; {basis})")


def render_markdown(report: dict) -> str:
    settings, cache, timing = report["settings"], report["cache"], report["timing"]
    names = list(report["checkpoints"])
    boot, env = settings["bootstrap"], settings["environment"]
    distances = ", ".join(f"{tracking:g}" for tracking in settings["tracking"])
    used = used_lines(report["fonts"])
    out = [
        "# Evaluasi sintetis dosis-respons jarak antar suku kata",
        "",
        "Citra yang SAMA untuk setiap checkpoint (berpasangan), tanpa NusaAksara: titik akhir utama eksperimen "
        "tracking fase 7 (aturan analisis fase 7 di CLAUDE.md).",
        "",
        f"- Baris: {settings['lines']} dari {settings['split_lines']} baris split val (`{settings['split']}`), sampel "
        f"acak seed {settings['seed']} (indeks diacak, N pertama). Bukan train, bukan NusaAksara; teks val tidak "
        "pernah dilatihkan (500 baris pertamanya dipakai training untuk memantau val CER). Aturan analisis fase 7 "
        f"menyebut >= {settings['planned_min_lines']} baris: sesudah penolakan run ini memakai {used} baris"
        + (" (font yang barisnya paling sedikit)" if len(report["fonts"]) > 1 else "")
        + f" x {len(settings['forms'])} bentuk = {used * len(settings['forms'])} citra per jarak dan font"
        + ("" if used >= settings["planned_min_lines"] else ", KURANG dari yang direncanakan") + ".",
        "- Bentuk: " + "; ".join(f"`{form}` = {text}" for form, text in settings["forms"].items()) + ".",
        f"- Render: `{settings['render']}`. Jarak t (em) ditambahkan sesudah tiap suku kata ortografis termasuk "
        "spasi, jadi celah antar-aksara bertambah t dan celah kata bertambah 2t (sama dengan augmentasi training "
        f"fase7). Jarak: {distances} em; 0-{settings['train_track_max']:g} em = rentang latih fase7_track "
        f"(`--track-max {settings['train_track_max']:g}`), di atasnya ekstrapolasi (bertanda *).",
        "- Font:",
        *[f"  - `{name}` (`{block['path']}`): {block['note']}." for name, block in report["fonts"].items()],
        "- Baris ditolak dari SEMUA kondisi sebuah font bila salah satu bentuknya tidak round-trip tokenizer, memuat "
        "karakter tanpa glyph di font, gagal dirender, atau melanggar T >= 1,5L (seperti G1) pada salah satu jarak: "
        "setiap jarak memakai baris yang sama.",
        f"- Inferensi: {settings['inference']}; {settings['threads']} thread torch. Lingkungan: Pillow "
        f"{env['pillow']}, RAQM {env['raqm']}, HarfBuzz {env['harfbuzz']}, torch {env['torch']}, kode src/ (render "
        f"sampai decode) `{env['code']}`.",
        "- Checkpoint (urutan = arah selisih A - B; pasangan pertama = dua yang pertama ada):",
        *[f"  - `{name}`: `{info['path']}` (langkah {info['step']}; {training_tracking(info)}; sisip aksara langka "
          f"p={info['rare_insert_prob'] or 0}; drop-space p={info['drop_space_prob'] or 0})"
          for name, info in report["checkpoints"].items()],
        *[f"  - DILEWATI, berkas tidak ada: `{entry['name']}` (`{entry['path']}`)"
          for entry in report["skipped_checkpoints"]],
        f"- SK 95% = bootstrap berpasangan per baris ({boot['resamples']} resample, seed {boot['seed']}, persentil "
        "2,5-97,5), satu set resample untuk semua jarak. Setiap baris dirender sendiri, jadi baris = satuan "
        "resampling.",
        f"- Bacaan (citra x checkpoint): {cache['computed']} dihitung pada run ini, {cache['reused']} dari cache "
        f"`{cache['file']}` (dipakai hanya bila isi font, isi checkpoint, teks, jarak, dan lingkungan di atas sama).",
        "",
        "Definisi (NFC, urutan logis):",
        "",
        "- Spasi palsu per 100 batas: spasi keluaran yang tidak berpasangan dengan spasi referensi, dibagi jumlah "
        "batas antar suku kata ortografis referensi yang bukan spasi (`logical_syllables`), x 100. Semua spasi palsu "
        "dihitung di mana pun letaknya (tepi baris, lebih dari satu per batas), jadi angkanya bisa melebihi 100.",
        "- Berpasangan = berada di celah yang sama. Karakter bukan-spasi referensi dan keluaran dijajarkan dengan "
        "`compare_runs.alignment` (biaya Levenshtein minimum, lalu kecocokan terbanyak, lalu substitusi sekelas); "
        "spasi keluaran dipasangkan dengan spasi referensi hanya bila celah keduanya bertemu pada penjajaran itu. "
        "Penjajaran string utuh tidak dipakai untuk spasi: spasi yang bergeser satu aksara seri dengan aksara itu, "
        "dan pemutus serinya menganggap spasi geser-kanan terbaca (8 baris berspasi, 0,3 em, `fase6_ctrl`: 26 dari 38 "
        "spasi \"terbaca\" vs 17 dari 38 bila celahnya harus sama).",
        "- Recall spasi (bentuk berspasi): spasi referensi yang berpasangan / jumlah spasi referensi.",
        "- CER: mikro, fungsi resmi `src.evaluate.summarize`. CER spasi dibuang = spasi dibuang dari keluaran saja "
        "(`cer_hyp_no_space`): kontrol bahwa bentuk aksara tetap terbaca. Pada bentuk berspasi spasi referensi yang "
        "ikut terbuang terhitung salah (lantai = porsi spasi di referensi), jadi di sana kontrolnya CER tak peka "
        "spasi (spasi dibuang dari kedua sisi); pada bentuk tanpa-spasi keduanya sama.",
        "- Selisih = A - B dalam poin (pt; untuk spasi palsu 1 pt = 1 spasi per 100 batas). Spasi palsu dan CER: "
        "negatif = A lebih baik. Recall spasi: positif = A lebih baik.",
    ]
    for font_name, block in report["fonts"].items():
        bare, spaced = block["reference"][NO_SPACE], block["reference"][SPACED]
        out += [
            "",
            f"## Font `{font_name}`",
            "",
            f"{block['lines']} baris dipakai (ditolak: {dropped_text(block['dropped'])}). Tanpa-spasi: "
            f"{bare['chars']} karakter, {bare['boundaries']} batas suku kata. Berspasi: {spaced['chars']} karakter, "
            f"{spaced['boundaries']} batas bukan-spasi, {spaced['spaces']} spasi.",
            "",
            "### Titik akhir utama: spasi palsu per 100 batas suku kata, bentuk `tanpa-spasi`",
            "",
            "| jarak (em) | " + " | ".join(f"`{name}`" for name in names) + " |",
            "|---:|" + "---:|" * len(names),
            *["| " + " | ".join(cells) + " |"
              for cells in matrix(block, names, NO_SPACE, "false_per_boundary", per100)],
            "",
            "### Dosis-respons per checkpoint",
        ]
        for name in names:
            out += ["", f"`{name}` ({training_tracking(report['checkpoints'][name])}):", "",
                    *checkpoint_table(block["results"][name])]
        if block["comparisons"]:
            out += ["", "### Selisih berpasangan"]
        for comparison in block["comparisons"]:
            out += ["", f"`{comparison['a']}` (A) - `{comparison['b']}` (B)"
                        + (", pasangan pertama:" if comparison["primary"] else ":"), "", *pair_table(comparison)]

    out += [
        "",
        "## Cek kewajaran (kontrol positif)",
        "",
        "Checkpoint yang dilatih tanpa jarak harus mereproduksi gejalanya pada bentuk tanpa-spasi: ~0 spasi palsu "
        f"pada jarak 0, jauh di atas 50 per 100 batas pada {TRAIN_TRACK_MAX:g} em, dengan CER spasi dibuang tetap "
        f"rendah. Dasar terukur 2026-10-03 (8 baris javatext, sebelum skrip ini ada): {BASELINE_NOTE}.",
        "",
        *([f"- {sanity_text(entry)}." for entry in report["sanity"]]
          or ["- Tidak ada checkpoint tanpa jarak pada run ini."]),
        "",
        "## Biaya",
        "",
        f"Diukur pada run ini ({settings['threads']} thread torch), hanya pada citra yang benar-benar dirender dan "
        "dibaca (\"-\" = semua dari cache); proses lain (training) mungkin sedang berjalan, jadi anggap kasar.",
        "",
        "| font | bentuk | jarak (em) | render ms/citra | "
        + " | ".join(f"inferensi `{name}` ms/citra" for name in names) + " | waktu dinding |",
        "|---|---|---:|---:|" + "---:|" * len(names) + "---:|",
    ]
    for font_name, block in report["fonts"].items():
        for t in block["timing"]:
            cells = ["-" if t["infer_ms"][name] is None else f"{t['infer_ms'][name]:.0f}" for name in names]
            out.append(f"| `{font_name}` | {t['form']} | {t['tracking']:g} | "
                       f"{'-' if t['render_ms'] is None else format(t['render_ms'], '.0f')} | " + " | ".join(cells)
                       + f" | {t['wall_s']:.0f} dtk |")
    out += [
        "",
        estimate_text(timing["estimate_full"], settings["threads"]).replace("perkiraan", "Perkiraan", 1)
        + ". Bacaan yang sudah ada di cache tidak dihitung ulang, jadi jalan ulang dan checkpoint yang menyusul hanya "
        "membayar bacaan baru.",
        "",
        "## Batasan",
        "",
        "- Citra sintetis bersih dengan jarak seragam per baris, dirender oleh fungsi yang sama dengan augmentasi "
        "training fase7 (`render_line(..., tracking=t)`). Untuk checkpoint yang dilatih dengan jarak, angka ini "
        "mengukur apakah augmentasinya dipelajari, bukan apakah ia berpindah ke cetakan nyata (perataan cetakan tidak "
        "seragam; tinta dan pindaian berbeda). G3 tetap angka untuk data nyata.",
        "- Font: lihat catatan per font di atas; javatext sekeluarga dengan font training CarakanJawa, jadi angka "
        "javatext bukan generalisasi ke font yang belum pernah dilihat.",
        "- Bentuk berspasi: celah kata = lebar spasi + 2 x jarak dan celah antar-aksara = jarak, seperti training "
        "fase7; perbedaan relatif keduanya mengecil saat jarak membesar.",
        f"- Ukuran render tetap {settings['size']} px (G1 memakai 56-72 px) dan tanpa augmentasi: efek jarak pada "
        "citra beraugmentasi tidak diukur di sini.",
        "- Penyebut spasi palsu = batas antar suku kata yang bukan spasi; pembilangnya semua spasi palsu, juga yang "
        "di tepi baris atau lebih dari satu per batas.",
        "- Satu run per checkpoint: SK hanya memuat keragaman baris, bukan keragaman antar-run training (seed, urutan "
        "batch).",
        "- Penjajaran karakter bukan-spasi bisa seri walau biaya minimum, kecocokan terbanyak, dan substitusi "
        "sekelas terbanyak (mis. satu dari dua aksara kembar hilang di samping spasi); pilihannya deterministik, "
        "tapi di sekitar aksara yang salah baca pembagian spasi asli/palsu bisa bergeser satu-dua.",
    ]
    return "\n".join(out) + "\n"


def summary_lines(report: dict) -> list[str]:
    settings, cache, timing = report["settings"], report["cache"], report["timing"]
    names = list(report["checkpoints"])
    width = max(12, *(len(name) for name in names))
    out = [f"{settings['lines']} baris val (seed {settings['seed']}) x {len(settings['forms'])} bentuk x "
           f"{len(settings['tracking'])} jarak, {settings['threads']} thread; checkpoint {', '.join(names)}; selisih "
           f"= A - B; * = di luar rentang latih 0-{settings['train_track_max']:g} em"]
    used = used_lines(report["fonts"])
    if used < settings["planned_min_lines"]:
        out.append(f"catatan: aturan analisis fase 7 menyebut >= {settings['planned_min_lines']} baris; run ini "
                   f"memakai {used} baris (dari {settings['lines']} yang diambil, sesudah penolakan) x "
                   f"{len(settings['forms'])} bentuk = {used * len(settings['forms'])} citra per jarak dan font")
    tables = (
        ("spasi palsu per 100 batas suku kata, tanpa-spasi (titik akhir utama)", NO_SPACE, "false_per_boundary",
         per100),
        ("spasi palsu per 100 batas suku kata, berspasi", SPACED, "false_per_boundary", per100),
        ("recall spasi, berspasi", SPACED, "space_recall", pct),
        ("CER literal, tanpa-spasi", NO_SPACE, "cer", pct),
        ("CER spasi dibuang dari keluaran, tanpa-spasi (kontrol bentuk aksara)", NO_SPACE, "cer_hyp_no_space", pct),
        ("CER literal, berspasi", SPACED, "cer", pct),
        ("CER spasi dibuang dari keluaran, berspasi (lantai = porsi spasi referensi)", SPACED, "cer_hyp_no_space", pct),
        ("CER tak peka spasi, berspasi (kontrol bentuk aksara)", SPACED, "cer_space_insensitive", pct),
    )
    for font_name, block in report["fonts"].items():
        bare, spaced = block["reference"][NO_SPACE], block["reference"][SPACED]
        out.append(f"== {font_name}: {block['lines']} baris dipakai (ditolak: {dropped_text(block['dropped'])}); "
                   f"batas suku kata tanpa-spasi {bare['boundaries']}, berspasi {spaced['boundaries']}; spasi "
                   f"referensi {spaced['spaces']}")
        for title, form, field, show in tables:
            out.append(f"-- {title}")
            out.append(f"   {'jarak em':>8}  " + "  ".join(f"{name:>{width}}" for name in names))
            out += [f"   {cells[0]:>8}  " + "  ".join(f"{cell:>{width}}" for cell in cells[1:])
                    for cells in matrix(block, names, form, field, show)]
        for comparison in block["comparisons"]:
            if not comparison["primary"]:
                continue
            out.append(f"-- selisih {comparison['a']} - {comparison['b']} (pasangan pertama), SK 95% bootstrap per "
                       "baris")
            for row in comparison["rows"]:
                cells = [f"{PAIR_METRICS[key][2]} {points(m['diff'])} {interval(m['bootstrap'])}"
                         for key, m in row["metrics"].items()]
                out.append(f"   {row['form']:11} {em(row['tracking']):>5} em: " + "; ".join(cells))
        others = len(block["comparisons"]) - 1
        if others > 0:
            out.append(f"   ({others} pasangan lain ada di laporan .md/.json)")
    out.append(f"cek kewajaran (checkpoint tanpa jarak saat training; dasar 2026-10-03: {BASELINE_NOTE}):")
    out += [f"  {sanity_text(entry).replace('`', '')}" for entry in report["sanity"]] or ["  tidak ada"]
    for font_name, block in report["fonts"].items():
        mean = mean_timing(block["timing"], names)
        if mean["render_ms"] is not None:
            out.append(f"waktu per citra {font_name} (rata-rata semua kondisi; rincian per jarak di .md): render "
                       f"{mean['render_ms']:.0f} ms; inferensi "
                       + ", ".join(f"{name} {ms:.0f} ms" for name, ms in mean["infer_ms"].items() if ms is not None))
    out.append(f"bacaan (citra x checkpoint): {cache['computed']} dihitung pada run ini, {cache['reused']} dari cache")
    out.append(estimate_text(timing["estimate_full"], settings["threads"]))
    return out


# --- skrip -----------------------------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lines", type=int, default=DEFAULT_LINES, help="baris split val; 0 = semua")
    parser.add_argument("--seed", type=int, default=0, help="seed sampel baris dan bootstrap")
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS), metavar="NAMA[=PATH]",
                        help="NAMA=PATH, atau NAMA = out/checkpoints/NAMA/last_snapshot.pt; urutan = arah selisih; "
                             f"yang tidak ada dilewati (default {' '.join(DEFAULT_CHECKPOINTS)})")
    parser.add_argument("--fonts", nargs="+", default=[str(WINDOWS_JAVANESE_TEXT)], metavar="FONT",
                        help="font yang dirender (default javatext); daftar ini mengganti bawaan")
    parser.add_argument("--tracking", nargs="+", type=float, default=list(DEFAULT_TRACKING), metavar="EM",
                        help="jarak tambahan antar suku kata (em; default "
                             f"{' '.join(f'{tracking:g}' for tracking in DEFAULT_TRACKING)})")
    parser.add_argument("--threads", type=int, default=0, help="thread torch untuk inferensi; 0 = bawaan torch")
    parser.add_argument("--bootstrap", type=int, default=10_000, help="jumlah resample; 0 = tanpa selang kepercayaan")
    parser.add_argument("--fresh", action="store_true", help="abaikan cache bacaan dan tulis ulang berkasnya")
    parser.add_argument("--out", default=str(OUT_DIR), help="direktori keluaran (default out/compare)")
    args = parser.parse_args(argv)
    for option in ("lines", "threads", "bootstrap"):
        if getattr(args, option) < 0:
            parser.error(f"--{option} tidak boleh negatif")
    if any(tracking < 0 for tracking in args.tracking):
        parser.error("--tracking tidak boleh negatif")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.threads:
        torch.set_num_threads(args.threads)
    started = time.time()
    out_dir = Path(args.out)
    fonts = [Path(font) for font in args.fonts]
    trackings = sorted(set(args.tracking))

    try:
        for font in fonts:
            if not font.is_file():
                raise EvalError(f"font tidak ada: {font}")
        if len({font.name for font in fonts}) != len(fonts):
            raise EvalError("--fonts: nama berkas font harus berbeda (laporan memakai nama berkas)")
        checkpoints, skipped = resolve_checkpoints(args.checkpoints)
        for entry in skipped:
            print(f"dilewati: checkpoint '{entry['name']}' tidak ada ({entry['path']})", flush=True)
        tokenizer = Tokenizer.load(TOKENIZER_PATH)
        val_lines = read_split("val")
        order = sample_indices(len(val_lines), args.lines, args.seed)
        texts = [val_lines[i] for i in order]
        info = {name: checkpoint_info(path) for name, path in checkpoints.items()}
        models = load_models(checkpoints, tokenizer)
    except EvalError as error:
        raise SystemExit(f"galat: {error}") from None

    names = list(checkpoints)
    digests = {name: info[name]["sha1"] for name in names}
    env = environment()
    print(f"eval_spacing: {len(texts)} baris val (seed {args.seed}) x {len(FORMS)} bentuk x jarak "
          f"{' '.join(f'{tracking:g}' for tracking in trackings)} em, font {', '.join(font.name for font in fonts)}, "
          f"checkpoint {', '.join(names)}; {torch.get_num_threads()} thread torch", flush=True)
    warm_up(models)

    cache_path = out_dir / f"{STEM}_cache.jsonl"
    partial_path = out_dir / f"{STEM}_partial.jsonl"
    cache = ReadingCache(cache_path, env, fresh=args.fresh)
    if cache.loaded or cache.foreign or cache.broken:
        print(f"cache {relative(cache_path)}: {cache.loaded} bacaan dimuat"
              + (f", {cache.foreign} dari versi/lingkungan lain diabaikan" if cache.foreign else "")
              + (f", {cache.broken} baris rusak dilewati" if cache.broken else ""), flush=True)
    font_blocks, records = {}, []
    try:
        units = []
        print(f"memeriksa {len(texts)} baris di {len(FORMS) * len(trackings)} kondisi per font (round-trip, glyph, "
              "render, T >= 1,5L)", flush=True)
        for font in fonts:
            tick = time.time()
            kept, dropped = select_lines(texts, order, font, tokenizer, trackings)
            print(f"  [{font.name}] {len(kept)} baris dipakai, {len(dropped)} ditolak "
                  f"({dropped_text(dropped)}); {time.time() - tick:.0f} dtk", flush=True)
            if not kept:
                raise SystemExit(f"galat: font {font.name}: tidak ada baris yang bisa dipakai (semua ditolak)")
            units.append((font, file_sha(font), kept, dropped))
        progress = Progress(sum(pending_readings(kept, font_sha, trackings, digests, cache)
                                for _, font_sha, kept, _ in units))
        total_readings = sum(len(kept) for _, _, kept, _ in units) * len(FORMS) * len(trackings) * len(names)
        print(f"bacaan (citra x checkpoint): {total_readings}, {progress.total} belum ada di cache", flush=True)
        conditions, done = len(fonts) * len(FORMS) * len(trackings), 0
        with partial_path.open("w", encoding="utf-8", newline="\n") as partial:
            for font, font_sha, kept, dropped in units:
                results = {name: {form: [] for form in FORMS} for name in names}
                stats: dict[str, dict[tuple, list[dict]]] = {name: {} for name in names}
                timings = []
                for form in FORMS:
                    for tracking in trackings:
                        label = f"{font.name} | {form} | {tracking:g} em"
                        read = read_condition(kept, font, font_sha, form, tracking, models, digests, cache, progress,
                                              label)
                        summaries = {}
                        for name in names:
                            pairs = [(row["reference"], row["hyp"][name]) for row in read["rows"]]
                            stats[name][form, tracking] = [line_stats(ref, hyp) for ref, hyp in pairs]
                            summaries[name] = summarize_condition(pairs, stats[name][form, tracking])
                            results[name][form].append({"tracking": tracking, **summaries[name]})
                        timings.append(read["timing"])
                        records += [{"font": font.name, "form": form, "tracking": tracking, **row}
                                    for row in read["rows"]]
                        done += 1
                        partial.write(json.dumps({"font": font.name, "form": form, "tracking": tracking,
                                                  "results": summaries, "timing": read["timing"]},
                                                 ensure_ascii=False) + "\n")
                        partial.flush()
                        print(f"  [{label}] kondisi {done}/{conditions}, {len(read['rows'])} baris, "
                              f"{read['timing']['wall_s']:.0f} dtk: spasi palsu per 100 batas "
                              + ", ".join(f"{name} {per100(s['false_per_boundary'])}" for name, s in summaries.items())
                              + "; CER tak peka spasi "
                              + ", ".join(pct(s["cer_space_insensitive"]) for s in summaries.values()), flush=True)
                records += [{"font": font.name, **entry} for entry in dropped]
                font_blocks[font.name] = {
                    "path": relative(font), "sha1": font_sha, "note": font_note(font),
                    "space_ratio": space_ratio(str(font)), "lines": len(kept), "dropped": dropped,
                    "reference": {form: reference_totals([line["text"][form] for line in kept]) for form in FORMS},
                    "results": results,
                    "comparisons": [{"a": a, "b": b, "primary": k == 0,
                                     "rows": compare_pair(stats[a], stats[b], args.bootstrap, args.seed)}
                                    for k, (a, b) in enumerate(combinations(names, 2))],
                    "timing": timings,
                }
    except EvalError as error:
        raise SystemExit(f"galat: {error}") from None
    finally:
        cache.close()

    measured = [timing for block in font_blocks.values() for timing in block["timing"]]
    computed = sum(sum(timing["computed"].values()) for timing in measured)
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "split": relative(ROOT / "data" / "splits" / "val.txt"), "split_lines": len(val_lines),
            "lines": len(texts), "seed": args.seed, "planned_min_lines": PLANNED_MIN_LINES,
            "sampling": "indeks split val diacak random.Random(seed).shuffle, N pertama",
            "forms": FORMS, "size": RENDER_SIZE, "pad": DEFAULT_PAD, "tracking": trackings,
            "train_track_max": TRAIN_TRACK_MAX,
            "render": f"src.render.render_line(teks, font, {RENDER_SIZE}, tracking=t), bersih, tanpa augmentasi",
            "inference": "src.infer.predict satu baris (greedy, to_tensor, urutan logis, NFC, pad_ratio 0), CPU",
            "threads": torch.get_num_threads(), "environment": env,
            "units": "false_per_boundary x 100 = spasi palsu per 100 batas suku kata; rasio lain = pecahan 0-1",
            "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "interval": "persentil 2,5-97,5",
                          "unit": "baris"},
        },
        "checkpoints": info,
        "skipped_checkpoints": skipped,
        "fonts": font_blocks,
        "sanity": sanity_checks(font_blocks, info),
        "cache": {"file": relative(cache_path), "version": CACHE_VERSION, "computed": computed,
                  "reused": total_readings - computed},
        "timing": {"estimate_full": estimate_full(measured, DEFAULT_LINES, DEFAULT_TRACKING, len(DEFAULT_CHECKPOINTS)),
                   "total_wall_s": time.time() - started},
    }
    write_json(out_dir / f"{STEM}.json", report)
    (out_dir / f"{STEM}.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")
    with (out_dir / f"{STEM}_lines.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    partial_path.unlink(missing_ok=True)  # laporan akhir sudah memuat semua kondisi

    print("\n".join(summary_lines(report)))
    print(f"hasil: {relative(out_dir / STEM)}.json, .md, _lines.jsonl; cache {relative(cache_path)} "
          f"({time.time() - started:.0f} dtk)")


if __name__ == "__main__":
    main()
