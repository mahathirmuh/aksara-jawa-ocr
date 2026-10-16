"""Ekspor hasil eksperimen untuk web `aksara-ocr-web` (kontrak data, skema 1).

  .venv/Scripts/python scripts/export_results.py            # ~12 menit di CPU (5 checkpoint + beam, 745 baris)

Terukur 2026-10-03 dengan dua run fase6, tanpa training yang berjalan (out/after_fase6.log): 741 dtk; tiga checkpoint
dasar + beam 568 dtk, lalu tiap run lanjutan (fase6, fase7) yang ikut menambah satu lintasan inferensi 745 baris,
~86 dtk. Dengan kelima run lanjutan (8 checkpoint + beam) kira-kira 17 menit: perkiraan, belum diukur, dan lebih
lama bila training sedang berjalan.

Keluaran di out/results/ (ditimpa):
  manifest.json      skema, waktu, pipeline, gerbang G1-G4, metrik agregat, ablasi, kesalahan aksara
  lines.jsonl        satu baris data nyata per baris: id, path citra, ukuran, label, tag
  predictions.jsonl  satu prediksi per (baris, pipeline): teks, CER, segmen beda per suku kata, kolom citra

Angka resmi (gerbang G1-G3, kesalahan aksara, kunci "official" di manifest) = run OFFICIAL_RUN. Laporan resminya
(out/eval/<run>_G1_10k.json, _G2_10k.json, _G3_full.json) wajib ada dan sah; lihat `official_gates`.

Semua angka dihitung di sini; web hanya menampilkan. Angka 745 baris dicocokkan dengan laporan resmi
(out/eval/*_G3_full.json, termasuk run lanjutan yang ikut, dan out/beam/fonts_o5.json) dan skrip berhenti kalau
berbeda atau laporannya tidak ada. Run lanjutan (OPTIONAL_CHECKPOINTS: fase6_rare, fase6_ctrl, fase7_track,
fase7_track_rare, fase7_ctrl) ikut satu per satu bila last_snapshot.pt-nya ada; yang belum punya snapshot dilewati
tanpa galat. Snapshot dari run yang belum selesai (langkah != args.steps) atau yang argumen training-nya (jarak
antar-aksara, aksara langka, saringan) tidak cocok dengan jenis run menghentikan skrip sebelum inferensi.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.align import greedy_path, line_cer, syllable_diff, syllable_spans  # noqa: E402
from src.beam import line_log_probs, prefix_beam_search  # noqa: E402
from src.charlm import CharLM  # noqa: E402
from src.decode import cer  # noqa: E402
from src.evaluate import TARGETS  # noqa: E402
from src.infer import load_checkpoint  # noqa: E402
from src.tokenizer import nfc  # noqa: E402

SCHEMA = 1
LABELS = ROOT / "data/real/nusaaksara/labels.tsv"
LM_PATH = ROOT / "data/charlm/o5_n300000_ds0.5.pkl"
LM_STD_PATH = ROOT / "data/charlm/o5_n300000_ds0.5_std0.5.pkl"  # src.charlm --standard-prob 0.5 (fase 8)
ALPHA, BETA, WIDTH = 0.25, 1.0, 16
# Laporan scripts/beam_eval.py run resmi yang boleh dipakai: akhiran nama berkas (out/beam/<run>_<akhiran>.json) ->
# model bahasanya. Bila lebih dari satu ada, yang dipakai yang CER beam-nya di dev TERENDAH (dev-nya harus sama), bukan
# yang terbaik di data uji: begitulah model bahasa berejaan baku dipilih untuk fase8_std (2026-10-16, dev 1,82% lawan
# 2,04%).
BEAM_REPORTS = {"o5": LM_PATH, "o5std": LM_STD_PATH}
LM_LABELS = {LM_PATH.name: "LM karakter o5", LM_STD_PATH.name: "LM karakter o5 ejaan baku"}
DEV_KEYS = ("augment", "drop_space_prob", "standard_prob", "lines")  # setelan dev yang harus sama untuk dibandingkan
CHECKPOINTS = {
    "crnn_4b": ROOT / "out/checkpoints/base/best_4b_step1500.pt",
    "crnn_core": ROOT / "out/checkpoints/fase5_core/last_snapshot.pt",
    "crnn_fonts": ROOT / "out/checkpoints/fase5_fonts/last_snapshot.pt",
}
PIPELINES = [
    {"key": "crnn_4b", "label": "CRNN 4b", "kind": "crnn", "status": "done",
     "config": "crnn:base/best_4b_step1500 · 2 font · tanpa augmentasi · greedy"},
    {"key": "crnn_core", "label": "CRNN fase5_core", "kind": "crnn", "status": "done",
     "config": "crnn:fase5_core@1298 · 2 font · aug fase5 · greedy"},
    {"key": "crnn_fonts", "label": "CRNN fase5_fonts", "kind": "crnn", "status": "done",
     "config": "crnn:fase5_fonts@1298 · 10 font · aug fase5 · greedy"},
    {"key": "crnn_fonts_beam", "label": "Beam + LM", "kind": "beam", "status": "done",
     "config": f"crnn:fase5_fonts@1298 · beam {WIDTH} · LM karakter o5 · α {ALPHA} · β {BETA}"},
    {"key": "vlm_zeroshot", "label": "VLM zero-shot", "kind": "vlm", "status": "done",
     "config": "vlm:frontier · zero-shot · uji buta 2026-09-14 (50 baris)"},
    {"key": "vlm_finetune", "label": "VLM fine-tune", "kind": "vlm", "status": "planned",
     "config": "vlm:GraniteDocling-258M / Qwen3-VL-2B · LoRA"},
    {"key": "combo", "label": "Gabungan", "kind": "combo", "status": "planned",
     "config": "kandidat CRNN + VLM · dinilai CTC + LM"},
    {"key": "gpt", "label": "GPT korektor", "kind": "llm", "status": "planned",
     "config": "gpt:gpt-5.6-* · teks saja · menunggu persetujuan kirim data"},
    {"key": "sahabatai", "label": "Sahabat-AI", "kind": "llm", "status": "planned",
     "config": "llm:gemma2-9b-cpt-sahabatai"},
]
# Run lanjutan: ikut diekspor hanya bila out/checkpoints/<run>/last_snapshot.pt ada (greedy saja; beam tetap dari
# fase5_fonts), disisipkan dengan urutan daftar ini tepat setelah crnn_fonts_beam (`add_optional`). Kunci = "crnn_" +
# nama run (scripts/compare_runs.py mencari laporan resminya dengan membuang awalan itu). Run yang ikut wajib selesai
# dan argumen training di checkpoint-nya wajib cocok dengan jenis run (`check_optional_run`):
#   track     jarak tambahan antar-aksara (--track-prob > 0, --track-max > 0); False = track_prob harus 0
#   rare      sisipan aksara langka (--rare-insert-prob > 0); False = tanpa sisipan dan tanpa adeg-adeg pembuka
#   filtered  sisipan tersaring (--rare-max-similarity > 0 dan --rare-attach); False = keduanya mati. Hanya untuk rare
# Tanpa --limit, CER 745 barisnya wajib sama dengan out/eval/<run>_G3_full.json. Langkah, titik awal, augmentasi,
# peluang, dan ambang di config dibaca dari checkpoint; `detail` diisi `optional_pipeline`.
OPTIONAL_CHECKPOINTS = [
    {"key": "crnn_fase6_rare", "run": "fase6_rare", "label": "CRNN fase6_rare",
     "track": False, "rare": True, "filtered": False,
     "detail": "10 font · aug {augment} · + aksara langka (sisip {insert} · adeg-adeg {opener}) · greedy"},
    {"key": "crnn_fase6_ctrl", "run": "fase6_ctrl", "label": "CRNN fase6_ctrl",
     "track": False, "rare": False, "filtered": False,
     "detail": "10 font · aug {augment} · tanpa aksara langka (pembanding) · greedy"},
    {"key": "crnn_fase7_track", "run": "fase7_track", "label": "CRNN fase7_track",
     "track": True, "rare": False, "filtered": False,
     "detail": "10 font · aug {augment} · jarak antar-aksara p {track} hingga {track_max} em · greedy"},
    {"key": "crnn_fase7_track_rare", "run": "fase7_track_rare", "label": "CRNN fase7_track_rare",
     "track": True, "rare": True, "filtered": True,
     "detail": "10 font · aug {augment} · jarak antar-aksara p {track} hingga {track_max} em · + aksara langka "
               "tersaring (sisip {insert} · adeg-adeg {opener} · mirip >= {similarity} · tempel) · greedy"},
    {"key": "crnn_fase7_ctrl", "run": "fase7_ctrl", "label": "CRNN fase7_ctrl",
     "track": False, "rare": False, "filtered": False,
     "detail": "10 font · aug {augment} · tanpa jarak tambahan (pembanding) · greedy"},
    {"key": "crnn_fase8_std", "run": "fase8_std", "label": "CRNN fase8_std",
     "track": True, "rare": False, "filtered": False, "standard": True,
     "detail": "10 font · aug {augment} · jarak antar-aksara p {track} hingga {track_max} em · ejaan baku p {standard} "
               "· greedy"},
    {"key": "crnn_fase8_ctrl", "run": "fase8_ctrl", "label": "CRNN fase8_ctrl",
     "track": True, "rare": False, "filtered": False, "standard": False,
     "detail": "10 font · aug {augment} · jarak antar-aksara p {track} hingga {track_max} em · tanpa ejaan baku "
               "(pembanding) · greedy"},
    {"key": "crnn_fase9_slant", "run": "fase9_slant", "label": "CRNN fase9_slant",
     "track": True, "rare": False, "filtered": False, "standard": True, "slant": True, "fixes": True,
     "detail": "9 font · aug {augment} · jarak p {track} hingga {track_max} em · ejaan baku p {standard} · miring p {slant} "
               "hingga {slant_max} · perbaikan font · greedy"},
    {"key": "crnn_fase9_ctrl", "run": "fase9_ctrl", "label": "CRNN fase9_ctrl",
     "track": True, "rare": False, "filtered": False, "standard": True, "slant": False, "fixes": True,
     "detail": "9 font · aug {augment} · jarak p {track} hingga {track_max} em · ejaan baku p {standard} · tanpa miring "
               "(pembanding) · perbaikan font · greedy"},
]
CHECKPOINT_DIR = ROOT / "out/checkpoints"
EVAL_DIR = ROOT / "out/eval"
# Run resmi: gerbang G1-G3 di manifest dan tabel kesalahan aksara dihitung darinya, dan web menandai pipelinenya
# "angka resmi" (kunci "official" di manifest). Nama run = direktori checkpoint dan awalan laporan resminya di
# out/eval/ (LEGACY_REPORTS untuk run lama). Mengganti run resmi: buat dulu <run>_G1_10k.json dan <run>_G2_10k.json
# (10.000 baris javatext; scripts/make_official.sh) di samping <run>_G3_full.json, lalu ubah konstanta ini dan
# DEFAULT_CHECKPOINT di src/serve.py (Demo; tests/test_export_results.py menjaga keduanya sama).
# Riwayat: fase5_fonts sampai 2026-10-04, fase7_track sampai 2026-10-16, lalu fase8_std (keputusan user; ejaan baku
# membantu menurut aturan analisis fase 8 di CLAUDE.md).
OFFICIAL_RUN = "fase8_std"
OFFICIAL_LINES = 10_000  # baris sintetis per gerbang G1/G2; tes cepat 100 baris bukan angka resmi
LEGACY_REPORTS = {"fase5_fonts": "fonts", "fase5_core": "core"}  # laporan run lama memakai awalan pendek
TOL = 1e-9  # selisih CER terhadap laporan resmi yang masih dianggap sama
# Kolom pipelines.config di web bertipe string = varchar(255) di PostgreSQL: config yang lebih panjang menggagalkan
# `php artisan aksara:import` (SQLite, yang dipakai test web, tidak membatasi panjang).
MAX_CONFIG = 255

CLASSIC = set("ꦯꦉꦟꦡꦑꦣꦦꦊꦽꦨꦓꦄꦆꦎꦌꦈꦍꦇ꧅꧄ꦬꦋꦰꦜꦞꦙꦘꦖꦐꦅ")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(refs: dict, hyps: dict, names: list[str]) -> dict:
    rr, hh = [refs[n] for n in names], [hyps[n] for n in names]
    return {"lines": len(names), "cer": cer(rr, hh), "cer_no_space": cer(rr, [h.replace(" ", "") for h in hh]),
            "exact": sum(a == b for a, b in zip(rr, hh)) / len(names)}


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def checkpoint_meta(path: Path) -> tuple[int, dict]:
    """Langkah dan argumen training (vars(args) dari src.train) yang tersimpan di checkpoint."""
    if not path.exists():
        raise SystemExit(f"checkpoint {path} tidak ada")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["step"], ckpt["args"]


def run_options(args: dict) -> dict:
    """Opsi training yang menentukan jenis run lanjutan, dari vars(args) src.train di checkpoint.

    Opsi yang belum ada saat checkpoint dibuat dianggap mati (0/False): checkpoint fase6 tidak punya track_prob,
    track_max, rare_max_similarity, maupun rare_attach, dan checkpoint sebelum fase8 tidak punya standard_prob.
    """
    return {"track": args.get("track_prob") or 0, "track_max": args.get("track_max") or 0,
            "standard": args.get("standard_prob") or 0,
            "slant": args.get("slant_prob") or 0, "slant_max": args.get("slant_max") or 0,
            "fixes": bool(args.get("font_fixes")),
            "insert": args.get("rare_insert_prob") or 0, "opener": args.get("rare_opener_prob") or 0,
            "similarity": args.get("rare_max_similarity") or 0, "attach": bool(args.get("rare_attach"))}


def check_optional_run(spec: dict, step: int, args: dict) -> None:
    """Berhenti bila run lanjutan belum selesai atau argumen training-nya tidak cocok dengan jenis run di `spec`."""
    key, o = spec["key"], run_options(args)
    if args.get("run") != spec["run"]:
        raise SystemExit(f"{key}: last_snapshot.pt milik run {args.get('run')!r}, bukan {spec['run']!r}")
    if not args.get("steps") or step != args["steps"]:
        raise SystemExit(f"{key}: last_snapshot.pt di langkah {step}, padahal run dijadwalkan {args.get('steps')} "
                         "langkah (run belum selesai); ekspor dihentikan")
    if spec["track"] and not (o["track"] > 0 and o["track_max"] > 0):
        raise SystemExit(f"{key}: run jarak antar-aksara, tetapi track_prob = {o['track']}, "
                         f"track_max = {o['track_max']}")
    if not spec["track"] and o["track"] != 0:
        raise SystemExit(f"{key}: run tanpa jarak tambahan, tetapi track_prob = {o['track']}")
    # Ejaan baku (src/spelling.py): spesifikasi run sebelum fase8 tidak punya kuncinya = run tanpa ejaan baku.
    if spec.get("standard") and not o["standard"] > 0:
        raise SystemExit(f"{key}: run ejaan baku, tetapi standard_prob = {o['standard']}")
    if not spec.get("standard") and o["standard"] != 0:
        raise SystemExit(f"{key}: run tanpa ejaan baku, tetapi standard_prob = {o['standard']}")
    # Kemiringan italic dan perbaikan font (fase 9): spesifikasi run sebelumnya tanpa kuncinya = mati.
    if spec.get("slant") and not (o["slant"] > 0 and o["slant_max"] > 0):
        raise SystemExit(f"{key}: run miring, tetapi slant_prob = {o['slant']}, slant_max = {o['slant_max']}")
    if not spec.get("slant") and o["slant"] != 0:
        raise SystemExit(f"{key}: run tanpa miring, tetapi slant_prob = {o['slant']}")
    if bool(spec.get("fixes")) != o["fixes"]:
        raise SystemExit(f"{key}: font_fixes = {o['fixes']}, padahal run ini {'dengan' if spec.get('fixes') else 'tanpa'} "
                         "perbaikan font")
    if spec["rare"] and not o["insert"] > 0:
        raise SystemExit(f"{key}: run aksara langka, tetapi rare_insert_prob = {o['insert']}")
    if not spec["rare"] and (o["insert"] != 0 or o["opener"] != 0):
        kind = "run jarak antar-aksara" if spec["track"] else "run pembanding"
        raise SystemExit(f"{key}: {kind} harus tanpa aksara langka, tetapi rare_insert_prob = {o['insert']}, "
                         f"rare_opener_prob = {o['opener']}")
    # Saringan hanya berarti bila ada sisipan: pada run tanpa aksara langka kedua opsi itu tidak dipakai training.
    if spec["rare"] and spec["filtered"] and not (o["similarity"] > 0 and o["attach"]):
        raise SystemExit(f"{key}: run aksara langka tersaring, tetapi rare_max_similarity = {o['similarity']}, "
                         f"rare_attach = {o['attach']}")
    if spec["rare"] and not spec["filtered"] and (o["similarity"] != 0 or o["attach"]):
        raise SystemExit(f"{key}: run aksara langka tanpa saringan, tetapi rare_max_similarity = {o['similarity']}, "
                         f"rare_attach = {o['attach']}")


def decimal(x: float) -> str:
    """0.15 -> "0,15" (desimal seperti teks config lain)."""
    return f"{x:g}".replace(".", ",")


def optional_pipeline(spec: dict, step: int, args: dict, start: str) -> dict:
    """Entri pipeline run lanjutan; config memuat langkahnya, mis. crnn:fase6_rare@1500 (lanjutan fase5_fonts@1298)."""
    o = run_options(args)
    detail = spec["detail"].format(augment=args.get("augment"), track=decimal(o["track"]),
                                   track_max=decimal(o["track_max"]), insert=decimal(o["insert"]),
                                   opener=decimal(o["opener"]), similarity=decimal(o["similarity"]),
                                   standard=decimal(o["standard"]), slant=decimal(o["slant"]),
                                   slant_max=decimal(o["slant_max"]))
    head = f"crnn:{spec['run']}@{step}" + (f" (lanjutan {start})" if start else "")
    config = f"{head} · {detail}"
    if len(config) > MAX_CONFIG:
        raise SystemExit(f"{spec['key']}: config {len(config)} karakter, batas kolom di web {MAX_CONFIG}: {config}")
    return {"key": spec["key"], "label": spec["label"], "kind": "crnn", "status": "done", "config": config}


def insert_after(pipelines: list[dict], entries: list[dict], after: str = "crnn_fonts_beam") -> list[dict]:
    """Salinan `pipelines` dengan `entries` (urutannya dipertahankan) tepat setelah pipeline `after`."""
    at = next(i for i, p in enumerate(pipelines) if p["key"] == after) + 1
    return pipelines[:at] + entries + pipelines[at:]


def add_optional(pipelines: list[dict], checkpoints: dict[str, Path],
                 checkpoint_dir: Path = CHECKPOINT_DIR) -> tuple[list[dict], dict[str, Path], list[dict]]:
    """Run lanjutan yang last_snapshot.pt-nya ada: diperiksa, lalu ditambahkan ke SALINAN pipelines & checkpoints.

    Mengembalikan (pipelines, checkpoints, spesifikasi run yang ikut). Daftar global tidak diubah.
    """
    checkpoints, entries, included = dict(checkpoints), [], []
    for spec in OPTIONAL_CHECKPOINTS:
        path = checkpoint_dir / spec["run"] / "last_snapshot.pt"
        if not path.exists():
            continue
        step, args = checkpoint_meta(path)
        check_optional_run(spec, step, args)
        start = ""
        if args.get("init"):  # bobot awal run, mis. out/checkpoints/fase5_fonts/last_snapshot.pt -> fase5_fonts@1298
            init = ROOT / args["init"]
            start = f"{init.parent.name}@{checkpoint_meta(init)[0]}"
        entries.append(optional_pipeline(spec, step, args, start))
        checkpoints[spec["key"]] = path
        included.append(spec)
    return insert_after(pipelines, entries), checkpoints, included


def optional_official(included: list[dict], eval_dir: Path = EVAL_DIR) -> dict[str, tuple[float, str]]:
    """CER resmi 745 baris (src.evaluate) tiap run lanjutan yang ikut: kunci -> (cer, sumber). Laporannya WAJIB ada."""
    official = {}
    for spec in included:
        path = eval_dir / f"{spec['run']}_G3_full.json"
        if not path.exists():
            labels = LABELS.relative_to(ROOT).as_posix()
            raise SystemExit(f"{spec['key']}: laporan resmi {path} tidak ada (buat dengan src.evaluate "
                             f"out/checkpoints/{spec['run']}/last_snapshot.pt --real {labels} --lines 0 "
                             f"--name {spec['run']}_G3_full); ekspor dihentikan")
        official[spec["key"]] = (read_json(path)["cer"], str(path))
    return official


def check_official(metrics: list[dict], official: dict[str, tuple[float, str]], tol: float = TOL) -> None:
    """CER 745 baris hasil ekspor harus sama dengan laporan resmi; pipeline tanpa metrik 745 baris juga galat."""
    exported = {m["pipeline"]: m["cer"] for m in metrics if m["scope"] == "nusaaksara_745"}
    for key, (value, source) in official.items():
        if key not in exported:
            raise SystemExit(f"{key}: tidak ada CER 745 baris untuk dicocokkan dengan laporan {source}")
        if abs(exported[key] - value) > tol:
            raise SystemExit(f"{key}: CER {exported[key]:.6%} berbeda dari laporan {source} ({value:.6%})")


def lm_facts(path: Path) -> dict | None:
    """Order, baris latih, peluang buang spasi, dan peluang ejaan baku sebuah model bahasa dari nama berkasnya
    (src.charlm: o<order>_n<baris>_ds<p>[_std<p>].pkl); None bila namanya tidak berpola itu."""
    match = re.fullmatch(r"o(\d+)_n(\d+)_ds([\d.]+)(?:_std([\d.]+))?\.pkl", path.name)
    if not match:
        return None
    return {"order": int(match[1]), "lines": int(match[2]), "drop_space_prob": float(match[3]),
            "standard_prob": float(match[4] or 0)}


def official_beam(official_key: str, run: str, snapshot: Path, beam_dir: Path | None = None) -> dict | None:
    """Beam + LM atas checkpoint run resmi, bila laporan scripts/beam_eval.py-nya ada (BEAM_REPORTS: out/beam/<run>_o5.json
    dengan model bahasa lama, <run>_o5std.json dengan model bahasa berejaan baku).

    Bobot LM (alpha, beta) dibaca dari laporan itu: disetel di dev sintetis untuk checkpoint INI, bukan bobot fase5_fonts
    (ALPHA, BETA). Bila ada lebih dari satu laporan, dipakai yang CER beam-nya di dev terendah. Berhenti bila sebuah
    laporan bukan milik checkpoint, model bahasa, atau lebar beam yang dijanjikan namanya, lebih tua dari last_snapshot.pt
    run itu, atau laporan-laporannya disetel di dev yang berbeda. Run resmi fase5_fonts sudah punya crnn_fonts_beam:
    tidak ada yang ditambah.
    """
    if official_key == "crnn_fonts":
        return None
    found = []
    for suffix, lm in BEAM_REPORTS.items():
        path = (beam_dir or ROOT / "out/beam") / f"{run}_{suffix}.json"
        if not path.exists():
            continue
        report = read_json(path)
        problem = ""
        if Path(report.get("checkpoint", "")).parent.name != run:
            problem = f"checkpoint {report.get('checkpoint')!r} bukan milik run {run}"
        elif Path(report.get("lm", "")).name != lm.name or report.get("beam_width") != WIDTH:
            problem = f"model bahasa {report.get('lm')!r} atau lebar beam {report.get('beam_width')} bukan yang dipakai ekspor"
        elif path.stat().st_mtime < snapshot.stat().st_mtime:
            problem = f"lebih tua dari {snapshot.name} run itu"
        if problem:
            raise SystemExit(f"run resmi {run}: laporan beam {path} tidak sah ({problem}); ekspor dihentikan")
        found.append((path, lm, report))
    if not found:
        return None
    if len({json.dumps([report["dev"].get(key) for key in DEV_KEYS]) for _, _, report in found}) > 1:
        raise SystemExit(f"run resmi {run}: laporan beam {', '.join(path.name for path, _, _ in found)} disetel di dev yang "
                         "berbeda, jadi model bahasanya tidak bisa dipilih; ekspor dihentikan")
    path, lm, report = min(found, key=lambda item: item[2]["dev"]["chosen"]["cer"])  # seri: urutan BEAM_REPORTS
    alpha, beta = report["dev"]["chosen"]["alpha"], report["dev"]["chosen"]["beta"]
    step = checkpoint_meta(snapshot)[0]
    return {"key": f"{official_key}_beam", "base": official_key, "alpha": alpha, "beta": beta, "report": report, "path": path,
            "lm": lm, "candidates": [p.name for p, _, _ in found],
            "pipeline": {"key": f"{official_key}_beam", "label": f"Beam + LM {run}", "kind": "beam", "status": "done",
                         "config": f"crnn:{run}@{step} · beam {WIDTH} · {LM_LABELS[lm.name]} · α {alpha} · β {beta}"}}


def beam_decoders(extra_beam: dict | None) -> dict[str, tuple[str, float, float, Path]]:
    """Pipeline greedy yang logit-nya juga dibaca dengan beam + LM -> (kunci pipeline beam, alpha, beta, model bahasa).
    Bobot dan model bahasa milik checkpoint masing-masing: fase5_fonts memakai ALPHA/BETA dan LM_PATH, checkpoint resmi
    memakai laporannya sendiri."""
    decoders = {"crnn_fonts": ("crnn_fonts_beam", ALPHA, BETA, LM_PATH)}
    if extra_beam:
        decoders[extra_beam["base"]] = (extra_beam["key"], extra_beam["alpha"], extra_beam["beta"], extra_beam["lm"])
    return decoders


def beam_pairs(extra_beam: dict | None) -> list[tuple[str, str]]:
    """Pasangan (beam + LM, greedy) yang dibandingkan per baris; pasangan milik run resmi, bila ada, paling akhir."""
    return [(beam_key, base_key) for base_key, (beam_key, *_) in beam_decoders(extra_beam).items()]


def tag_pair(extra_beam: dict | None) -> tuple[str, str]:
    """Pasangan yang menentukan tag "beam membaik/memburuk": milik run resmi bila beam + LM-nya ikut diekspor (kalau
    tidak fase5_fonts). Web membaca pasangan yang sama lewat Pipeline::beamPair()."""
    return beam_pairs(extra_beam)[-1]


def beam_official(extra_beam: dict | None) -> dict[str, tuple[float, str]]:
    """CER 745 baris beam + LM run resmi menurut laporannya, untuk dicocokkan dengan hasil ekspor (check_official)."""
    if not extra_beam:
        return {}
    return {extra_beam["key"]: (extra_beam["report"]["real"]["beam_cer"], str(extra_beam["path"]))}


def beam_delta(ref: str, hyps: dict[str, dict[str, str]], name: str, pair: tuple[str, str]) -> float:
    """CER baris beam + LM dikurangi CER baris greedy untuk satu pasangan: negatif = beam membaik."""
    beam_key, base_key = pair
    return line_cer(ref, hyps[beam_key][name]) - line_cer(ref, hyps[base_key][name])


def mark_beam_changes(metrics: list[dict], refs: dict, hyps: dict, names: list[str], pairs: list[tuple[str, str]]) -> None:
    """Jumlah baris membaik dan memburuk tiap pasangan, ditulis ke metrik 745 baris pipeline beam-nya."""
    for pair in pairs:
        deltas = [beam_delta(refs[n], hyps, n, pair) for n in names]
        for m in metrics:
            if m["scope"] == "nusaaksara_745" and m["pipeline"] == pair[0]:
                m["better"], m["worse"] = sum(d < 0 for d in deltas), sum(d > 0 for d in deltas)


def synth_metrics(key: str, base: str, report: dict) -> list[dict]:
    """Metrik sintetis (dev dan font di luar data latih) sebuah pasangan dari laporan scripts/beam_eval.py-nya."""
    rows = []
    for scope, block in [("synth_dev", report["dev"]), ("synth_heldout", report["clean_heldout_font"])]:
        rows.append({"scope": scope, "pipeline": base, "lines": block["lines"], "cer": block["greedy_cer"]})
        rows.append({"scope": scope, "pipeline": key, "lines": block["lines"], "cer": block["beam_cer"]})
    return rows


def official_pipeline(checkpoints: dict[str, Path], run: str = OFFICIAL_RUN) -> str:
    """Kunci pipeline yang membaca checkpoint run resmi; berhenti bila checkpoint itu tidak ikut diekspor."""
    keys = [key for key, path in checkpoints.items() if path.parent.name == run]
    if len(keys) != 1:
        raise SystemExit(f"run resmi {run!r} tidak ikut diekspor (out/checkpoints/{run}/last_snapshot.pt belum ada?)")
    return keys[0]


def gate_problem(code: str, report: dict, run: str, real_lines: int) -> str:
    """Alasan laporan src.evaluate tidak sah sebagai gerbang `code` untuk `run`; "" bila sah.

    G1/G2: OFFICIAL_LINES baris split test, hanya font javatext, bersih (G1) atau preset heavy (G2).
    G3: semua baris nyata, tanpa margin (`--pad-ratio` bukan jalur resmi).
    """
    if report.get("goal") != code:
        return f"goal {report.get('goal')!r}, bukan {code}"
    if Path(report.get("checkpoint", "")).parent.name != run:
        return f"checkpoint {report.get('checkpoint')!r} bukan milik run {run}"
    total = report["results"]["semua"]["lines"]
    if code == "G3":
        if report.get("pad_ratio", 0.0) != 0.0:
            return f"pad_ratio {report.get('pad_ratio')}, gerbang G3 tanpa margin"
        return "" if total == real_lines else f"{total} baris, bukan seluruh {real_lines} baris nyata"
    augment = "none" if code == "G1" else "heavy"
    if report.get("split") != "test" or report.get("augment") != augment:
        return f"split {report.get('split')!r} dan augmentasi {report.get('augment')!r}, bukan 'test' dan {augment!r}"
    fonts = [name for name in report["results"] if name != "semua"]
    if fonts != ["javatext.ttf"]:
        return f"font {fonts}, bukan javatext.ttf saja"
    total += report["results"]["javatext.ttf"].get("rejected_by_min_frames", 0)  # baris ditolak T >= 1,5L tetap dihitung
    return "" if total == OFFICIAL_LINES else f"{total} baris, bukan {OFFICIAL_LINES}"


def official_gates(run: str, label: str, snapshot: Path, real_lines: int, eval_dir: Path = EVAL_DIR) -> list[dict]:
    """Gerbang G1-G3 dari laporan resmi src.evaluate milik run resmi, plus G4 (round-trip tokenizer, Fase 1).

    Berhenti bila sebuah laporan tidak ada, tidak sah (`gate_problem`), atau lebih tua dari last_snapshot.pt run itu.
    """
    prefix = LEGACY_REPORTS.get(run, run)
    lines = f"{OFFICIAL_LINES:,}".replace(",", ".")
    specs = [
        ("G1", f"{prefix}_G1_10k", "Sintetis bersih, font uji",
         f"{lines} baris · javatext (sekeluarga dengan font training CarakanJawa) · {label}"),
        ("G2", f"{prefix}_G2_10k", "Sintetis augmentasi berat", f"{lines} baris · javatext · preset heavy · {label}"),
        ("G3", f"{prefix}_G3_full", "Baris foto/pindaian nyata",
         f"{real_lines} baris NusaAksara · {label} · greedy (jalur resmi)"),
    ]
    gates = []
    for code, name, title, basis in specs:
        path = eval_dir / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"run resmi {run}: laporan {code} {path} tidak ada (G1/G2: scripts/make_official.sh {run}); "
                             "ekspor dihentikan")
        report = read_json(path)
        problem = gate_problem(code, report, run, real_lines)
        if not problem and path.stat().st_mtime < snapshot.stat().st_mtime:
            problem = f"lebih tua dari {snapshot.name} run itu"
        if problem:
            raise SystemExit(f"run resmi {run}: laporan {code} {path} tidak sah ({problem}); ekspor dihentikan")
        gates.append({"code": code, "name": title, "value": report["cer"], "target": TARGETS[code],
                      "passed": report["cer"] < TARGETS[code], "basis": basis, "source": f"out/eval/{name}.json"})
    gates.append({"code": "G4", "name": "Round-trip tokenizer", "value": 1.0, "target": 1.0, "passed": True,
                  "basis": "1.034.357 / 1.034.357 baris korpus (Fase 1)", "source": "tests/test_tokenizer.py"})
    return gates


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "out/results"))
    parser.add_argument("--limit", type=int, default=0, help="hanya N baris pertama (uji cepat; lewati cek angka resmi)")
    args = parser.parse_args(argv)
    started = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipelines, checkpoints, included = add_optional(PIPELINES, CHECKPOINTS)
    optional_keys = {spec["key"] for spec in included}
    for p in pipelines:
        if p["key"] in optional_keys:
            print(f"run lanjutan ikut: {p['key']} = {p['config']}", flush=True)
    skipped = [spec["run"] for spec in OPTIONAL_CHECKPOINTS if spec["key"] not in optional_keys]
    if skipped:  # snapshot baru dibuat setelah run selesai (`evaluate` di scripts/fase6_common.sh)
        print(f"run lanjutan dilewati (last_snapshot.pt belum ada): {', '.join(skipped)}", flush=True)
    official_key = official_pipeline(checkpoints, OFFICIAL_RUN)
    official_label = next(p["label"] for p in pipelines if p["key"] == official_key)
    print(f"angka resmi: {official_key} (run {OFFICIAL_RUN})", flush=True)
    extra_beam = official_beam(official_key, OFFICIAL_RUN, checkpoints[official_key])
    if extra_beam:
        pipelines = insert_after(pipelines, [extra_beam["pipeline"]], official_key)
        print(f"beam + LM run resmi ikut: {extra_beam['pipeline']['config']}", flush=True)
    official: dict[str, tuple[float, str]] = {}
    if not args.limit:  # laporan resmi dibaca SEBELUM inferensi panjang supaya laporan yang hilang langsung ketahuan
        fonts_json, core_json = EVAL_DIR / "fonts_G3_full.json", EVAL_DIR / "core_G3_full.json"
        beam_json = ROOT / "out/beam/fonts_o5.json"
        official = {"crnn_fonts": (read_json(fonts_json)["cer"], str(fonts_json)),
                    "crnn_core": (read_json(core_json)["cer"], str(core_json)),
                    "crnn_fonts_beam": (read_json(beam_json)["real"]["beam_cer"], str(beam_json)),
                    **optional_official(included), **beam_official(extra_beam)}

    with LABELS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    # Sebelum inferensi dan sebelum --limit memotong baris: gerbang G3 = seluruh baris nyata.
    gates = official_gates(OFFICIAL_RUN, official_label, checkpoints[official_key], len(rows))
    vlm_path = ROOT / "out/eval/vlm_blind_50.json"
    vlm_items = read_json(vlm_path)["items"] if vlm_path.exists() else []
    vlm = {Path(it["image_path"]).name: it for it in vlm_items}
    if args.limit:
        keep = [r for r in rows if Path(r["image_path"]).name in vlm][: args.limit // 2]
        keep += [r for r in rows if r not in keep][: args.limit - len(keep)]
        rows = keep
    names = [Path(r["image_path"]).name for r in rows]
    refs = {Path(r["image_path"]).name: nfc(r["text"]) for r in rows}
    blind = [n for n in vlm if n in refs]

    decoders = beam_decoders(extra_beam)
    lms = {path: CharLM.load(path) for path in {spec[3] for spec in decoders.values()}}
    hyps: dict[str, dict[str, str]] = {p["key"]: {} for p in pipelines if p["status"] == "done"}
    spans: dict[str, dict[str, list | None]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for key, path in checkpoints.items():
        model, tok = load_checkpoint(str(path), "cpu")
        spans[key] = {}
        for r in rows:
            n = Path(r["image_path"]).name
            img = Image.open(LABELS.parent / r["image_path"])
            sizes[n] = img.size
            lp = line_log_probs(img, model, "cpu", 0.0)
            hyps[key][n] = tok.to_logical(tok.decode(greedy_path(lp)[0]))
            spans[key][n] = syllable_spans(lp, tok, img.width, img.height)
            if key in decoders:
                beam_key, alpha, beta, lm_path = decoders[key]
                beam_ids = prefix_beam_search(lp, tok.charset, lms[lm_path], alpha, beta, WIDTH)
                hyps[beam_key][n] = tok.to_logical(tok.decode(beam_ids))
        print(f"{key}: {len(rows)} baris ({time.time() - started:.0f} dtk)", flush=True)
    hyps["vlm_zeroshot"] = {n: nfc(vlm[n]["transcription"]) for n in blind}

    metrics = []
    for key, hh in hyps.items():
        covered = [n for n in names if n in hh]
        if len(covered) == len(names):
            metrics.append({"scope": "nusaaksara_745", "pipeline": key, **aggregate(refs, hh, names)})
        if blind:
            metrics.append({"scope": "blind_50", "pipeline": key, **aggregate(refs, hh, blind)})
    mark_beam_changes(metrics, refs, hyps, names, beam_pairs(extra_beam))

    if not args.limit:  # angka 745 baris harus sama dengan laporan resmi (termasuk run lanjutan yang ikut)
        check_official(metrics, official)
        print(f"cek angka resmi: sama ({', '.join(official)})", flush=True)

    metrics += synth_metrics("crnn_fonts_beam", "crnn_fonts", read_json(ROOT / "out/beam/fonts_o5.json"))
    if extra_beam:
        metrics += synth_metrics(extra_beam["key"], extra_beam["base"], extra_beam["report"])

    confusion = Counter()
    for n in names:
        r, h = refs[n].replace(" ", ""), hyps[official_key][n].replace(" ", "")
        for op in Levenshtein.editops(r, h):
            confusion[(op.tag, r[op.src_pos] if op.tag != "insert" else "", h[op.dest_pos] if op.tag != "delete" else "")] += 1
    kinds = {"replace": "sub", "delete": "del", "insert": "ins"}

    manifest = {
        "schema": SCHEMA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "metopenv5",
        "limited": bool(args.limit),
        "pipelines": [{**p, "sort": i} for i, p in enumerate(pipelines)],
        "official": official_key,
        "gates": gates,
        "metrics": metrics,
        "ablation": read_json(ROOT / "out/ablation.json"),
        "confusion": {
            "pipeline": official_key, "scope": "nusaaksara_745",
            "totals": {kinds[k]: sum(c for (op, _, _), c in confusion.items() if op == k) for k in kinds},
            "items": [{"kind": kinds[op], "ref": a, "hyp": b, "count": c} for (op, a, b), c in confusion.most_common()],
        },
    }

    lines_out, preds_out = [], []
    for r in rows:
        n = Path(r["image_path"]).name
        ref = refs[n]
        tags = []
        if ref.startswith("꧋"):
            tags.append("adeg-adeg")
        if any(ch in CLASSIC for ch in ref):
            tags.append("aksara klasik")
        delta = beam_delta(ref, hyps, n, tag_pair(extra_beam))
        if delta < 0:
            tags.append("beam membaik")
        elif delta > 0:
            tags.append("beam memburuk")
        if " " not in ref:
            tags.append("label tanpa spasi")
        if n in vlm:
            tags.append("uji buta")
        w, h = sizes[n]
        lines_out.append({"dataset": "nusaaksara", "id": n, "image": f"data/real/nusaaksara/{r['image_path']}",
                          "width": w, "height": h, "reference": ref, "condition": r.get("condition", ""),
                          "source_id": r.get("source_id", ""), "tags": tags})
        for key, hh in hyps.items():
            if n not in hh:
                continue
            item = {"line": n, "pipeline": key, "text": hh[n], "cer": line_cer(ref, hh[n]),
                    "segments": syllable_diff(ref, hh[n]), "spans": spans.get(key, {}).get(n)}
            if key == "vlm_zeroshot":
                item["confidence"] = vlm[n].get("confidence")
            preds_out.append(item)

    write_atomic(out_dir / "lines.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines_out))
    write_atomic(out_dir / "predictions.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preds_out))
    write_atomic(out_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"{len(lines_out)} baris, {len(preds_out)} prediksi, {len(metrics)} metrik -> {out_dir} "
          f"({time.time() - started:.0f} dtk)", flush=True)


if __name__ == "__main__":
    main()
