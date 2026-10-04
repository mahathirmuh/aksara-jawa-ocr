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
ALPHA, BETA, WIDTH = 0.25, 1.0, 16
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
]
CHECKPOINT_DIR = ROOT / "out/checkpoints"
EVAL_DIR = ROOT / "out/eval"
# Run resmi: gerbang G1-G3 di manifest dan tabel kesalahan aksara dihitung darinya, dan web menandai pipelinenya
# "angka resmi" (kunci "official" di manifest). Nama run = direktori checkpoint dan awalan laporan resminya di
# out/eval/ (LEGACY_REPORTS untuk run lama). Mengganti run resmi: buat dulu <run>_G1_10k.json dan <run>_G2_10k.json
# (10.000 baris javatext; scripts/make_official.sh) di samping <run>_G3_full.json, lalu ubah konstanta ini dan
# DEFAULT_CHECKPOINT di src/serve.py (Demo; tests/test_export_results.py menjaga keduanya sama).
# Riwayat: fase5_fonts sampai 2026-10-04, lalu fase7_track (keputusan user).
OFFICIAL_RUN = "fase7_track"
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
    track_max, rare_max_similarity, maupun rare_attach.
    """
    return {"track": args.get("track_prob") or 0, "track_max": args.get("track_max") or 0,
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
                                   opener=decimal(o["opener"]), similarity=decimal(o["similarity"]))
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
    official: dict[str, tuple[float, str]] = {}
    if not args.limit:  # laporan resmi dibaca SEBELUM inferensi panjang supaya laporan yang hilang langsung ketahuan
        fonts_json, core_json = EVAL_DIR / "fonts_G3_full.json", EVAL_DIR / "core_G3_full.json"
        beam_json = ROOT / "out/beam/fonts_o5.json"
        official = {"crnn_fonts": (read_json(fonts_json)["cer"], str(fonts_json)),
                    "crnn_core": (read_json(core_json)["cer"], str(core_json)),
                    "crnn_fonts_beam": (read_json(beam_json)["real"]["beam_cer"], str(beam_json)),
                    **optional_official(included)}

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

    lm = CharLM.load(LM_PATH)
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
            if key == "crnn_fonts":
                beam_ids = prefix_beam_search(lp, tok.charset, lm, ALPHA, BETA, WIDTH)
                hyps["crnn_fonts_beam"][n] = tok.to_logical(tok.decode(beam_ids))
        print(f"{key}: {len(rows)} baris ({time.time() - started:.0f} dtk)", flush=True)
    hyps["vlm_zeroshot"] = {n: nfc(vlm[n]["transcription"]) for n in blind}

    metrics = []
    for key, hh in hyps.items():
        covered = [n for n in names if n in hh]
        if len(covered) == len(names):
            metrics.append({"scope": "nusaaksara_745", "pipeline": key, **aggregate(refs, hh, names)})
        if blind:
            metrics.append({"scope": "blind_50", "pipeline": key, **aggregate(refs, hh, blind)})
    beam_cmp = [line_cer(refs[n], hyps["crnn_fonts_beam"][n]) - line_cer(refs[n], hyps["crnn_fonts"][n]) for n in names]
    for m in metrics:
        if m["scope"] == "nusaaksara_745" and m["pipeline"] == "crnn_fonts_beam":
            m["better"], m["worse"] = sum(d < 0 for d in beam_cmp), sum(d > 0 for d in beam_cmp)

    if not args.limit:  # angka 745 baris harus sama dengan laporan resmi (termasuk run lanjutan yang ikut)
        check_official(metrics, official)
        print(f"cek angka resmi: sama ({', '.join(official)})", flush=True)

    beam = read_json(ROOT / "out/beam/fonts_o5.json")
    for scope, block in [("synth_dev", beam["dev"]), ("synth_heldout", beam["clean_heldout_font"])]:
        metrics.append({"scope": scope, "pipeline": "crnn_fonts", "lines": block["lines"], "cer": block["greedy_cer"]})
        metrics.append({"scope": scope, "pipeline": "crnn_fonts_beam", "lines": block["lines"], "cer": block["beam_cer"]})

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
        delta = line_cer(ref, hyps["crnn_fonts_beam"][n]) - line_cer(ref, hyps["crnn_fonts"][n])
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
