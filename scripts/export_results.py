"""Ekspor hasil eksperimen untuk web `aksara-ocr-web` (kontrak data, skema 1).

  .venv/Scripts/python scripts/export_results.py            # ~14 menit di CPU (5 checkpoint + beam, 745 baris)

Tanpa run fase6 (3 checkpoint + beam) ~9 menit; tiap run fase6 yang ikut menambah satu lintasan inferensi 745 baris.

Keluaran di out/results/ (ditimpa):
  manifest.json      skema, waktu, pipeline, gerbang G1-G4, metrik agregat, ablasi, kesalahan aksara
  lines.jsonl        satu baris data nyata per baris: id, path citra, ukuran, label, tag
  predictions.jsonl  satu prediksi per (baris, pipeline): teks, CER, segmen beda per suku kata, kolom citra

Semua angka dihitung di sini; web hanya menampilkan. Angka 745 baris dicocokkan dengan laporan resmi
(out/eval/*_G3_full.json, termasuk run fase6 yang ikut, dan out/beam/fonts_o5.json) dan skrip berhenti kalau
berbeda atau laporannya tidak ada. Run fase6 ikut bila last_snapshot.pt-nya ada; snapshot dari run yang belum
selesai (langkah != args.steps) atau yang flag aksara langkanya tidak cocok dengan nama run menghentikan skrip.
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
# fase5_fonts), disisipkan dengan urutan daftar ini tepat setelah crnn_fonts_beam (`add_optional`). Run yang ikut
# wajib selesai dan flag aksara langkanya cocok dengan `rare`; tanpa --limit, CER 745 barisnya wajib sama dengan
# out/eval/<run>_G3_full.json. Langkah, titik awal, augmentasi, dan peluang aksara langka di config dibaca dari
# checkpoint.
OPTIONAL_CHECKPOINTS = [
    {"key": "crnn_fase6_rare", "run": "fase6_rare", "label": "CRNN fase6_rare", "rare": True,
     "detail": "10 font · aug {augment} · + aksara langka (sisip {insert} · adeg-adeg {opener}) · greedy"},
    {"key": "crnn_fase6_ctrl", "run": "fase6_ctrl", "label": "CRNN fase6_ctrl", "rare": False,
     "detail": "10 font · aug {augment} · tanpa aksara langka (pembanding) · greedy"},
]
CHECKPOINT_DIR = ROOT / "out/checkpoints"
EVAL_DIR = ROOT / "out/eval"
TOL = 1e-9  # selisih CER terhadap laporan resmi yang masih dianggap sama

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


def check_optional_run(spec: dict, step: int, args: dict) -> None:
    """Berhenti bila run lanjutan belum selesai atau flag aksara langkanya tidak cocok dengan nama run."""
    key = spec["key"]
    insert, opener = args.get("rare_insert_prob") or 0, args.get("rare_opener_prob") or 0
    if args.get("run") != spec["run"]:
        raise SystemExit(f"{key}: last_snapshot.pt milik run {args.get('run')!r}, bukan {spec['run']!r}")
    if not args.get("steps") or step != args["steps"]:
        raise SystemExit(f"{key}: last_snapshot.pt di langkah {step}, padahal run dijadwalkan {args.get('steps')} "
                         "langkah (run belum selesai); ekspor dihentikan")
    if spec["rare"] and not insert > 0:
        raise SystemExit(f"{key}: run aksara langka, tetapi rare_insert_prob = {insert}")
    if not spec["rare"] and (insert != 0 or opener != 0):
        raise SystemExit(f"{key}: run pembanding harus tanpa aksara langka, tetapi rare_insert_prob = {insert}, "
                         f"rare_opener_prob = {opener}")


def decimal(x: float) -> str:
    """0.15 -> "0,15" (desimal seperti teks config lain)."""
    return f"{x:g}".replace(".", ",")


def optional_pipeline(spec: dict, step: int, args: dict, start: str) -> dict:
    """Entri pipeline run lanjutan; config memuat langkahnya, mis. crnn:fase6_rare@1500 (lanjutan fase5_fonts@1298)."""
    detail = spec["detail"].format(augment=args.get("augment"), insert=decimal(args.get("rare_insert_prob") or 0),
                                   opener=decimal(args.get("rare_opener_prob") or 0))
    head = f"crnn:{spec['run']}@{step}" + (f" (lanjutan {start})" if start else "")
    return {"key": spec["key"], "label": spec["label"], "kind": "crnn", "status": "done",
            "config": f"{head} · {detail}"}


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
        r, h = refs[n].replace(" ", ""), hyps["crnn_fonts"][n].replace(" ", "")
        for op in Levenshtein.editops(r, h):
            confusion[(op.tag, r[op.src_pos] if op.tag != "insert" else "", h[op.dest_pos] if op.tag != "delete" else "")] += 1
    kinds = {"replace": "sub", "delete": "del", "insert": "ins"}

    evals = {k: read_json(ROOT / f"out/eval/{k}.json") for k in ["fonts_G1_10k", "fonts_G2_10k", "fonts_G3_full"]}
    gates = [
        {"code": "G1", "name": "Sintetis bersih, font baru", "value": evals["fonts_G1_10k"]["cer"], "target": 0.02,
         "passed": evals["fonts_G1_10k"]["passed"], "basis": "10.000 baris · font held-out javatext", "source": "out/eval/fonts_G1_10k.json"},
        {"code": "G2", "name": "Sintetis augmentasi berat", "value": evals["fonts_G2_10k"]["cer"], "target": 0.05,
         "passed": evals["fonts_G2_10k"]["passed"], "basis": "10.000 baris · preset heavy", "source": "out/eval/fonts_G2_10k.json"},
        {"code": "G3", "name": "Baris foto/pindaian nyata", "value": evals["fonts_G3_full"]["cer"], "target": 0.08,
         "passed": evals["fonts_G3_full"]["passed"], "basis": "745 baris NusaAksara · greedy (jalur resmi)", "source": "out/eval/fonts_G3_full.json"},
        {"code": "G4", "name": "Round-trip tokenizer", "value": 1.0, "target": 1.0, "passed": True,
         "basis": "1.034.357 / 1.034.357 baris korpus (Fase 1)", "source": "tests/test_tokenizer.py"},
    ]

    manifest = {
        "schema": SCHEMA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "metopenv5",
        "limited": bool(args.limit),
        "pipelines": [{**p, "sort": i} for i, p in enumerate(pipelines)],
        "gates": gates,
        "metrics": metrics,
        "ablation": read_json(ROOT / "out/ablation.json"),
        "confusion": {
            "pipeline": "crnn_fonts", "scope": "nusaaksara_745",
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
