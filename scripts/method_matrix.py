"""Matriks perbandingan metode OCR aksara Jawa pada baris nyata NusaAksara (data uji, sekali baca per metode).

Semua metode dinilai dengan aturan yang sama: keluaran dinormalkan NFC dan baris baru / tab di ujungnya dibuang
(spasi tetap dihitung, sama dengan evaluasi resmi G3), lalu CER (jarak Levenshtein per karakter) dihitung terhadap label
pada urutan logis. Tidak ada pembersihan lain, jadi keluaran yang memuat huruf Latin atau penjelasan ikut dihitung salah. Ukuran utama = 308 baris cetak (scope repo); 745 baris
(G3) dan tulisan tangan dilaporkan di sampingnya bila metodenya membaca semua baris.
CER tidak berbatas atas (sisipan tidak dibatasi; keluaran kosong = 100%), jadi matriks juga melaporkan NED (normalized
edit distance, ICDAR 2019): jumlah suntingan dibagi jumlah panjang terpanjang label/keluaran per baris, selalu 0-100%.

  python scripts/method_matrix.py tesseract [--lang jav]       Tesseract 5 siap pakai (tidak punya model aksara Jawa)
  python scripts/method_matrix.py crnn CKPT --name NAME         model kita (CRNN / Transformer), greedy, CPU
  python scripts/method_matrix.py gpt --model M [--subset cetak|semua] [--workers 8]   VLM GPT zero-shot (BERBAYAR)
  python scripts/method_matrix.py ingest BERKAS --name NAME --label L --group G        keluaran pembaca lain (mis. Claude)
  python scripts/method_matrix.py latency                       waktu baca per baris di CPU (50 baris cetak)
  python scripts/method_matrix.py decode CKPT --name N --label L --alpha A --beta B [--lm LM]   cara membaca lain (ablasi)
  python scripts/method_matrix.py score                         -> out/compare/method_matrix.json dan .md
  python scripts/method_matrix.py ablation                      -> out/compare/ablation_matrix.json dan .md
  python scripts/method_matrix.py export                        -> out/results/matrix.json (php artisan aksara:matrix)
  python scripts/method_matrix.py prompt                        prompt VLM (satu untuk semua VLM)

Keluaran tiap metode: out/methods/<name>.jsonl ({"line", "text", ...}) dan <name>.meta.json. Prompt VLM ditetapkan
2026-10-18 sebelum satu pun VLM membaca baris uji, dan sidiknya dicatat. Teks label tidak ditulis ke berkas skor.
"""

import argparse
import base64
import hashlib
import json
import random
import subprocess
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_slant import KINDS, kind  # noqa: E402

RESULTS = ROOT / "out/results"
METHODS = ROOT / "out/methods"
OUT = ROOT / "out/compare"
TESSERACT = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
REFERENCE = "crnn_fase9_ctrl"  # pembanding selisih = model resmi, greedy

PROMPT = (
    "The image contains one line of text in Javanese script (Aksara Jawa / Hanacaraka), cropped from a printed book "
    "or a manuscript. Transcribe it exactly as written into Unicode Javanese script (block U+A980-U+A9DF), using "
    "standard Unicode logical order (for example taling U+A9BA is stored after its base consonant even though it is "
    "drawn to the left of it). Do not translate, do not romanize, do not correct the spelling, and do not add text "
    "that is not visible. Output only the Javanese-script text, with no explanation."
)
PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:12]

# Metode yang prediksinya sudah ada di ekspor hasil (out/results/predictions.jsonl).
EXPORTED = {
    "crnn_fase9_ctrl": ("CRNN + CTC, greedy (model resmi)", "kita"),
    "crnn_fase9_ctrl_beam": ("CRNN + CTC + beam + LM karakter", "kita"),
    "vlm_zeroshot": ("Claude, zero-shot (uji buta 2026-09-14, prompt lain)", "vlm"),
}


def load_lines() -> list[dict]:
    rows = [json.loads(x) for x in (RESULTS / "lines.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        r["kind"] = kind(r["source_id"])
    return rows


def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").strip("\r\n\t")


def write_method(name: str, rows: list[dict], meta: dict) -> None:
    METHODS.mkdir(parents=True, exist_ok=True)
    with (METHODS / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (METHODS / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ditulis: {METHODS / name}.jsonl ({len(rows)} baris)")


def subset(lines: list[dict], name: str) -> list[dict]:
    if name == "semua":
        return lines
    if name == "cetak":
        return [r for r in lines if r["kind"] == "cetak"]
    raise SystemExit(f"subset tidak dikenal: {name}")


# --- pembaca -----------------------------------------------------------------------------------------------------

def run_tesseract(args) -> None:
    lines = load_lines()
    tessdata = ROOT / "out/tessdata"
    if not (tessdata / f"{args.lang}.traineddata").exists():
        raise SystemExit(f"{tessdata / args.lang}.traineddata tidak ada (unduh dari github.com/tesseract-ocr/tessdata_best)")

    def one(line):
        start = time.perf_counter()
        out = subprocess.run([str(TESSERACT), str(ROOT / line["image"]), "stdout", "--tessdata-dir", str(tessdata),
                              "-l", args.lang, "--psm", "7"], capture_output=True, timeout=120)
        return {"line": line["id"], "text": norm(out.stdout.decode("utf-8", "replace")),
                "seconds": round(time.perf_counter() - start, 3)}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, lines))
    version = subprocess.run([str(TESSERACT), "--version"], capture_output=True).stdout.decode().splitlines()[0]
    write_method(f"tesseract_{args.lang}", rows, {
        "label": f"Tesseract 5 siap pakai (model '{args.lang}')", "group": "umum", "tool": version,
        "note": "Tesseract tidak punya model aksara Jawa; 'jav' = bahasa Jawa beraksara Latin. psm 7 (satu baris).",
        "workers": args.workers})


def run_crnn(args) -> None:
    import torch
    from PIL import Image

    from src.infer import load_checkpoint, predict

    torch.set_num_threads(args.threads)
    model, tokenizer = load_checkpoint(args.checkpoint)
    rows = []
    for line in load_lines():
        start = time.perf_counter()
        text = predict(Image.open(ROOT / line["image"]), model, tokenizer)
        rows.append({"line": line["id"], "text": norm(text), "seconds": round(time.perf_counter() - start, 4)})
    config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_config"]
    write_method(args.name, rows, {
        "label": args.label or args.name, "group": args.group, "checkpoint": args.checkpoint, "model_config": config,
        "params": sum(p.numel() for p in model.parameters()), "threads": args.threads})


def run_decode(args) -> None:
    """Cara membaca lain pada checkpoint yang sama (ablasi): beam search dengan / tanpa model bahasa karakter."""
    import torch
    from PIL import Image

    from src.beam import line_log_probs, prefix_beam_search
    from src.charlm import CharLM
    from src.infer import load_checkpoint

    torch.set_num_threads(args.threads)
    model, tokenizer = load_checkpoint(args.checkpoint)
    lm = CharLM.load(ROOT / args.lm)
    rows = []
    for line in load_lines():
        start = time.perf_counter()
        lp = line_log_probs(Image.open(ROOT / line["image"]), model, "cpu")
        ids = prefix_beam_search(lp, tokenizer.charset, lm, args.alpha, args.beta, args.width)
        rows.append({"line": line["id"], "text": tokenizer.to_logical(tokenizer.decode(ids)),
                     "seconds": round(time.perf_counter() - start, 4)})
    write_method(args.name, rows, {
        "label": args.label, "group": "ablasi-baca", "checkpoint": args.checkpoint, "lm": args.lm,
        "alpha": args.alpha, "beta": args.beta, "width": args.width, "threads": args.threads})


def image_input(path: Path) -> list[dict]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return [{"role": "user", "content": [{"type": "input_text", "text": PROMPT},
                                         {"type": "input_image", "image_url": f"data:image/png;base64,{data}"}]}]


def run_gpt(args) -> None:
    from src.gpt import GPTClient

    client = GPTClient()
    lines = subset(load_lines(), args.subset)

    def one(line):
        for attempt in range(6):
            start = time.perf_counter()
            try:
                res = client.generate(args.model, image_input(ROOT / line["image"]))
            except Exception as e:  # batas laju / galat jaringan: coba lagi dengan jeda
                if attempt == 5:
                    return {"line": line["id"], "text": "", "error": f"{type(e).__name__}: {str(e)[:200]}"}
                time.sleep(5 * 2 ** attempt)
                continue
            usage = res.usage or {}
            return {"line": line["id"], "text": norm(res.text), "raw": res.text, "model": res.model,
                    "cached": res.cached, "seconds": round(time.perf_counter() - start, 2),
                    "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}

    rows, done = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, line) for line in lines]
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(lines):
                print(f"{done}/{len(lines)}", flush=True)
    order = {line["id"]: i for i, line in enumerate(lines)}
    rows.sort(key=lambda r: order[r["line"]])
    errors = sum("error" in r for r in rows)
    served = sorted({r.get("model") for r in rows if r.get("model")})
    write_method(f"gpt_{args.model}", rows, {
        "label": f"{args.model}, zero-shot", "group": "vlm", "model_requested": args.model, "model_served": served,
        "prompt_sha": PROMPT_SHA, "subset": args.subset, "errors": errors,
        "input_tokens": sum(r.get("input_tokens") or 0 for r in rows),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in rows)})
    if errors:
        print(f"PERINGATAN: {errors} baris gagal dibaca (teks kosong, dihitung salah semua)")


def run_ingest(args) -> None:
    rows = [json.loads(x) for x in Path(args.file).read_text(encoding="utf-8").splitlines() if x.strip()]
    known = {line["id"] for line in load_lines()}
    bad = [r.get("line") for r in rows if r.get("line") not in known]
    if bad:
        raise SystemExit(f"{len(bad)} baris tidak dikenal, mis. {bad[:3]}")
    seen = [r["line"] for r in rows]
    if len(seen) != len(set(seen)):
        raise SystemExit("ada baris ganda")
    out = [{"line": r["line"], "text": norm(r.get("text", ""))} for r in rows]
    write_method(args.name, out, {"label": args.label, "group": args.group, "prompt_sha": args.prompt_sha or None,
                                  "model": args.model or None, "source": str(args.file)})


def run_latency(args) -> None:
    """Waktu baca per baris di CPU untuk pembaca lokal, pada 50 baris cetak pertama (sesudah 3 baris pemanasan)."""
    import torch
    from PIL import Image

    from src import serve

    torch.set_num_threads(args.threads)
    lines = [r for r in load_lines() if r["kind"] == "cetak"][:53]
    images = [Image.open(ROOT / r["image"]).convert("L") for r in lines]
    report = {"threads": args.threads, "lines": 50}
    for pipeline in serve.PIPELINES:
        times = []
        for i, img in enumerate(images):
            start = time.perf_counter()
            serve.predict_image(img, pipeline)
            if i >= 3:
                times.append(time.perf_counter() - start)
        report[pipeline] = round(sum(times) / len(times), 4)
        print(f"{pipeline}: {report[pipeline] * 1000:.0f} ms/baris")
    (METHODS / "latency.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


# --- penilaian ---------------------------------------------------------------------------------------------------

def javanese_only(text: str) -> bool:
    return all(0xA980 <= ord(c) <= 0xA9DF or c == " " for c in text)


def methods() -> dict[str, dict]:
    found: dict[str, dict] = {}
    exported: dict[str, dict] = {}
    for x in (RESULTS / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            if r["pipeline"] in EXPORTED:
                exported.setdefault(r["pipeline"], {})[r["line"]] = r["text"]
    for key, texts in exported.items():
        label, group = EXPORTED[key]
        found[key] = {"label": label, "group": group, "texts": texts, "meta": {"source": "out/results/predictions.jsonl"}}
    for path in sorted(METHODS.glob("*.jsonl")):
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        found[path.stem] = {"label": meta["label"], "group": meta["group"], "meta": meta,
                            "texts": {r["line"]: r.get("raw", r["text"]) for r in rows},
                            "seconds": [r["seconds"] for r in rows if r.get("seconds") is not None]}
    return found


def stats(rows: list[dict]) -> dict:
    chars = sum(r["chars"] for r in rows)
    return {"lines": len(rows), "pages": len({r["page"] for r in rows}), "chars": chars,
            "cer": sum(r["edits"] for r in rows) / chars,
            "ned": sum(r["edits"] for r in rows) / sum(r["norm"] for r in rows),
            "cer_no_space": sum(r["edits_ns"] for r in rows) / chars,
            "exact": sum(r["edits"] == 0 for r in rows) / len(rows),
            "collapsed": sum(r["edits"] >= r["chars"] for r in rows) / len(rows),
            "non_javanese": sum(not r["jv_only"] for r in rows) / len(rows)}


def page_delta(a: list[dict], b: list[dict], den: str = "chars", resamples: int = 10_000, seed: int = 0) -> list[float]:
    """SK 95% bootstrap per halaman untuk selisih a - b (baris yang sama di kedua sisi). den "chars" = CER (pembagi
    panjang label, sama di kedua sisi); "norm" = NED (pembagi panjang terpanjang label/keluaran, milik tiap sisi)."""
    pages: dict[str, list[tuple]] = {}
    for ra, rb in zip(a, b):
        pages.setdefault(ra["page"], []).append((ra["edits"], ra[den], rb["edits"], rb[den]))
    names = sorted(pages)
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        sample = [pages[rng.choice(names)] for _ in names]
        ea, da, eb, db = (sum(x[k] for p in sample for x in p) for k in range(4))
        diffs.append(ea / da - eb / db)
    diffs.sort()
    return [diffs[int(0.025 * (resamples - 1))], diffs[int(0.975 * (resamples - 1))]]


def run_score(args) -> None:
    lines = load_lines()
    by_id = {r["id"]: r for r in lines}
    found = methods()
    if REFERENCE not in found:
        raise SystemExit(f"pembanding {REFERENCE} tidak ada di {RESULTS}")
    scored = {}
    for key, m in found.items():
        rows = []
        for line_id, text in m["texts"].items():
            line = by_id[line_id]
            ref = line["reference"]
            hyp = norm(text)
            rows.append({"line": line_id, "page": line["source_id"], "kind": line["kind"], "chars": len(ref),
                         "norm": max(len(ref), len(hyp)),
                         "edits": Levenshtein.distance(ref, hyp), "edits_ns": Levenshtein.distance(ref, hyp.replace(" ", "")),
                         "jv_only": javanese_only(hyp)})
        scored[key] = {r["line"]: r for r in rows}
    ref_rows = scored[REFERENCE]
    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": REFERENCE, "prompt_sha": PROMPT_SHA,
              "rules": "NFC + buang baris baru/tab di ujung (spasi dihitung); CER Levenshtein per karakter, urutan logis; "
                       "tanpa pembersihan lain",
              "methods": {}}
    for key, m in found.items():
        rows = scored[key]
        entry = {"label": m["label"], "group": m["group"], "meta": {k: v for k, v in m["meta"].items() if k != "texts"},
                 "covered": len(rows), "subsets": {}}
        if m.get("seconds"):
            entry["seconds_per_line"] = sum(m["seconds"]) / len(m["seconds"])
        for name in list(KINDS) + ["semua"]:
            ids = [r["id"] for r in lines if name == "semua" or r["kind"] == name]
            if not all(i in rows for i in ids):
                continue
            mine = [rows[i] for i in ids]
            entry["subsets"][name] = stats(mine)
            if key != REFERENCE:
                entry["subsets"][name]["delta_vs_reference"] = (
                    entry["subsets"][name]["cer"] - stats([ref_rows[i] for i in ids])["cer"])
                entry["subsets"][name]["ci95_pages"] = page_delta(mine, [ref_rows[i] for i in ids])
                entry["subsets"][name]["ned_delta_vs_reference"] = (
                    entry["subsets"][name]["ned"] - stats([ref_rows[i] for i in ids])["ned"])
                entry["subsets"][name]["ned_ci95_pages"] = page_delta(mine, [ref_rows[i] for i in ids], "norm")
        # Subset bersama untuk metode yang hanya membaca sebagian (mis. uji buta 50 baris).
        if not entry["subsets"]:
            ids = [i for i in (r["id"] for r in lines) if i in rows]
            entry["subsets"]["dibaca"] = stats([rows[i] for i in ids])
            entry["subsets"]["dibaca"]["reference_cer"] = stats([ref_rows[i] for i in ids])["cer"]
            entry["subsets"]["dibaca"]["reference_ned"] = stats([ref_rows[i] for i in ids])["ned"]
        report["methods"][key] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "method_matrix.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["| Metode | NED cetak | Δ NED cetak, SK 95% | CER cetak (308) | Δ CER cetak, SK 95% | Tulisan tangan (437) | "
          "Semua (745) | Tanpa spasi (cetak) | Baris persis (cetak) | Runtuh ≥100% (cetak) | Non-aksara (cetak) |",
          "|" + "---|" * 11]

    def pct(x):
        return "—" if x is None else f"{100 * x:.2f}%"

    for key, e in sorted(report["methods"].items(), key=lambda kv: (kv[1]["subsets"].get("cetak", {}).get("cer", 9))):
        if e["group"] in ("ablasi-baca", "arsitektur"):  # tabel sendiri (ablation, export)
            continue
        c, h, s = (e["subsets"].get(k) for k in ("cetak", "tulisan tangan", "semua"))
        d = dn = "—"
        if c and "ci95_pages" in c:
            lo, hi = c["ci95_pages"]
            d = f"{100 * c['delta_vs_reference']:+.2f} [{100 * lo:+.2f}; {100 * hi:+.2f}]"
            lo, hi = c["ned_ci95_pages"]
            dn = f"{100 * c['ned_delta_vs_reference']:+.2f} [{100 * lo:+.2f}; {100 * hi:+.2f}]"
        if not c and "dibaca" in e["subsets"]:
            x = e["subsets"]["dibaca"]
            md.append(f"| {e['label']} | {pct(x['cer'])} pada {x['lines']} baris dibaca (resmi {pct(x['reference_cer'])}) "
                      f"| — | — | — | — | — | — | — | — | — |")
            continue
        md.append(f"| {e['label']} | {pct(c and c['ned'])} | {dn} | {pct(c and c['cer'])} | {d} | {pct(h and h['cer'])} | {pct(s and s['cer'])} | "
                  f"{pct(c and c['cer_no_space'])} | {pct(c and c['exact'])} | {pct(c and c['collapsed'])} | "
                  f"{pct(c and c['non_javanese'])} |")
    (OUT / "method_matrix.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# Ablasi komponen metode kita: tiap pasangan beda satu hal (perlakuan lawan kontrol, titik lanjut dan langkah sama),
# kecuali dua baris rujukan di akhir. Prediksi dari ekspor hasil; selisih dan SK per halaman dari eval_slant.real.
ABLATION = [
    ("data", "Font tambahan (8 font)", "crnn_fonts", "crnn_core", "dipakai"),
    ("data", "Jarak antar suku kata acak", "crnn_fase7_track", "crnn_fase7_ctrl", "dipakai"),
    ("data", "Ejaan baku di separuh baris", "crnn_fase8_std", "crnn_fase8_ctrl", "dipakai"),
    ("data", "Sisipan aksara langka, versi 1", "crnn_fase6_rare", "crnn_fase6_ctrl", "tidak dipakai"),
    ("data", "Sisipan aksara langka, versi 2", "crnn_fase7_track_rare", "crnn_fase7_track", "tidak dipakai"),
    ("data", "Sisipan aksara langka, versi 2 diulang", "crnn_fase10_rare", "crnn_fase10_ctrl", "tidak dipakai"),
    ("data", "Render miring (italic)", "crnn_fase9_slant", "crnn_fase9_ctrl", "tidak dipakai"),
    ("data", "Pecahan baris tetangga", "crnn_fase11_neighbor", "crnn_fase11_ctrl", "tidak dipakai"),
    ("baca", "Beam 16 + LM karakter ejaan baku", "crnn_fase9_ctrl_beam", "crnn_fase9_ctrl", "tersedia (Demo)"),
    ("baca", "Beam 16 tanpa LM", "abl_beam_nolm", "crnn_fase9_ctrl", "pembanding"),
    ("baca", "Beam 16 + LM karakter ejaan korpus", "abl_beam_lm_lama", "crnn_fase9_ctrl", "pembanding"),
    ("rujukan", "Derau antar-run (resep sama diulang)", "crnn_fase11_ctrl", "crnn_fase10_ctrl", "rujukan"),
    ("rujukan", "+1.500 langkah tanpa perlakuan", "crnn_fase10_ctrl", "crnn_fase9_ctrl", "rujukan"),
]


def run_ablation(args) -> None:
    from eval_slant import real

    results = RESULTS
    staging = None
    extra = [a for _, _, a, _, _ in ABLATION if (METHODS / f"{a}.jsonl").exists()]
    if extra:
        # eval_slant.real membaca prediksi dari folder hasil; metode di out/methods digabung ke salinan sementara.
        import shutil
        import tempfile

        staging = Path(tempfile.mkdtemp(prefix="abl"))
        shutil.copyfile(RESULTS / "lines.jsonl", staging / "lines.jsonl")
        with (staging / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            f.write((RESULTS / "predictions.jsonl").read_text(encoding="utf-8"))
            for name in extra:
                for x in (METHODS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
                    if x.strip():
                        r = json.loads(x)
                        f.write(json.dumps({"pipeline": name, "line": r["line"], "text": r["text"]}, ensure_ascii=False) + "\n")
        results = staging
    rows = []
    for group, label, a, b, status in ABLATION:
        try:
            rep = real(a, b, results, OUT / "ablation")
        except SystemExit as e:
            print(f"dilewati {label}: {e}")
            continue
        row = {"group": group, "label": label, "a": a, "b": b, "status": status}
        for key in ("cetak", "tulisan tangan", "semua"):
            g = rep["kinds"].get(key) or rep["groups"][key]
            row[key] = {"cer_a": g["cer_a"], "cer_b": g["cer_b"], "delta": g["cer_a"] - g["cer_b"],
                        "ci95_pages": rep["ci95_pages"][key], "lines": g["lines"], "pages": g["pages"]}
        rows.append(row)
    aug = json.loads((ROOT / "out/ablation.json").read_text(encoding="utf-8")) if (ROOT / "out/ablation.json").exists() else None
    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "pairs": rows, "augment": aug}
    (OUT / "ablation_matrix.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    def cell(c):
        lo, hi = c["ci95_pages"]
        return f"{100 * c['delta']:+.2f} [{100 * lo:+.2f}; {100 * hi:+.2f}]"

    md = ["| Kelompok | Komponen | Perlakuan − kontrol | Cetak (308) | Tulisan tangan (437) | Semua (745) | Status |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['group']} | {r['label']} | {r['a'].replace('crnn_', '')} − {r['b'].replace('crnn_', '')} | "
                  f"{cell(r['cetak'])} | {cell(r['tulisan tangan'])} | {cell(r['semua'])} | {r['status']} |")
    (OUT / "ablation_matrix.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if staging:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)


# Fakta tetap per metode untuk tabel (selain angka yang dihitung): keterangan, berjalan offline atau tidak, dan apakah
# metodenya dilatih untuk aksara Jawa. Metode yang tidak terdaftar memakai label dari berkas meta-nya.
FACTS = {
    "crnn_fase9_ctrl": ("kita", True, True, "CNN + BiLSTM + CTC, 4,6 juta parameter, dilatih pada data sintetis; jalur angka resmi."),
    "crnn_fase9_ctrl_beam": ("kita", True, True, "Model yang sama, dibaca dengan beam 16 + model bahasa karakter (order 5); "
                             "bobot disetel di dev sintetis."),
    "gpt_gpt-5.6-sol": ("vlm", False, False, "Model unggulan GPT-5.6 lewat API, setelan bawaan (bernalar sebelum menjawab)."),
    "gpt_gpt-4o": ("vlm", False, False, "VLM umum lewat API; model yang juga diuji makalah NusaAksara (ACL 2025)."),
    "vlm_claude_opus55": ("vlm", False, False, "Dibaca lewat subagen Claude Code (8 pembaca paralel, citra bernama acak). "
                          "Beberapa pembaca membuka 4-5 citra sekaligus, bukan satu per satu."),
    "tesseract_jav": ("umum", True, False, "Mesin OCR terbuka; tidak punya model aksara Jawa, jadi dipakai model bahasa "
                      "Jawa beraksara Latin ('jav'). Menunjukkan bahwa OCR umum tidak bisa dipakai."),
    "vlm_zeroshot": ("vlm", False, False, "Uji buta 2026-09-14 pada 50 baris (22 cetak), prompt dan versi model tidak tercatat."),
    "arch_crnn_base": ("arsitektur", True, True, "BiLSTM 2 lapis dua arah; checkpoint run base (fase 4b)."),
    "arch_transformer": ("arsitektur", True, True, "Encoder Transformer 4 lapis, 4 kepala, feed-forward 768, posisi sinusoidal."),
    "arch_crnn_full": ("arsitektur", True, True, "BiLSTM; model resmi (rantai base sampai fase9_ctrl)."),
    "archT_fase9_ctrl": ("arsitektur", True, True, "Transformer dilatih ulang lewat rantai yang sama dengan model resmi."),
}
# Tabel arsitektur utama: pasangan tahap awal (resep base) dan pasangan resep penuh; tahap antara di architecture_stages.
ARCH_MAIN = ["arch_crnn_base", "arch_transformer", "arch_crnn_full", "archT_fase9_ctrl"]
PRICES = {"gpt-4o": (2.50, 10.00)}  # USD per juta token masukan / keluaran, harga publik OpenAI


def sub_cell(s: dict | None, full: bool = False) -> dict | None:
    if not s:
        return None
    out = {"cer": s["cer"], "ned": s["ned"], "lines": s["lines"], "pages": s["pages"]}
    if "delta_vs_reference" in s:
        out |= {"delta": s["delta_vs_reference"], "ci95": s["ci95_pages"],
                "ned_delta": s["ned_delta_vs_reference"], "ned_ci95": s["ned_ci95_pages"]}
    if full:
        out |= {k: s[k] for k in ("cer_no_space", "exact", "collapsed", "non_javanese")}
    return out


# Pembanding arsitektur per tahap rantai model resmi: (tahap, bacaan Transformer, bacaan CRNN, folder run Transformer,
# folder run CRNN). Bacaan Transformer dari out/methods (scripts/run_arch_transformer_full.sh); CRNN dari ekspor hasil,
# kecuali tahap base (out/methods/arch_crnn_base). Val = validasi bersih terakhir di log run (font latih).
STAGES = [
    ("base", "arch_transformer", "arch_crnn_base", "arch_transformer", "base"),
    ("fase5_fonts", "archT_fase5_fonts", "crnn_fonts", "archT_fase5_fonts", "fase5_fonts"),
    ("fase6_ctrl", "archT_fase6_ctrl", "crnn_fase6_ctrl", "archT_fase6_ctrl", "fase6_ctrl"),
    ("fase7_track", "archT_fase7_track", "crnn_fase7_track", "archT_fase7_track", "fase7_track"),
    ("fase8_std", "archT_fase8_std", "crnn_fase8_std", "archT_fase8_std", "fase8_std"),
    ("fase9_ctrl", "archT_fase9_ctrl", "crnn_fase9_ctrl", "archT_fase9_ctrl", "fase9_ctrl"),
]


def last_val(run: str) -> float | None:
    log = ROOT / "out/checkpoints" / run / "log.jsonl"
    if not log.exists():
        return None
    vals = [json.loads(x)["val_cer"] for x in log.read_text(encoding="utf-8").splitlines() if '"val_cer"' in x]
    return vals[-1] if vals else None


def arch_stages(lines: list[dict]) -> list[dict]:
    """Transformer lawan CRNN di tiap tahap rantai yang sudah dibaca: CER/NED baris cetak dan semua, selisih CER cetak
    dengan SK 95% per halaman, dan val bersih terakhir masing-masing."""
    exported: dict[str, dict] = {}
    for x in (RESULTS / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            exported.setdefault(r["pipeline"], {})[r["line"]] = r["text"]

    def texts(name: str) -> dict | None:
        path = METHODS / f"{name}.jsonl"
        if path.exists():
            return {r["line"]: r.get("raw", r["text"]) for r in
                    (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}
        return exported.get(name)

    def rows(t: dict) -> list[dict]:
        out = []
        for line in lines:
            ref, hyp = line["reference"], norm(t[line["id"]])
            out.append({"page": line["source_id"], "kind": line["kind"], "chars": len(ref),
                        "norm": max(len(ref), len(hyp)), "edits": Levenshtein.distance(ref, hyp)})
        return out

    def rate(sel: list[dict], den: str) -> float:
        return sum(r["edits"] for r in sel) / sum(r[den] for r in sel)

    stages = []
    for stage, t_name, c_name, t_run, c_run in STAGES:
        t, c = texts(t_name), texts(c_name)
        if t is None or c is None or len(t) < len(lines) or len(c) < len(lines):
            continue
        a, b = rows(t), rows(c)
        pa = [r for r in a if r["kind"] == "cetak"]
        pb = [r for r in b if r["kind"] == "cetak"]
        stages.append({"stage": stage, "transformer": t_name, "crnn": c_name,
                       "val_transformer": last_val(t_run), "val_crnn": last_val(c_run),
                       "cetak": {"cer_t": rate(pa, "chars"), "cer_c": rate(pb, "chars"), "ned_t": rate(pa, "norm"),
                                 "ned_c": rate(pb, "norm"), "delta": rate(pa, "chars") - rate(pb, "chars"),
                                 "ci95": page_delta(pa, pb)},
                       "semua": {"cer_t": rate(a, "chars"), "cer_c": rate(b, "chars")}})
    return stages


def run_export(args) -> None:
    """Satu berkas siap tampil untuk web (php artisan aksara:matrix): out/results/matrix.json, skema 1."""
    mm = json.loads((OUT / "method_matrix.json").read_text(encoding="utf-8"))
    ab = json.loads((OUT / "ablation_matrix.json").read_text(encoding="utf-8"))
    lines = load_lines()
    methods_out, arch_out = [], []
    for key, e in mm["methods"].items():
        group = e["group"]
        if group == "ablasi-baca":
            continue
        subsets = e["subsets"]
        meta = e["meta"]
        row = {"key": key, "label": e["label"], "group": FACTS.get(key, (group,))[0],
               "offline": FACTS[key][1] if key in FACTS else None, "trained": FACTS[key][2] if key in FACTS else None,
               "note": FACTS[key][3] if key in FACTS else meta.get("note", ""),
               "params": meta.get("params") or (4_590_909 if key.startswith("crnn_fase9_ctrl") else None),
               "seconds_per_line": e.get("seconds_per_line"),
               "cetak": sub_cell(subsets.get("cetak"), True), "tulisan_tangan": sub_cell(subsets.get("tulisan tangan")),
               "semua": sub_cell(subsets.get("semua")), "partial": None}
        if "dibaca" in subsets:
            d = subsets["dibaca"]
            row["partial"] = {"lines": d["lines"], "cer": d["cer"], "reference_cer": d["reference_cer"],
                              "ned": d["ned"], "reference_ned": d["reference_ned"]}
        if meta.get("output_tokens") is not None and e["covered"]:
            row["output_tokens_per_line"] = meta["output_tokens"] / e["covered"]
            price = PRICES.get(meta.get("model_requested", ""))
            if price:
                row["usd_per_1000_lines"] = 1000 * (meta["input_tokens"] * price[0] + meta["output_tokens"] * price[1]) / 1e6 / e["covered"]
        if meta.get("model_served"):
            row["model_served"] = meta["model_served"]
        if group != "arsitektur":
            methods_out.append(row)
        elif key in ARCH_MAIN:
            arch_out.append(row)
    # Waktu baca model kita diukur terpisah lewat jalur layanan Demo (subperintah latency), bukan saat ekspor hasil.
    latency_path = METHODS / "latency.json"
    if latency_path.exists():
        latency = json.loads(latency_path.read_text(encoding="utf-8"))
        for row in methods_out:
            pipeline = {"crnn_fase9_ctrl": "crnn_greedy", "crnn_fase9_ctrl_beam": "crnn_beam_lm"}.get(row["key"])
            if pipeline in latency:
                row["seconds_per_line"] = latency[pipeline]
    order = {"kita": 0, "vlm": 1, "umum": 2}
    methods_out.sort(key=lambda r: (r["cetak"] is None, order.get(r["group"], 3), r["cetak"]["ned"] if r["cetak"] else 9))
    arch_out.sort(key=lambda r: ARCH_MAIN.index(r["key"]))
    for row in arch_out:
        for gate in ("G1", "G2"):
            path = ROOT / f"out/eval/{row['key']}_{gate}_2k.json"
            row[gate.lower()] = json.loads(path.read_text(encoding="utf-8"))["cer"] if path.exists() else None
        log = ROOT / "out/checkpoints" / Path(json.loads((METHODS / f"{row['key']}.meta.json").read_text(encoding="utf-8"))["checkpoint"]).parent.name / "log.jsonl"
        vals = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if '"val_cer"' in x] if log.exists() else []
        row["val_clean"] = vals[-1]["val_cer"] if vals else None
        row["val_steps"] = vals[-1]["step"] if vals else None
    pairs = [{"group": r["group"], "label": r["label"], "a": r["a"], "b": r["b"], "status": r["status"],
              **{k.replace(" ", "_"): r[k] for k in ("cetak", "tulisan tangan", "semua")}} for r in ab["pairs"]]
    augment = [{"k": r["k"], "op": r["added"], "clean": r["clean"], "heavy": r["heavy"], "g3": r["G3"],
                "g3_no_space": r["G3_no_space"]} for r in (ab.get("augment") or [])]
    kinds = {name: {"lines": sum(r["kind"] == name for r in lines),
                    "pages": len({r["source_id"] for r in lines if r["kind"] == name})} for name in KINDS}
    report = {"schema": 1, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": mm["reference"],
              "prompt": PROMPT, "prompt_sha": PROMPT_SHA, "rules": mm["rules"],
              "subsets": {"cetak": kinds["cetak"], "tulisan_tangan": kinds["tulisan tangan"],
                          "semua": {"lines": len(lines), "pages": len({r["source_id"] for r in lines})}},
              "methods": methods_out, "architecture": arch_out, "architecture_stages": arch_stages(lines),
              "ablation": pairs, "augment": augment}
    (RESULTS / "matrix.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ditulis: {RESULTS / 'matrix.json'} ({len(methods_out)} metode, {len(arch_out)} arsitektur, "
          f"{len(report['architecture_stages'])} tahap arsitektur, {len(pairs)} pasangan ablasi, {len(augment)} augmentasi)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tesseract")
    t.add_argument("--lang", default="jav")
    t.add_argument("--workers", type=int, default=4)
    c = sub.add_parser("crnn")
    c.add_argument("checkpoint")
    c.add_argument("--name", required=True)
    c.add_argument("--label", default="")
    c.add_argument("--group", default="kita")
    c.add_argument("--threads", type=int, default=4)
    d = sub.add_parser("decode")
    d.add_argument("checkpoint")
    d.add_argument("--name", required=True)
    d.add_argument("--label", required=True)
    d.add_argument("--lm", default="data/charlm/o5_n300000_ds0.5_std0.5.pkl")
    d.add_argument("--alpha", type=float, required=True)
    d.add_argument("--beta", type=float, required=True)
    d.add_argument("--width", type=int, default=16)
    d.add_argument("--threads", type=int, default=4)
    g = sub.add_parser("gpt")
    g.add_argument("--model", required=True)
    g.add_argument("--subset", default="cetak", choices=["cetak", "semua"])
    g.add_argument("--workers", type=int, default=8)
    i = sub.add_parser("ingest")
    i.add_argument("file")
    i.add_argument("--name", required=True)
    i.add_argument("--label", required=True)
    i.add_argument("--group", default="vlm")
    i.add_argument("--model", default="")
    i.add_argument("--prompt-sha", default="")
    lat = sub.add_parser("latency")
    lat.add_argument("--threads", type=int, default=4)
    sub.add_parser("score")
    sub.add_parser("ablation")
    sub.add_parser("export")
    sub.add_parser("prompt")
    args = parser.parse_args(argv)
    if args.cmd == "prompt":
        print(PROMPT)
        print(f"sidik: {PROMPT_SHA}")
        return
    {"tesseract": run_tesseract, "crnn": run_crnn, "decode": run_decode, "gpt": run_gpt, "ingest": run_ingest, "latency": run_latency,
     "score": run_score, "ablation": run_ablation, "export": run_export}[args.cmd](args)


if __name__ == "__main__":
    main()
