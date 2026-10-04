"""Layanan model untuk web (Demo): citra satu baris -> Unicode aksara Jawa.

  .venv/Scripts/python -m uvicorn src.serve:app --host 127.0.0.1 --port 8011

Model dan LM dimuat sekali saat permintaan pertama. Jalan di CPU; hanya mendengarkan 127.0.0.1.
Konfigurasi lewat environment variable:
  OCR_CHECKPOINT  default DEFAULT_CHECKPOINT = checkpoint run resmi (OFFICIAL_RUN di scripts/export_results.py)
  OCR_LM          default data/charlm/o5_n300000_ds0.5.pkl
Setelan beam (alpha 0,25, beta 1,0, lebar 16) dipilih di dev dengan checkpoint fase5_fonts (`scripts/beam_eval.py`)
dan belum disetel ulang untuk checkpoint lain; jalur resmi adalah greedy.
"""

import io
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.align import ctc_logp, greedy_path, lm_logp, syllable_spans
from src.beam import line_log_probs, prefix_beam_search
from src.charlm import CharLM
from src.infer import load_checkpoint
from src.tokenizer import logical_syllables

ROOT = Path(__file__).resolve().parents[1]
ALPHA, BETA, WIDTH = 0.25, 1.0, 16
# Harus sama dengan run resmi di scripts/export_results.py (dijaga tests/test_export_results.py).
DEFAULT_CHECKPOINT = "out/checkpoints/fase7_track/last_snapshot.pt"
DEFAULT_LM = "data/charlm/o5_n300000_ds0.5.pkl"
PIPELINES = ("crnn_greedy", "crnn_beam_lm")
MAX_BYTES = 8 * 1024 * 1024

app = FastAPI(title="Aksara OCR", description="Citra satu baris -> Unicode aksara Jawa")


def _path(env: str, default: str) -> Path:
    path = Path(os.environ.get(env, default))
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=1)
def resources():
    checkpoint = _path("OCR_CHECKPOINT", DEFAULT_CHECKPOINT)
    lm_path = _path("OCR_LM", DEFAULT_LM)
    model, tokenizer = load_checkpoint(str(checkpoint), "cpu")
    return model, tokenizer, CharLM.load(lm_path), checkpoint, lm_path


def config_label(pipeline: str, checkpoint: Path, lm_path: Path) -> str:
    name = f"crnn:{checkpoint.parent.name}"
    if pipeline == "crnn_greedy":
        return f"{name} · greedy"
    return f"{name} · beam {WIDTH} · LM {lm_path.stem} · α {ALPHA} · β {BETA}"


def predict_image(img: Image.Image, pipeline: str = "crnn_beam_lm", pad_ratio: float = 0.0) -> dict:
    if pipeline not in PIPELINES:
        raise ValueError(f"pipeline tidak dikenal: {pipeline}; pilihan: {', '.join(PIPELINES)}")
    started = time.perf_counter()
    model, tokenizer, lm, checkpoint, lm_path = resources()
    lp = line_log_probs(img, model, "cpu", pad_ratio)
    greedy_text = tokenizer.to_logical(tokenizer.decode(greedy_path(lp)[0]))
    candidates = [{"source": "crnn_greedy", "text": greedy_text}]
    text = greedy_text
    if pipeline == "crnn_beam_lm":
        text = tokenizer.to_logical(tokenizer.decode(prefix_beam_search(lp, tokenizer.charset, lm, ALPHA, BETA, WIDTH)))
        if text != greedy_text:
            candidates.append({"source": "beam_lm", "text": text})
    for c in candidates:
        c["ctc_logp"] = round(ctc_logp(lp, c["text"], tokenizer), 3)
        c["lm_logp"] = round(lm_logp(c["text"], tokenizer, lm), 3)
        c["score"] = round(c["ctc_logp"] + ALPHA * c["lm_logp"] + BETA * len(tokenizer.to_visual(c["text"])), 3)
        c["chosen"] = c["text"] == text
    spans = syllable_spans(lp, tokenizer, img.width, img.height, pad_ratio) or []
    syllables = [{"text": s, "x0": x0, "x1": x1} for s, (x0, x1) in zip(logical_syllables(greedy_text), spans)]
    return {
        "pipeline": pipeline,
        "config": config_label(pipeline, checkpoint, lm_path),
        "text": text,
        "candidates": candidates,
        "syllables": syllables,
        "image": {"width": img.width, "height": img.height},
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.get("/health")
def health() -> dict:
    model, tokenizer, lm, checkpoint, lm_path = resources()
    return {"status": "ok", "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "lm": lm_path.relative_to(ROOT).as_posix(),
            "pipelines": list(PIPELINES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), pipeline: str = Form("crnn_beam_lm"), pad_ratio: float = Form(0.0)) -> dict:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Citra lebih dari 8 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(422, "Berkas bukan citra yang bisa dibaca") from None
    try:
        return predict_image(img, pipeline, pad_ratio)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
