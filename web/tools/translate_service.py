"""Layanan terjemahan untuk halaman Demo (tahap 3 alur).

  ../.venv/Scripts/python -m uvicorn translate_service:app --app-dir tools --host 127.0.0.1 --port 8012

Model (2,5 GB) mulai dimuat di latar begitu layanan dinyalakan, supaya permintaan pertama dari web tidak
habis waktu menunggu pemuatan (~1 menit, lebih lama bila CPU sedang dipakai training). Setelah itu satu
kalimat ~1-3 detik di CPU.
"""

import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nllb  # noqa: E402

_ready = threading.Event()


def _warm_up() -> None:
    nllb._load()
    _ready.set()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="Aksara · terjemahan", description="Jawa (Latin) -> Indonesia, NLLB lokal", lifespan=lifespan)


class TranslateRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": nllb.MODEL, "direction": f"{nllb.SRC} -> {nllb.TGT}", "loaded": _ready.is_set()}


@app.post("/translate")
def translate_one(req: TranslateRequest) -> dict:
    started = time.perf_counter()
    text = nllb.translate([req.text[:1000]])[0]
    return {"translation": text, "model": nllb.MODEL, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
