"""Layanan terjemahan untuk halaman Demo (tahap 3 alur) dan halaman Terjemahan.

  ../.venv/Scripts/python -m uvicorn translate_service:app --app-dir tools --host 127.0.0.1 --port 8012

Model (2,5 GB) mulai dimuat di latar begitu layanan dinyalakan, supaya permintaan pertama dari web tidak
habis waktu menunggu pemuatan (~1 menit, lebih lama bila CPU sedang dipakai training). Setelah itu satu
kalimat ~1-3 detik di CPU.

POST /translate
  {"text": "..."}                                          Jawa (Latin) -> Indonesia, satu teks (dipakai Demo)
  {"texts": ["...", ...], "source": "id", "target": "jv"}   beberapa kalimat sekaligus, arah mana pun dari jv/id
Bahasa Jawa di sini selalu beraksara Latin; alih aksara dikerjakan web (PHP).
"""

import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nllb  # noqa: E402

MAX_TEXTS = 16
MAX_CHARS = 1000
_ready = threading.Event()


def _warm_up() -> None:
    nllb._load()
    _ready.set()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="Aksara · terjemahan", description="Jawa (Latin) <-> Indonesia, NLLB lokal", lifespan=lifespan)


class TranslateRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    source: str = "jv"
    target: str = "id"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": nllb.MODEL, "direction": f"{nllb.SRC} -> {nllb.TGT}",
            "directions": ["jv-id", "id-jv"], "loaded": _ready.is_set()}


@app.post("/translate")
def translate(req: TranslateRequest) -> dict:
    if req.source not in nllb.LANGUAGES or req.target not in nllb.LANGUAGES or req.source == req.target:
        raise HTTPException(422, "source dan target harus dua bahasa berbeda dari: " + ", ".join(nllb.LANGUAGES))
    texts = req.texts if req.texts is not None else [req.text or ""]
    if len(texts) > MAX_TEXTS:
        raise HTTPException(422, f"paling banyak {MAX_TEXTS} teks per permintaan")
    started = time.perf_counter()
    out = nllb.translate([t[:MAX_CHARS] for t in texts], src=nllb.LANGUAGES[req.source], tgt=nllb.LANGUAGES[req.target])
    meta = {"model": nllb.MODEL, "direction": f"{req.source}-{req.target}",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
    # Bentuk lama (satu "text") dipertahankan untuk Demo.
    return {"translations": out, **meta} if req.texts is not None else {"translation": out[0], **meta}
