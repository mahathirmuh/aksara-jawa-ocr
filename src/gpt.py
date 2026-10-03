"""Wadah API GPT (OpenAI) untuk ablasi koreksi pasca-OCR.

Keputusan user 2026-09-14: wadah kunci API hanya untuk GPT; model dipilih belakangan. Modul ini
hanya wadahnya: kunci, klien, dan cache respons. Logika koreksi belum ada.

Kunci tidak pernah ditulis ke kode, log, maupun cache. Urutan sumber OPENAI_API_KEY:
  1. environment variable
  2. berkas .env di root repo (template: .env.example)

  python -m src.gpt check                  # status kunci (disamarkan), tanpa jaringan
  python -m src.gpt ping --model MODEL     # satu panggilan kecil, BERBAYAR

Respons disimpan di out/gpt_cache/<sha256>.json, berkunci (model, parameter, input). Ablasi yang
diulang tidak membayar ulang, dan setiap angka bisa ditelusuri ke respons mentahnya beserta nama
model yang dilaporkan server dan waktu panggilan. OpenAI tidak menerbitkan snapshot bertanggal
untuk seri gpt-5.6/gpt-6, jadi nama model dari server itulah jejaknya.
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
CACHE_DIR = ROOT / "out" / "gpt_cache"
KEY_ENV = "OPENAI_API_KEY"


class MissingAPIKey(RuntimeError):
    """OPENAI_API_KEY belum diisi."""


@dataclass(frozen=True)
class GPTResult:
    text: str
    model: str | None  # nama model menurut server, bukan yang diminta
    usage: Any
    cached: bool
    created_utc: str
    cache_path: Path | None


def lookup_key(env_path: Path = ENV_PATH, environ: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    """(kunci, sumber). Environment variable mengalahkan .env; nilai kosong dianggap tidak diisi."""
    environ = os.environ if environ is None else environ
    if environ.get(KEY_ENV, "").strip():
        return environ[KEY_ENV].strip(), "environment variable"
    value = ((dotenv_values(env_path).get(KEY_ENV) or "") if env_path.exists() else "").strip()
    return (value, ".env") if value else (None, None)


def mask(secret: str | None) -> str:
    if not secret:
        return "(kosong)"
    if len(secret) <= 8:
        return f"(terisi, {len(secret)} karakter)"
    return f"...{secret[-4:]} ({len(secret)} karakter)"


def cache_key(model: str, input: Any, params: Mapping) -> str:
    blob = json.dumps({"model": model, "input": input, "params": dict(params)}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return json.loads(json.dumps(obj, ensure_ascii=False, default=lambda o: getattr(o, "__dict__", str(o))))


def _write_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


class GPTClient:
    """Klien Responses API OpenAI dengan cache respons di disk."""

    def __init__(self, api_key: str | None = None, cache_dir: Path = CACHE_DIR, client: Any = None,
                 env_path: Path = ENV_PATH, environ: Mapping[str, str] | None = None):
        if api_key is None:
            api_key, _ = lookup_key(env_path, environ)
        if not api_key:
            raise MissingAPIKey(
                f"{KEY_ENV} belum diisi: isi di berkas .env (template .env.example) "
                "atau set environment variable dengan nama itu."
            )
        self.cache_dir = Path(cache_dir)
        if client is None:
            from openai import OpenAI

            # Kunci hanya diteruskan ke klien openai; tidak disimpan sebagai atribut di sini.
            client = OpenAI(api_key=api_key)
        self._client = client

    def __repr__(self) -> str:
        return f"GPTClient(cache_dir={str(self.cache_dir)!r})"

    def generate(self, model: str, input: Any, use_cache: bool = True, **params) -> GPTResult:
        """`input`: teks atau daftar pesan. `params` diteruskan apa adanya ke responses.create
        (mis. reasoning={"effort": "none"}) dan ikut menentukan kunci cache.
        """
        path = self.cache_dir / f"{cache_key(model, input, params)}.json"
        if use_cache and path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            return GPTResult(record["text"], record["model"], record["usage"], True, record["created_utc"], path)

        response = self._client.responses.create(model=model, input=input, **params)
        record = {
            "model_requested": model,
            "model": getattr(response, "model", None),
            "params": params,
            "input": input,
            "text": response.output_text,
            "usage": _jsonable(getattr(response, "usage", None)),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "response": _jsonable(response),
        }
        if use_cache:
            _write_atomic(path, record)
        return GPTResult(record["text"], record["model"], record["usage"], False, record["created_utc"],
                         path if use_cache else None)


def check(env_path: Path = ENV_PATH, environ: Mapping[str, str] | None = None) -> list[str]:
    """Laporan status tanpa jaringan dan tanpa membuka kunci."""
    key, source = lookup_key(env_path, environ)
    cached = sum(1 for _ in CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else 0
    return [
        f".env: {'ada' if env_path.exists() else 'belum ada (salin .env.example ke .env)'}",
        f"{KEY_ENV}: {mask(key)}" + (f" dari {source}" if source else ""),
        f"status: {'siap' if key else 'BELUM SIAP: kunci kosong'}",
        f"cache respons: {cached} berkas di {CACHE_DIR}",
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="status kunci tanpa memanggil API")
    ping = sub.add_parser("ping", help="satu panggilan kecil untuk memastikan kunci & model (berbayar)")
    ping.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    if args.command == "check":
        print("\n".join(check()))
        return
    result = GPTClient().generate(args.model, "Balas dengan satu kata: ok", use_cache=False)
    print(f"model menurut server: {result.model}; balasan: {result.text[:80]!r}; usage: {result.usage}")


if __name__ == "__main__":
    main()
