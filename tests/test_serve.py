"""Test layanan /predict (FastAPI) dengan checkpoint asli, di CPU."""

import io

import pytest
from PIL import Image

from src.serve import ROOT, app, predict_image
from src.render import render_line

CHECKPOINT = ROOT / "out/checkpoints/fase5_fonts/last_snapshot.pt"
LM = ROOT / "data/charlm/o5_n300000_ds0.5.pkl"
FONT = ROOT / "fonts/NotoSansJavanese-Regular.ttf"
pytestmark = pytest.mark.skipif(not (CHECKPOINT.exists() and LM.exists()), reason="checkpoint/LM belum ada")

TEXT = "ꦲꦏꦸꦧꦺꦴꦠꦼꦤ꧀ꦲꦺꦴꦫ"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    return TestClient(app)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_image_reads_clean_render():
    out = predict_image(render_line(TEXT, FONT, 64), "crnn_beam_lm")
    assert out["text"] == TEXT
    chosen = [c for c in out["candidates"] if c["chosen"]]
    assert len(chosen) == 1 and chosen[0]["text"] == TEXT
    assert all(0 <= s["x0"] < s["x1"] <= 1 for s in out["syllables"])
    assert "".join(s["text"] for s in out["syllables"]) == out["candidates"][0]["text"]


def test_predict_endpoint(client):
    r = client.post("/predict", files={"file": ("baris.png", png_bytes(render_line(TEXT, FONT, 64)), "image/png")},
                    data={"pipeline": "crnn_greedy"})
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline"] == "crnn_greedy" and body["text"] == TEXT
    assert body["config"].startswith("crnn:fase5_fonts")


def test_predict_rejects_non_image_and_unknown_pipeline(client):
    assert client.post("/predict", files={"file": ("x.txt", b"bukan citra", "text/plain")}).status_code == 422
    r = client.post("/predict", files={"file": ("b.png", png_bytes(render_line(TEXT, FONT, 64)), "image/png")},
                    data={"pipeline": "vlm"})
    assert r.status_code == 422


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
