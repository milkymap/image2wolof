import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from image2wolof.api_server import CaptionServer
from image2wolof.services.captioner import Captioner, detokenize, generate_text, load_checkpoint
from image2wolof.settings import VariablesConfig

from conftest import CHECKPOINT


class FakeTokenizer:
    def decode(self, ids): return " ".join(map(str, ids))


class FakeCaptioner:
    device = "cpu"
    def __call__(self, images): return [f"{im.size[0]}x{im.size[1]}" for im in images]


def png(size=(8, 6)) -> bytes:
    buf = io.BytesIO(); Image.new("RGB", size).save(buf, "PNG"); return buf.getvalue()


def test_detokenize_stops_at_eos_and_drops_specials():
    assert detokenize(FakeTokenizer(), [1, 5, 0, 6, 2, 7]) == "5 6"
    assert detokenize(FakeTokenizer(), [1, 5, 6]) == "5 6"


def test_api_is_neutral():
    client = TestClient(CaptionServer(FakeCaptioner(), VariablesConfig(max_files=2)).app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404
    assert "Légendes wolof en direct" in client.get("/").text
    out = client.post("/caption", files=[("files", ("a.png", png())), ("files", ("b.txt", b"nope"))]).json()
    assert out[0]["caption"] == "8x6" and "error" in out[1]
    assert client.post("/caption", files=[("files", (f"{i}.png", png())) for i in range(3)]).status_code == 413


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="runs/ not present")
def test_checkpoint_generates_text():
    model, tok, ck = load_checkpoint(CHECKPOINT, "cpu")
    texts = generate_text(model, torch.randn(2, 196, 768), tok, max_tokens=12)
    assert len(texts) == 2 and all(isinstance(t, str) for t in texts)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="runs/ not present")
def test_captioner_on_images():
    captioner = Captioner(CHECKPOINT, "cuda" if torch.cuda.is_available() else "cpu", max_tokens=24)
    texts = captioner([Image.new("RGB", (320, 240), "white"), Image.new("RGB", (64, 64), "green")])
    assert len(texts) == 2 and all(texts)
