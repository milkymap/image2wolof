import io
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from .services.captioner import Captioner
from .settings import VariablesConfig

PAGE = Path(__file__).resolve().parent / "static" / "index.html"


class CaptionServer:
    """Neutral public API: GET / (upload page), GET /health, POST /caption (multipart field "files")."""

    def __init__(self, captioner: Captioner, variables_config: VariablesConfig):
        self.captioner, self.config = captioner, variables_config
        self.lock = threading.Lock()  # one batch on the device at a time
        self.app = FastAPI(title="Captioning", docs_url=None, redoc_url=None, openapi_url=None)
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        self.app.get("/")(lambda: FileResponse(PAGE))
        self.app.get("/health")(lambda: {"status": "ok"})
        self.app.post("/caption")(self.caption)

    def caption(self, files: list[UploadFile] = File(...)) -> list[dict]:
        """[{filename, caption, ms}] or {filename, error} per uploaded image."""
        if len(files) > self.config.max_files: raise HTTPException(413, f"at most {self.config.max_files} files per request")
        limit, results, images = self.config.max_mb * 2**20, [], []
        for f in files:
            data = f.file.read(limit + 1)
            try:
                if len(data) > limit: raise ValueError(f"file larger than {self.config.max_mb} MB")
                with Image.open(io.BytesIO(data)) as im: images.append((len(results), im.convert("RGB")))
                results.append({"filename": f.filename})
            except Exception as e:
                results.append({"filename": f.filename, "error": f"{type(e).__name__}: {e}"[:300]})
        if images:
            with self.lock:
                t0 = time.perf_counter()
                texts = self.captioner([im for _, im in images])
                if self.captioner.device == "cuda": torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
            for (i, _), text in zip(images, texts):
                results[i].update(caption=text, ms=round(ms / len(images), 1))
        return results

    def listen(self):
        uvicorn.run(self.app, host=self.config.host, port=self.config.port)
