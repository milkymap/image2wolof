"""Checkpoint loading and greedy caption generation from images or cached features."""
from pathlib import Path

import sentencepiece as spm
import torch
from PIL import Image

from .dataset import EOS, sha256
from .features import check_processor, encode, load_backbone
from .model import CaptionDecoder


def load_checkpoint(path: Path, device: str) -> tuple[CaptionDecoder, spm.SentencePieceProcessor, dict]:
    """Decoder, its tokenizer (tokenizer.model next to the checkpoint) and the raw checkpoint."""
    path = Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    tok_path = path.parent / "tokenizer.model"
    if sha256(tok_path) != ck["tokenizer_sha256"]: raise ValueError("tokenizer does not match checkpoint")
    model = CaptionDecoder.from_checkpoint(ck).to(device).eval()
    return model, spm.SentencePieceProcessor(model_file=str(tok_path)), ck


def detokenize(tok, ids: list[int]) -> str:
    ids = ids[1:]
    ids = ids[:ids.index(EOS)] if EOS in ids else ids
    return tok.decode([x for x in ids if x > EOS])


@torch.no_grad()
def generate_text(model: CaptionDecoder, features: torch.Tensor, tok, batch_size: int = 128, max_tokens: int = 96) -> list[str]:
    model.eval(); texts = []
    cuda = features.device.type == "cuda"
    for start in range(0, len(features), batch_size):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            ids = model.generate(features[start:start + batch_size], max_tokens)
        texts += [detokenize(tok, seq) for seq in ids.tolist()]
    return texts


class Captioner:
    """Frozen SigLIP2 (from the checkpoint's backbone spec) + decoder, ready to caption PIL images."""

    def __init__(self, checkpoint: Path, device: str, max_tokens: int = 96):
        self.device, self.max_tokens = device, max_tokens
        self.decoder, self.tok, self.ck = load_checkpoint(checkpoint, device)
        spec = self.ck["vision_backbone"]
        self.processor, self.vision = load_backbone(spec["model_id"], spec["revision"], device)
        check_processor(self.processor, spec)

    def __call__(self, images: list[Image.Image]) -> list[str]:
        pixels = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")["pixel_values"].to(self.device)
        return generate_text(self.decoder, encode(self.vision, pixels), self.tok, max_tokens=self.max_tokens)
