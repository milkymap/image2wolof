"""Five-caption loss and multi-reference BLEU / chrF (sacrebleu) of a checkpoint on one split."""
import json
from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch

from .captioner import generate_text, load_checkpoint
from .dataset import CAPTIONS, PAD, batch, load_split, pairs, sha256
from .features import backbone_key
from .model import objective


@torch.no_grad()
def five_caption_loss(model, data: dict, batch_size: int, smoothing: float = 0.1) -> float:
    """Token-weighted loss over all five captions of every image."""
    model.eval(); total = count = 0
    device = data["tokens"].device
    for start in range(0, pairs(data), batch_size):
        features, tokens = batch(data, torch.arange(start, min(start + batch_size, pairs(data)), device=device))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = objective(model, features, tokens, smoothing)
        n = int(tokens[:, 1:].ne(PAD).sum()); total += loss.item() * n; count += n
    return total / count


def score(hyps: list[str], rows: list[dict]) -> dict:
    refs = [[r["captions_wo"][k] for r in rows] for k in range(CAPTIONS)]
    bleu, chrf = sacrebleu.metrics.BLEU(), sacrebleu.metrics.CHRF()
    return {"bleu": bleu.corpus_score(hyps, refs).score, "chrf": chrf.corpus_score(hyps, refs).score,
            "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature())}


def evaluate(checkpoint: Path, root: Path, split: str, out: Path, device: str, batch_size: int = 128,
             max_tokens: int = 96, smoothing: float = 0.1) -> dict:
    """Write metrics.json and predictions.json to `out`; the loss needs the checkpoint tokenizer to be the data one."""
    checkpoint, root, out = Path(checkpoint), Path(root), Path(out)
    if out.resolve() == checkpoint.parent.resolve(): raise ValueError("refusing to write next to the checkpoint")
    if split == "test" and (out / "metrics.json").exists(): raise ValueError(f"{out} already has test metrics")
    model, tok, ck = load_checkpoint(checkpoint, device)
    data = load_split(root, split, spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model")), device)
    if backbone_key(ck["vision_backbone"]) != backbone_key(data["backbone"]): raise ValueError("visual feature mismatch")
    same_tokenizer = sha256(root / "tokenizer.model") == ck["tokenizer_sha256"]
    loss = five_caption_loss(model, data, batch_size, smoothing) if same_tokenizer else None
    hyps = generate_text(model, data["features"], tok, batch_size, max_tokens)
    metrics = {"checkpoint": str(checkpoint), "epoch": ck.get("epoch"), "split": split, "images": len(hyps),
               "references_per_image": CAPTIONS, "data": str(root), "five_caption_loss": loss, **score(hyps, data["rows"]),
               "mean_prediction_words": sum(len(h.split()) for h in hyps) / len(hyps)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "predictions.json").write_text(json.dumps(
        [dict(image_id=r["image_id"], prediction=h, references=r["captions_wo"]) for h, r in zip(hyps, data["rows"])],
        ensure_ascii=False, indent=2))
    return metrics
