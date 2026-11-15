"""Frozen SigLIP2 patch tokens (last_hidden_state, 196 x 768): bfloat16 compute, float16 cache per split."""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipImageProcessor, SiglipVisionModel

from .dataset import read_manifest, sha256

SHAPE = (196, 768)
PREPROCESSING = ("do_resize", "size", "resample", "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std")


def load_backbone(model_id: str, revision: str, device: str):
    """Official processor and frozen vision tower at a pinned revision."""
    processor = SiglipImageProcessor.from_pretrained(model_id, revision=revision)
    model = SiglipVisionModel.from_pretrained(model_id, revision=revision, attn_implementation="sdpa")
    model = model.to(device).eval().requires_grad_(False)
    assert (model.config.image_size, model.config.patch_size, model.config.hidden_size) == (224, 16, SHAPE[1])
    return processor, model


def backbone_spec(model_id: str, revision: str, processor) -> dict:
    return {"model_id": model_id, "revision": revision, "feature": "last_hidden_state", "shape": list(SHAPE),
            "dtype": "float16", "compute_dtype": "bfloat16", "processor": processor.to_dict()}


def backbone_key(spec: dict) -> dict:
    """Fields that define the visual features (paths and processor serialisation may differ)."""
    return {k: spec[k] for k in ("model_id", "revision", "feature", "shape")}


def check_processor(processor, spec: dict):
    got, want = processor.to_dict(), spec["processor"]
    diff = [k for k in PREPROCESSING if got.get(k) != want.get(k)]
    if diff: raise ValueError(f"image processor differs from the one used for the features: {diff}")


@torch.inference_mode()
def encode(vision, pixels: torch.Tensor) -> torch.Tensor:
    cuda = pixels.device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        return vision(pixel_values=pixels).last_hidden_state


class Images(Dataset):
    def __init__(self, paths, processor): self.paths, self.processor = paths, processor
    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            return self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]


def cache_path(root: Path, split: str) -> Path:
    return Path(root) / "features" / f"{split}.pt"


def cache(root: Path, images: Path, splits: list[str], model_id: str, revision: str, device: str,
          batch_size: int = 128, workers: int = 8) -> dict:
    """Encode every image of the split manifests and save features/{split}.pt next to them."""
    processor, vision = load_backbone(model_id, revision, device)
    spec = backbone_spec(model_id, revision, processor)
    report = {}
    for split in splits:
        names = [r["image_id"] for r in read_manifest(root, split)]
        loader = DataLoader(Images([Path(images) / n for n in names], processor), batch_size=batch_size,
                            num_workers=workers, pin_memory=device == "cuda")
        chunks = []
        for pixels in loader:
            values = encode(vision, pixels.to(device, non_blocking=True)).half().cpu()
            assert tuple(values.shape[1:]) == SHAPE and torch.isfinite(values).all()
            chunks.append(values)
        destination = cache_path(root, split)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"features": torch.cat(chunks), "filenames": names, "backbone": spec,
                   "manifest_sha256": sha256(Path(root) / f"{split}.jsonl")}
        torch.save(payload, destination.with_suffix(".tmp")); destination.with_suffix(".tmp").replace(destination)
        report[split] = {"images": len(names), "bytes": destination.stat().st_size}
        print(split, json.dumps(report[split]), flush=True)
    return report


def load_features(root: Path, split: str) -> dict:
    payload = torch.load(cache_path(root, split), map_location="cpu", weights_only=True)
    if payload["manifest_sha256"] != sha256(Path(root) / f"{split}.jsonl"): raise ValueError("cache built from another manifest")
    if tuple(payload["features"].shape[1:]) != SHAPE or not torch.isfinite(payload["features"]).all(): raise ValueError("bad features")
    return payload
