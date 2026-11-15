"""Machine-translated Wolof captions -> train/validation/test manifests, SentencePiece tokenizer, tensors for training."""
import gzip
import hashlib
import io
import json
import unicodedata
from pathlib import Path

import sentencepiece as spm
import torch

SPLITS = ("train", "validation", "test")
CAPTIONS = 5
MAX_LENGTH = 512
PAD, BOS, EOS = 0, 1, 2


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lines(path: Path) -> list[str]:
    raw = Path(path).read_bytes()
    return (gzip.decompress(raw) if Path(path).suffix == ".gz" else raw).decode("utf-8").splitlines()


def read_translations(paths: list[Path]) -> dict[str, list[dict]]:
    """Rows grouped by split and sorted by image, captions NFC-normalised."""
    rows, seen = {s: [] for s in SPLITS}, set()
    for path in paths:
        for line in read_lines(path):
            r = json.loads(line)
            if r["image_id"] in seen: raise ValueError(f"duplicate image {r['image_id']}")
            if r["split"] not in SPLITS: raise ValueError(f"unknown split {r['split']}")
            captions = [unicodedata.normalize("NFC", c).strip() for c in r["captions_wo"]]
            if len(captions) != CAPTIONS or not all(captions): raise ValueError(f"bad captions for {r['image_id']}")
            seen.add(r["image_id"])
            rows[r["split"]].append(dict(image_id=r["image_id"], split=r["split"], captions_wo=captions, captions_en=r["captions_en"]))
    for split in SPLITS: rows[split].sort(key=lambda r: r["image_id"])
    return rows


def train_tokenizer(captions: list[str], vocab_size: int) -> bytes:
    """Unigram SentencePiece, byte fallback, identity normalisation, pad/bos/eos/unk = 0/1/2/3; no paths in the model."""
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(captions), model_writer=model, vocab_size=vocab_size, model_type="unigram",
                                   character_coverage=1.0, byte_fallback=True, normalization_rule_name="identity",
                                   pad_id=PAD, bos_id=BOS, eos_id=EOS, unk_id=3, input_sentence_size=0,
                                   shuffle_input_sentence=False, minloglevel=2)
    return model.getvalue()


def build(translations: list[Path], out: Path, vocab_size: int, tokenizer: Path | None = None, images: Path | None = None) -> dict:
    """Write {split}.jsonl, tokenizer.model (fitted on train captions, or copied) and report.json into `out`."""
    rows = read_translations(translations)
    if images and (missing := [r["image_id"] for s in SPLITS for r in rows[s] if not (Path(images) / r["image_id"]).is_file()]):
        raise FileNotFoundError(f"{len(missing)} images missing from {images}, e.g. {missing[:3]}")
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[split]), encoding="utf-8")
    model = Path(tokenizer).read_bytes() if tokenizer else train_tokenizer([c for r in rows["train"] for c in r["captions_wo"]], vocab_size)
    (out / "tokenizer.model").write_bytes(model)
    tok = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))
    lengths = {s: sorted(len(tok.encode(c)) + 2 for r in rows[s] for c in r["captions_wo"]) for s in SPLITS}
    if max(v[-1] for v in lengths.values()) > MAX_LENGTH: raise ValueError("caption exceeds positional capacity")
    report = dict(sources={str(p): sha256(p) for p in translations},
                  images={s: len(rows[s]) for s in SPLITS}, pairs={s: CAPTIONS * len(rows[s]) for s in SPLITS},
                  vocab_size=tok.vocab_size(), tokenizer=f"copied from {tokenizer}" if tokenizer else "trained on train captions_wo",
                  tokenizer_sha256=sha256(out / "tokenizer.model"),
                  token_lengths={s: dict(max=v[-1], p99=v[int(.99 * len(v))], mean=round(sum(v) / len(v), 2)) for s, v in lengths.items()},
                  manifests_sha256={s: sha256(out / f"{s}.jsonl") for s in SPLITS})
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def read_manifest(root: Path, split: str) -> list[dict]:
    return [json.loads(line) for line in (Path(root) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def encode_captions(rows: list[dict], tok) -> torch.Tensor:
    """[images, 5, width] token ids, BOS ... EOS, zero padded."""
    ids = [[[BOS] + tok.encode(c) + [EOS] for c in r["captions_wo"]] for r in rows]
    width = max(len(c) for x in ids for c in x)
    if width > MAX_LENGTH: raise ValueError("caption exceeds positional capacity")
    tokens = torch.zeros(len(rows), CAPTIONS, width, dtype=torch.long)
    for i, x in enumerate(ids):
        for j, c in enumerate(x): tokens[i, j, :len(c)] = torch.tensor(c)
    return tokens


def load_split(root: Path, split: str, tok, device: str) -> dict:
    """Cached features + tokenised captions of one split, on `device`."""
    from .features import load_features
    rows = read_manifest(root, split)
    payload = load_features(root, split)
    if payload["filenames"] != [r["image_id"] for r in rows]: raise ValueError("feature order mismatch")
    tokens = encode_captions(rows, tok)
    return dict(features=payload["features"].to(device), tokens=tokens.to(device), lengths=tokens.ne(PAD).sum(-1).to(device),
                rows=rows, backbone=payload["backbone"])


def pairs(data: dict) -> int:
    return data["tokens"].shape[0] * CAPTIONS


def batch(data: dict, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair p -> image p // 5, caption p % 5; tokens trimmed to the longest caption of the batch."""
    image, caption = pair_indices // CAPTIONS, pair_indices % CAPTIONS
    n = int(data["lengths"][image, caption].max())
    return data["features"][image], data["tokens"][image, caption, :n]
