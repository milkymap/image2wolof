"""Decoder training on cached features: every (image, caption) pair once per epoch, best.pt on the validation loss."""
import json
import math
import time
from pathlib import Path

import sentencepiece as spm
import torch

from ..settings import VariablesConfig
from .captioner import generate_text
from .dataset import PAD, batch, load_split, pairs, sha256
from .evaluate import five_caption_loss
from .features import backbone_key
from .model import CaptionDecoder, objective


def atomic_save(obj, path: Path):
    tmp = Path(f"{path}.tmp"); torch.save(obj, tmp); tmp.replace(path)


def append_jsonl(path: Path, record: dict):
    with path.open("a") as f: f.write(json.dumps(record) + "\n")


def train(root: Path, out: Path, config: VariablesConfig, resume: Path | None = None, init_from: Path | None = None,
          max_steps: int = 0) -> float:
    """Train from scratch (or from `init_from` weights), early stopping on the five-caption validation loss."""
    root, out = Path(root), Path(out)
    if (out / "last.pt").exists() and not resume: raise ValueError("run exists: use --resume or another --out")
    if init_from and resume: raise ValueError("--init-from starts a new run, --resume continues one")
    out.mkdir(parents=True, exist_ok=True)
    device = config.torch_device()
    torch.manual_seed(config.seed); torch.cuda.manual_seed_all(config.seed); torch.set_float32_matmul_precision("high")
    cuda = device == "cuda"

    tok_path = root / "tokenizer.model"
    tok, tok_sha = spm.SentencePieceProcessor(model_file=str(tok_path)), sha256(tok_path)
    tr, vl = load_split(root, "train", tok, device), load_split(root, "validation", tok, device)
    backbone = tr["backbone"]
    if backbone_key(vl["backbone"]) != backbone_key(backbone): raise ValueError("train and validation features differ")
    model = CaptionDecoder(vocab_size=tok.vocab_size(), d=config.d_model, layers=config.layers, heads=config.heads, ff=config.ff,
                           dropout=config.dropout, feature_dim=backbone["shape"][1], grid_size=math.isqrt(backbone["shape"][0])).to(device)
    initialization = "scratch"
    if init_from:
        init = torch.load(init_from, map_location="cpu", weights_only=True)
        if init["tokenizer_sha256"] != tok_sha: raise ValueError("--init-from checkpoint uses another tokenizer")
        if backbone_key(init["vision_backbone"]) != backbone_key(backbone): raise ValueError("visual feature mismatch")
        source = CaptionDecoder.from_checkpoint(init)
        if source.config != model.config: raise ValueError(f"architecture mismatch: {source.config} != {model.config}")
        model.load_state_dict(source.state_dict())
        initialization = {"checkpoint": str(init_from), "sha256": sha256(init_from), "epoch": init.get("epoch")}

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, fused=cuda)
    per_epoch = math.ceil(pairs(tr) / config.batch_size)
    total_steps, warmup = per_epoch * config.epochs, min(config.warmup, per_epoch)

    def rate(step):
        if step < warmup: return (step + 1) / warmup
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, total_steps - warmup))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)

    def state(epoch, loss):
        return {"model": model.state_dict(), "config": model.config, "vision_backbone": backbone, "tokenizer_sha256": tok_sha,
                "epoch": epoch, "validation_loss": loss, "initialization": initialization}

    def samples(n=8):
        texts = generate_text(model, vl["features"][:n], tok, max_tokens=config.max_tokens)
        return [dict(image_id=r["image_id"], prediction=t, references=r["captions_wo"]) for t, r in zip(texts, vl["rows"][:n])]

    (out / "args.json").write_text(json.dumps({"data": str(root), "out": str(out), "resume": str(resume) if resume else None,
                                               "init_from": str(init_from) if init_from else None, "max_steps": max_steps,
                                               **config.model_dump(exclude={"host", "port", "max_files", "max_mb"})}, indent=2))
    (out / "tokenizer.model").write_bytes(tok_path.read_bytes())
    first = step = bad = 0; best = float("inf")
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        if ck["tokenizer_sha256"] != tok_sha or backbone_key(ck["vision_backbone"]) != backbone_key(backbone): raise ValueError("resume mismatch")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        first, step, best, bad, initialization = ck["epoch"], ck["step"], ck["best"], ck["bad"], ck["initialization"]
        torch.set_rng_state(ck["rng"].cpu())
        if cuda: torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
    print(f"train images={tr['tokens'].shape[0]} pairs={pairs(tr)} validation images={vl['tokens'].shape[0]} vocab={tok.vocab_size()} "
          f"steps/epoch={per_epoch} params={sum(p.numel() for p in model.parameters())} device={device}", flush=True)
    if init_from:
        loss = five_caption_loss(model, vl, config.batch_size, config.label_smoothing)
        append_jsonl(out / "metrics.jsonl", {"epoch": 0, "validation_loss": loss, "step": 0})

    for epoch in range(first, config.epochs):
        model.train(); order = torch.randperm(pairs(tr), device=device); total = count = 0
        start_time = time.perf_counter()
        for b, start in enumerate(range(0, len(order), config.batch_size)):
            features, tokens = batch(tr, order[start:start + config.batch_size]); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = objective(model, features, tokens, config.label_smoothing)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip, error_if_nonfinite=True)
            opt.step(); sched.step(); step += 1
            n = int(tokens[:, 1:].ne(PAD).sum()); total += loss.item() * n; count += n
            if b % config.log_every == 0:
                print(f"epoch={epoch + 1} batch={b}/{per_epoch} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.perf_counter() - start_time:.0f}s", flush=True)
            if max_steps and step >= max_steps:
                loss = five_caption_loss(model, vl, config.batch_size, config.label_smoothing)
                report = {"steps": step, "seconds": time.perf_counter() - start_time, "train_loss": total / count,
                          "validation_loss": loss, "samples": samples()}
                atomic_save(state(0, loss), out / "smoke.pt")
                (out / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
                return loss
        seconds = time.perf_counter() - start_time
        loss = five_caption_loss(model, vl, config.batch_size, config.label_smoothing)
        improved = loss < best
        best, bad = (loss, 0) if improved else (best, bad + 1)
        (out / f"samples_{epoch + 1:02d}.json").write_text(json.dumps(samples(), ensure_ascii=False, indent=2))
        metrics = {"epoch": epoch + 1, "train_loss": total / count, "validation_loss": loss, "best_validation": best,
                   "seconds": seconds, "step": step}
        append_jsonl(out / "metrics.jsonl", metrics)
        print(json.dumps(metrics), flush=True)
        if improved: atomic_save(state(epoch + 1, loss), out / "best.pt")
        last = state(epoch + 1, loss)
        last.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), step=step, best=best, bad=bad,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if cuda else [])
        atomic_save(last, out / "last.pt")
        if bad >= config.patience: print("early stopping", flush=True); break
    return best
