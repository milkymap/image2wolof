import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import click
from dotenv import load_dotenv

DATA = click.Path(exists=True, file_okay=False, path_type=Path)
SPLIT = click.Choice(["train", "validation", "test"])


def variables_config(**overrides):
    """VariablesConfig from .env / environment, CLI options (when given) taking precedence."""
    load_dotenv()
    from .settings import VariablesConfig
    return VariablesConfig(**{k: v for k, v in overrides.items() if v is not None})


@click.group()
def cli():
    """image2wolof: Wolof image captioning with a frozen SigLIP2 encoder and a six-layer transformer decoder."""


@cli.command()
@click.argument("translations", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path), help="dataset directory to write")
@click.option("--vocab-size", type=int, default=None)
@click.option("--tokenizer", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="reuse this tokenizer.model")
@click.option("--images", type=DATA, help="check that every image is in this Flickr30k folder")
def build(translations, out, vocab_size, tokenizer, images):
    """Split caption jsonl(.gz) files into train/validation/test manifests and fit the tokenizer on train."""
    from .services.dataset import build as build_dataset
    config = variables_config(vocab_size=vocab_size)
    print(json.dumps(build_dataset(list(translations), out, config.vocab_size, tokenizer, images), indent=2))


@cli.command()
@click.argument("data", type=DATA)
@click.option("--images", required=True, type=DATA, help="folder of the JPEG files named by image_id")
@click.option("--split", "splits", multiple=True, type=SPLIT, help="default: all three")
@click.option("--batch-size", type=int, default=128)
@click.option("--workers", type=int, default=8)
@click.option("--device", default=None)
def cache(data, images, splits, batch_size, workers, device):
    """Cache the frozen SigLIP2 patch tokens of every image of DATA's manifests (DATA/features/{split}.pt)."""
    from .services.features import cache as cache_features
    config = variables_config(device=device)
    cache_features(data, images, list(splits) or ["train", "validation", "test"], config.siglip_model, config.siglip_revision,
                   config.torch_device(), batch_size, workers)


@cli.command()
@click.argument("data", type=DATA)
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path), help="run directory")
@click.option("--resume", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="last.pt to continue")
@click.option("--init-from", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="start from these weights")
@click.option("--epochs", type=int, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--lr", type=float, default=None)
@click.option("--patience", type=int, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--max-steps", type=int, default=0, help="smoke test: stop after N steps (writes smoke.pt / smoke.json)")
def train(data, out, resume, init_from, epochs, batch_size, lr, patience, seed, max_steps):
    """Train the decoder on DATA's cached features; best.pt on the five-caption validation loss."""
    from .services.trainer import train as train_decoder
    config = variables_config(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, seed=seed)
    best = train_decoder(data, out, config, resume, init_from, max_steps)
    print(f"done best_validation={best:.4f}")


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--data", required=True, type=DATA)
@click.option("--split", type=SPLIT, default="validation")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), help="default: runs/<run>-eval/<split>")
@click.option("--device", default=None)
def evaluate(checkpoint, data, split, out, device):
    """Five-caption loss, BLEU and chrF (five references) of CHECKPOINT on one split."""
    from .services.evaluate import evaluate as evaluate_checkpoint
    config = variables_config(device=device)
    run = checkpoint.parent.name.removesuffix("-scratch")
    out = out or checkpoint.parent.parent / f"{run}-eval" / split
    metrics = evaluate_checkpoint(checkpoint, data, split, out, config.torch_device(), config.batch_size, config.max_tokens,
                                  config.label_smoothing)
    print(json.dumps(metrics, indent=2))


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--host", default=None)
@click.option("--port", type=int, default=None)
@click.option("--device", default=None)
def serve(checkpoint, host, port, device):
    """Serve the captioning page and API (/caption, /health) for CHECKPOINT."""
    from .api_server import CaptionServer
    from .services.captioner import Captioner
    config = variables_config(host=host, port=port, device=device)
    CaptionServer(Captioner(checkpoint, config.torch_device(), config.max_tokens), config).listen()


def main() -> None:
    cli()
