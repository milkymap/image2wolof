import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = next(iter(sorted(ROOT.glob("runs/*/best.pt"))), ROOT / "runs" / "missing.pt")  # any trained run
WORDS = "góor jigéen xale xaj fas oto welo mbedd ndox géej garab kër simis bu xonq weex ñuul mboq wert bulo ñu ngi mu ci kaw ak".split()


@pytest.fixture
def translations(tmp_path: Path) -> Path:
    """Tiny translation file: 12 train, 3 validation, 3 test images, five random captions each."""
    rng, lines = random.Random(0), []
    for i, split in enumerate(["train"] * 12 + ["validation"] * 3 + ["test"] * 3):
        captions = [" ".join(rng.choices(WORDS, k=rng.randint(4, 12))).capitalize() + "." for _ in range(5)]
        lines.append(json.dumps({"image_id": f"{1000 - i}.jpg", "split": split, "captions_en": ["x"] * 5, "captions_wo": captions}))
    path = tmp_path / "translations.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
