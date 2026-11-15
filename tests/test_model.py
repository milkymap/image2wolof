import pytest
import torch

from image2wolof.services.model import CaptionDecoder, objective

from conftest import CHECKPOINT


def small():
    return CaptionDecoder(vocab_size=50, d=32, layers=2, heads=4, ff=64)


def test_forward_and_objective():
    model, tokens = small(), torch.randint(3, 50, (2, 7))
    assert model(torch.randn(2, 196, 768), tokens).shape == (2, 7, 50)
    assert torch.isfinite(objective(model, torch.randn(2, 196, 768), tokens))
    assert model.head.weight is model.embed.weight


def test_generate_starts_with_bos_and_pads_after_eos():
    model = small().eval()
    ids = model.generate(torch.randn(3, 196, 768), max_tokens=8)
    assert (ids[:, 0] == 1).all() and ids.shape[1] <= 9
    for row in ids.tolist():
        if 2 in row: assert all(x == 0 for x in row[row.index(2) + 1:])


def test_bad_feature_shape():
    with pytest.raises(ValueError):
        small()(torch.randn(1, 49, 768), torch.ones(1, 3, dtype=torch.long))


def test_legacy_config_round_trip():
    model = small()
    ck = {"config": {**model.config, "memory_layers": 0, "add_memory_positions": False}, "model": model.state_dict()}
    assert CaptionDecoder.from_checkpoint(ck).config == model.config
    with pytest.raises(ValueError):
        CaptionDecoder.from_checkpoint({**ck, "config": {**ck["config"], "memory_layers": 2}})


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="runs/ not present")
def test_research_checkpoint_loads_strictly():
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = CaptionDecoder.from_checkpoint(ck)
    assert model.config == dict(vocab_size=4000, d=512, layers=6, heads=8, ff=2048, dropout=0.1, feature_dim=768, grid_size=14)
