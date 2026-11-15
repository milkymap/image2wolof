import gzip

import pytest
import sentencepiece as spm
import torch

from image2wolof.services.dataset import BOS, EOS, batch, build, encode_captions, read_manifest, read_translations


def test_build_splits_and_tokenizer(translations, tmp_path):
    report = build([translations], tmp_path / "data", vocab_size=320)
    assert report["images"] == {"train": 12, "validation": 3, "test": 3}
    assert report["vocab_size"] == 320
    rows = read_manifest(tmp_path / "data", "train")
    assert [r["image_id"] for r in rows] == sorted(r["image_id"] for r in rows)
    tok = spm.SentencePieceProcessor(model_file=str(tmp_path / "data" / "tokenizer.model"))
    assert (tok.pad_id(), tok.bos_id(), tok.eos_id(), tok.unk_id()) == (0, 1, 2, 3)


def test_build_reuses_tokenizer(translations, tmp_path):
    build([translations], tmp_path / "a", vocab_size=320)
    report = build([translations], tmp_path / "b", vocab_size=320, tokenizer=tmp_path / "a" / "tokenizer.model")
    assert (tmp_path / "a" / "tokenizer.model").read_bytes() == (tmp_path / "b" / "tokenizer.model").read_bytes()
    assert report["tokenizer"].startswith("copied")


def test_duplicate_image_rejected(translations):
    with pytest.raises(ValueError, match="duplicate"):
        read_translations([translations, translations])


def test_encode_and_batch(translations, tmp_path):
    build([translations], tmp_path / "data", vocab_size=320)
    tok = spm.SentencePieceProcessor(model_file=str(tmp_path / "data" / "tokenizer.model"))
    rows = read_manifest(tmp_path / "data", "train")
    tokens = encode_captions(rows, tok)
    assert tokens.shape[:2] == (12, 5) and (tokens[:, :, 0] == BOS).all()
    data = dict(features=torch.randn(12, 196, 768), tokens=tokens, lengths=tokens.ne(0).sum(-1))
    features, ids = batch(data, torch.tensor([0, 7, 59]))
    assert features.shape == (3, 196, 768)
    assert (ids[1] == tokens[1, 2, :ids.shape[1]]).all() and ids.shape[1] == int(data["lengths"][[0, 1, 11], [0, 2, 4]].max())
    assert all(EOS in row for row in ids.tolist())


def test_build_from_gzip_checks_images(translations, tmp_path):
    gz = tmp_path / "captions.jsonl.gz"
    gz.write_bytes(gzip.compress(translations.read_bytes()))
    images = tmp_path / "images"; images.mkdir()
    with pytest.raises(FileNotFoundError):
        build([gz], tmp_path / "data", vocab_size=320, images=images)
    for r in read_translations([translations]).values():
        for row in r: (images / row["image_id"]).touch()
    build([gz], tmp_path / "data", vocab_size=320, images=images)
    build([translations], tmp_path / "plain", vocab_size=320)
    assert (tmp_path / "data" / "train.jsonl").read_bytes() == (tmp_path / "plain" / "train.jsonl").read_bytes()
    assert (tmp_path / "data" / "tokenizer.model").read_bytes() == (tmp_path / "plain" / "tokenizer.model").read_bytes()
