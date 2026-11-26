"""Tests for Hugging Face Transformers integration and tokenizer parity."""
import random
from pathlib import Path

import pytest
import torch
from transformers import pipeline

from zest.hf import (
    LemonConfig,
    LemonForMaskedLM,
    LemonForSequenceClassification,
    LemonForTokenClassification,
    LemonModel,
    LemonTokenizer,
)
from zest.model.theme_encoder import ThemeEncoder
from zest.tokenizer import ZESTTokenizer

VOCAB_PATH = Path(__file__).parents[1] / "data" / "tokens" / "tedlh_zest_32K.json"


def _tiny_config(**kw) -> LemonConfig:
    defaults = dict(
        vocab_size=32000,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        ff_mult=2,
        max_position_embeddings=64,
        proj_dim=16,
        num_proj_layers=1,
        proj_ff_mult=2,
        aggregator_num_heads=4,
        residue_hidden_dim=16,
        residue_num_heads=2,
        recycled_num_cycles=3,
    )
    defaults.update(kw)
    return LemonConfig(**defaults)


def test_tokenizer_exact_parity_with_zest():
    py_tok = ZESTTokenizer(str(VOCAB_PATH))
    hf_tok = LemonTokenizer(vocab_file=str(VOCAB_PATH))
    assert hf_tok.vocab_size == py_tok.vocab_size

    rng = random.Random(7)
    aa20 = "ACDEFGHIKLMNPQRSTVWY"
    noisy = aa20 + "XBUZO \n\t"
    seqs = [
        "",
        "A",
        "X",
        "MKTLLLTLVVVTIVCLDLGYT",
        "mktl lltl\nvvvt X B U Z O acdef",
        "XXXXMKTAYXXXX",
    ]
    for _ in range(150):
        alph = noisy if rng.random() < 0.3 else aa20
        s = "".join(rng.choices(alph, k=rng.randint(3, 300)))
        if rng.random() < 0.25:
            s = s.lower()
        seqs.append(s)

    for s in seqs:
        clean = py_tok._ws.sub("", s).upper()
        segs = py_tok._segment(list(clean), dropout=0.0)
        py_ids = [t for _, t in segs]
        py_lens = [l for l, _ in segs]

        hf_ids, hf_lens = hf_tok.segment(s)
        assert hf_ids == py_ids
        assert hf_lens == py_lens
        assert hf_tok.encode(s) == py_ids
        assert hf_tok.decode(hf_ids, skip_special_tokens=True) == py_tok.decode(py_ids, skip_special=True)


def test_tokenizer_save_and_reload(tmp_path):
    tok = LemonTokenizer(vocab_file=str(VOCAB_PATH))
    tok.save_pretrained(tmp_path)
    reloaded = LemonTokenizer.from_pretrained(tmp_path)
    seq = "MKTLLLTLVVVTIVCLDLGYT"
    assert reloaded.encode(seq) == tok.encode(seq)
    batch = reloaded.encode_for_residues([seq, "ACDEFGHIKLMNPQRSTVWY"], return_tensors="pt")
    assert batch["input_ids"].shape[0] == 2
    assert batch["token_lengths"].shape == batch["input_ids"].shape


def test_hf_models_parity_with_theme_encoder():
    torch.manual_seed(0)
    zest_model = ThemeEncoder(
        vocab_size=64, embed_dim=32, num_layers=2, nhead=4, ff_mult=2,
        max_tokens=32, proj_dim=16, num_proj_layers=1, proj_ff_mult=2,
        residue_hidden_dim=16, residue_heads=2, recycled_num_cycles=3,
    ).eval()

    cfg = LemonConfig.from_zest_config(zest_model.config, num_cycles=3)
    hf_model = LemonModel(cfg).eval()
    hf_model.load_state_dict(zest_model.state_dict(), strict=True)

    tokens = torch.randint(5, 64, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[1, 7:] = 0
    tl = torch.full((2, 10), 3, dtype=torch.long)
    tl[1, 7:] = 0
    gi = torch.arange(30).unsqueeze(0).repeat(2, 1)

    with torch.no_grad():
        z_proj = zest_model.project_family(tokens, mask, temperature=None)
        z_tok = zest_model.encode_tokens(tokens, mask)
        z_aa, z_res = zest_model.forward_residue_mlm(tokens, mask, tl, gi, return_features=True)

        hf_out = hf_model(input_ids=tokens, attention_mask=mask)
        hf_aa, hf_res = hf_model.encode_residues(tokens, mask, tl, gi)

    torch.testing.assert_close(hf_out.last_hidden_state, z_tok, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(hf_out.pooler_output, z_proj, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(hf_aa, z_aa, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(hf_res, z_res, atol=1e-5, rtol=1e-5)


def test_all_hf_heads_and_pipelines():
    cfg = _tiny_config(num_labels=3)
    tok = LemonTokenizer(vocab_file=str(VOCAB_PATH))
    seq = "MKTLLLTLVVVTIVCLDLGYT"

    # feature-extraction pipeline
    model = LemonModel(cfg).eval()
    fe = pipeline("feature-extraction", model=model, tokenizer=tok, device=-1)
    feats = fe(seq)
    assert isinstance(feats, list) and len(feats[0]) == len(tok.encode(seq))

    # fill-mask pipeline
    mlm = LemonForMaskedLM(cfg).eval()
    fm = pipeline("fill-mask", model=mlm, tokenizer=tok, device=-1, top_k=3)
    masked_seq = f"MKTLLLTL{tok.mask_token}VVVTIVCLDLGYT"
    preds = fm(masked_seq)
    assert len(preds) == 3

    # sequence classification
    seq_cls = LemonForSequenceClassification(cfg).eval()
    inputs = tok(seq, return_tensors="pt")
    out_sc = seq_cls(**inputs, labels=torch.tensor([1]))
    assert out_sc.logits.shape == (1, 3) and torch.isfinite(out_sc.loss)

    # token / residue classification
    tok_cls = LemonForTokenClassification(cfg).eval()
    res_in = tok.encode_for_residues([seq], return_tensors="pt")
    out_rc = tok_cls(**res_in)
    assert out_rc.logits.shape == (1, len(seq), 3)
    out_tc = tok_cls(**inputs)
    assert out_tc.logits.shape == (1, inputs["input_ids"].shape[1], 3)
