"""Unit tests for RecycledResidueDecoder and ThemeEncoder checkpoint guardrails."""
import pytest
import torch

from zest.data.finetune import finetune_collate_fn
from zest.data.pretrain import _sample_span_mask, mlm_collate_fn
from zest.model.heads import RecycledResidueDecoder
from zest.model.theme_encoder import HEAD_PREFIX, ThemeEncoder, create_theme_encoder


def _tiny(max_tokens=8, **kw):
    return ThemeEncoder(
        vocab_size=32, embed_dim=16, num_layers=1, nhead=2,
        max_tokens=max_tokens, proj_dim=8, num_proj_layers=1,
        residue_hidden_dim=16, residue_heads=2, **kw,
    )


def _batch(batch=2, max_tokens=8, per_token=4):
    tokens = torch.randint(5, 32, (batch, max_tokens))
    mask = torch.ones(batch, max_tokens, dtype=torch.bool)
    token_lengths = torch.full((batch, max_tokens), per_token, dtype=torch.long)
    n_res = int(token_lengths[0].sum())
    global_indices = torch.arange(n_res).unsqueeze(0).repeat(batch, 1)
    return tokens, mask, token_lengths, global_indices


def test_hidden_dim_must_divide_num_heads():
    with pytest.raises(ValueError):
        RecycledResidueDecoder(768, max_residues=3072, hidden_dim=384, num_heads=7)


def test_create_theme_encoder_presets():
    model = create_theme_encoder("test", vocab_size=64, max_tokens=32)
    assert isinstance(model.profile_expansion_head, RecycledResidueDecoder)
    assert model.config["sequence_head"] == "recycled_decoder"
    with pytest.raises(ValueError, match="Unknown preset"):
        create_theme_encoder("enormous", vocab_size=64)


def test_forward_shapes_and_all_cycles():
    model = _tiny()
    tokens, mask, token_lengths, global_indices = _batch()

    proj = model.project_family(tokens, mask)
    assert proj.shape == (2, 8)
    assert torch.isfinite(proj).all()

    tok_logits, res_logits, z_res = model(
        tokens, mask, token_lengths, global_indices, return_residue_features=True
    )
    assert tok_logits.shape == (2, 8, 32)
    assert res_logits.shape == (2, 32, 20)
    assert z_res.shape == (2, 32, 16)

    all_logits, z_final = model.forward_all_cycles_residue(
        tokens, mask, token_lengths, global_indices, num_cycles=3
    )
    assert len(all_logits) == 3
    assert all(lg.shape == (2, 32, 20) for lg in all_logits)
    assert z_final.shape == (2, 32, 16)


def test_expand_tokens_and_probe_accessors():
    model = _tiny()
    tokens, mask, token_lengths, global_indices = _batch()
    z = model.encode_tokens(tokens, mask)

    head = model.profile_expansion_head
    expanded, local = head._expand_tokens(z, token_lengths, 32)
    assert expanded.shape == (2, 32, 16)
    assert local.shape == (2, 32)

    gi = global_indices.clamp(0, model.max_residues - 1)
    emb = head.get_residue_embeddings(z, token_lengths, gi, model.global_pos_emb)
    feats = head.get_residue_logits(z, token_lengths, gi, model.global_pos_emb)
    assert emb.shape == (2, 32, 16)
    assert feats.shape == (2, 32, 16)


def test_save_and_load_roundtrip_and_hp2_compat(tmp_path):
    """Verify roundtrip save/load plus hp2-style checkpoints without cycle_film
    and with dropped legacy config keys."""
    model = _tiny()
    ckpt_path = tmp_path / "hp2_like.pth"
    model.save_pretrained(ckpt_path)

    # Simulate hp2 checkpoint: no cycle_film in state_dict + extra legacy config keys
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw["config"]["residue_blocks"] = 2
    raw["config"]["residue_window"] = 1
    del raw["model_state_dict"][HEAD_PREFIX + "cycle_film.weight"]
    torch.save(raw, ckpt_path)

    loaded, _ = ThemeEncoder.load_pretrained(ckpt_path, device="cpu")
    assert isinstance(loaded.profile_expansion_head, RecycledResidueDecoder)
    torch.testing.assert_close(
        loaded.core.token_emb.weight, model.core.token_emb.weight
    )
    # Missing cycle_film stays zero-initialized (identity)
    assert torch.all(loaded.profile_expansion_head.cycle_film.weight == 0)


def test_load_resizes_cycle_film_to_checkpoint_cycles(tmp_path):
    model = _tiny(recycled_num_cycles=7)
    ckpt_path = tmp_path / "cyc7.pth"
    model.save_pretrained(ckpt_path)

    # Even if config claims 3 cycles, cycle_film weight shape wins
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw["config"]["recycled_num_cycles"] = 3
    torch.save(raw, ckpt_path)

    loaded, _ = ThemeEncoder.load_pretrained(ckpt_path, device="cpu")
    assert loaded.profile_expansion_head.num_cycles == 7
    assert loaded.profile_expansion_head.cycle_film.num_embeddings == 7


def test_load_rejects_legacy_head_unless_reinit(tmp_path):
    model = _tiny()
    ckpt_path = tmp_path / "legacy.pth"
    model.save_pretrained(ckpt_path)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw["config"]["sequence_head"] = "expansion"
    torch.save(raw, ckpt_path)

    with pytest.raises(ValueError, match="legacy sequence_head='expansion'"):
        ThemeEncoder.load_pretrained(ckpt_path, device="cpu")

    loaded, _ = ThemeEncoder.load_pretrained(ckpt_path, device="cpu", reinit_head=True)
    torch.testing.assert_close(
        loaded.core.token_emb.weight, model.core.token_emb.weight
    )


def test_load_rejects_missing_encoder_parameter(tmp_path):
    model = _tiny()
    ckpt_path = tmp_path / "broken.pth"
    model.save_pretrained(ckpt_path)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    del raw["model_state_dict"]["core.token_emb.weight"]
    torch.save(raw, ckpt_path)

    with pytest.raises(RuntimeError, match="missing 1 parameters outside"):
        ThemeEncoder.load_pretrained(ckpt_path, device="cpu")


def test_span_mask_and_collates():
    torch.manual_seed(0)
    attn_mask = torch.ones(2, 30, dtype=torch.bool)
    mask = _sample_span_mask(attn_mask, mask_prob=0.25, min_span=3, max_span=6)
    assert mask.shape == (2, 30)
    assert mask.dtype == torch.bool
    assert mask.any()

    sample = {
        "input_ids": torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.long),
        "token_lengths": torch.tensor([3, 3, 3, 3, 3, 3], dtype=torch.long),
        "global_indices": torch.arange(18, dtype=torch.long),
        "aa_labels": torch.arange(18, dtype=torch.long),
    }
    batch = mlm_collate_fn([sample, sample], pad_id=0, mask_id=4, vocab_size=32, span_mask=True)
    assert batch["input_ids"].shape == (2, 6)
    assert batch["aa_labels"].shape == (2, 18)

    ft_sample = {
        "anchor_ids": [5, 6, 7],
        "positive_ids": [8, 9, 10],
        "a_label": 1, "t_label": 2, "h_label": 3,
        "a_pos_label": 1, "t_pos_label": 2, "h_pos_label": 3,
        "probability": 0.9,
        "anchor_token_lengths": [2, 2, 2],
        "anchor_aa_labels": [0, 1, 2, 3, 4, 5],
        "positive_token_lengths": [2, 2, 2],
        "positive_aa_labels": [5, 4, 3, 2, 1, 0],
    }
    ft_batch = finetune_collate_fn(
        [ft_sample, ft_sample], pad_id=0, mask_id=4, vocab_size=32,
        mask_prob=0.3, span_mask=True,
    )
    assert ft_batch["anchor_ids"].shape == (2, 3)
    assert ft_batch["positive_ids"].shape == (2, 3)
    assert ft_batch["anchor_aa_labels"].shape == (2, 6)
