# ZEST / LEMON — Contributor & Agent Guide

## Development Rules
- Do **not** run heavy PyTorch workloads or GPU training directly on HPC login nodes; submit via your cluster scheduler (e.g., SLURM) or run on a dedicated compute node.
- Keep `zest/` free of hardcoded cluster paths, large binaries/checkpoints, and experiment-specific imports.

## Architecture Decisions
- `ThemeEncoder` uses **`RecycledResidueDecoder` exclusively** (legacy `ExpansionHead`, `ResidueDecoderHead`, and pre-v2 adapters have been removed).
- `ThemeEncoder.load_pretrained()` supports both pre-`cycle_film` recycling checkpoints and multi-cycle `cycle_film` checkpoints (auto-resizing `cycle_film` to match the checkpoint).
- Experimental tracks (alignment, domain segmentation, Gradio Spaces) live outside this core repository.
- Run `pytest tests/` before opening or merging PRs.
