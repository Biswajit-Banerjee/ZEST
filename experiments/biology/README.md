# Biology Domain: Protein Remote Homology & Tokenizer Analysis

## Task
Detect remote homologs — proteins sharing the same structural fold or superfamily
with low sequence identity (<20%).

## Tokenizers
- **ZEST** (`data/tokens/tedlh_zest_32K.json`): Evolution-informed motif tokenizer (32K vocab, ~3.3 residues/token)
- **BPE-32K** (`data/tokens/tedlh_bpe_32k.json`): Standard subword BPE trained on protein sequences

## τ(T) Analysis

Compute mutual information between token vocabulary and structural labels (CPU only):

```bash
python experiments/biology/compute_tau.py \
    --scope-fasta data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa \
    --cath-fasta data/cath-dataset-nonredundant-S20-v4_4_0.fa
```
