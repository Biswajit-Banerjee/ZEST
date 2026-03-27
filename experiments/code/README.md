# Code Domain: Code Intelligence

## Task
Detect code clones (functionally equivalent programs with different syntax)
and predict defects. Demonstrates that ZEST's domain-aware tokenization
principle generalizes beyond biology to programming languages.

## Tokenizers
- **ZEST-Code**: Syntax-aware tokenizer built from tree-sitter lexer token streams
  - Step 1: Lex CodeSearchNet functions → stream of ~200 lexer token types
  - Step 2: Mine top-32K recurring patterns (n-grams of lexer tokens)
  - Step 3: Build vocab_map.json, use existing ZESTTokenizer class
- **BPE-Code**: Standard byte-level BPE (32K vocab) trained on raw CodeSearchNet text
- **Single-Token**: Character-level baseline (ASCII printable, ~95 tokens)

## Datasets
- **Pretraining**: CodeSearchNet (6.4M functions, 6 languages: Go, Java, JS, PHP, Python, Ruby)
- **Clone Detection**: POJ-104 (104 algorithm problems, 500 implementations each, MAP@R)
- **Defect Detection**: Devign (27K C functions, binary classification, Accuracy)

## Model Sizes (layer pruning from one large pretrained model)
| Variant | Layers | Params | Source |
|---------|--------|--------|--------|
| Small   | 4      | ~15M   | Pruned |
| Medium  | 8      | ~25M   | Pruned |
| Large   | 16     | ~70M   | Full   |

## Experiments

### 1. Scaling curve (ZEST-Code vs BPE-Code)
Pretrain one large model per tokenizer on CodeSearchNet MLM.
Prune to {4L, 8L, 16L}, evaluate on POJ-104 + Devign.

### 2. τ(T) analysis
Compute MI between token vocabulary and code task labels:
- POJ-104: token ↔ algorithm class (104 classes)
- Devign: token ↔ vulnerable/safe (2 classes)

### 3. Baselines
- CodeBERT (125M, BPE tokenizer)
- GraphCodeBERT (125M, BPE + data flow)
- UniXcoder (125M)

## Pipeline

```bash
# Step 1: Build ZEST-Code tokenizer
python experiments/code/build_tokenizer.py \
    --codesearchnet-dir data/code/codesearchnet/ \
    --output data/code/zest_code_vocab.json

# Step 2: Pretrain (MLM on CodeSearchNet)
python experiments/code/pretrain_code.py \
    --tokenizer zest-code \
    --vocab data/code/zest_code_vocab.json \
    --data data/code/codesearchnet/ \
    --model-size large

# Step 3: Prune + evaluate
python experiments/code/prune_and_eval.py \
    --checkpoint outputs/code/zest_large/best.pth \
    --layer-counts 4 8 16 \
    --benchmark poj104 devign

# Step 4: Compute τ
python experiments/code/compute_tau.py \
    --poj104-dir data/code/poj104/ \
    --tokenizers zest-code:data/code/zest_code_vocab.json bpe:data/code/bpe_32k.json
```
