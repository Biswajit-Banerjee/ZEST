"""
Build a ZEST-Code tokenizer from CodeSearchNet using tree-sitter lexing.

Pipeline:
  1. Lex each function → stream of lexer token types (e.g., DEF ID LPAREN ...)
  2. Count all n-grams (2..max_n) of lexer token types across the corpus
  3. Select top-K most frequent patterns as the ZEST-Code vocabulary
  4. Cluster similar patterns (fuzzy matching) → vocab_map.json
  5. The base alphabet is the set of unique lexer token types (~200)

The resulting vocab_map.json can be used directly with ZESTTokenizer,
treating lexer token types as the "amino acids" (base alphabet).

Usage:
    python experiments/code/build_tokenizer.py \
        --codesearchnet-dir data/code/codesearchnet/ \
        --languages python java javascript \
        --vocab-size 32000 \
        --max-ngram 16 \
        --output data/code/zest_code_vocab.json
"""
import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# tree-sitter lexing
# ---------------------------------------------------------------------------

def get_parser(language: str):
    """Get a tree-sitter parser for the given language."""
    try:
        import tree_sitter_languages
        return tree_sitter_languages.get_parser(language)
    except ImportError:
        raise ImportError(
            "Install tree-sitter-languages: pip install tree-sitter-languages"
        )


def lex_function(source: str, language: str) -> List[str]:
    """
    Lex source code into a stream of token type names.

    Returns e.g. ["def", "identifier", "(", "identifier", ",", "identifier", ")", ":"]
    """
    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))

    tokens = []
    def _walk(node):
        if node.child_count == 0:  # leaf node
            tokens.append(node.type)
        else:
            for child in node.children:
                _walk(child)

    _walk(tree.root_node)
    return tokens


def lex_corpus(
    data_dir: str,
    languages: List[str],
    max_functions: Optional[int] = None,
) -> Tuple[List[List[str]], Set[str]]:
    """
    Lex all functions in a CodeSearchNet directory.

    Expected layout: data_dir/{language}/train.jsonl (one JSON per line with "code" field)

    Returns:
        token_streams: list of token-type sequences
        alphabet: set of all unique token types seen
    """
    token_streams = []
    alphabet = set()
    count = 0

    for lang in languages:
        lang_dir = Path(data_dir) / lang
        jsonl_files = sorted(lang_dir.glob("*.jsonl")) + sorted(lang_dir.glob("*.jsonl.gz"))
        if not jsonl_files:
            _log.warning("No JSONL files found for %s in %s", lang, lang_dir)
            continue

        _log.info("Lexing %s...", lang)
        for jf in jsonl_files:
            import gzip
            opener = gzip.open if str(jf).endswith(".gz") else open
            with opener(jf, "rt") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        code = obj.get("code") or obj.get("func") or obj.get("function")
                        if not code:
                            continue
                        tokens = lex_function(code, lang)
                        if len(tokens) < 5:
                            continue
                        token_streams.append(tokens)
                        alphabet.update(tokens)
                        count += 1
                        if max_functions and count >= max_functions:
                            return token_streams, alphabet
                    except Exception:
                        continue

        _log.info("  %s: %d functions lexed so far", lang, count)

    _log.info("Total: %d functions, %d unique token types", count, len(alphabet))
    return token_streams, alphabet


# ---------------------------------------------------------------------------
# Pattern mining
# ---------------------------------------------------------------------------

def mine_ngram_patterns(
    token_streams: List[List[str]],
    max_n: int = 16,
    min_count: int = 50,
) -> Counter:
    """
    Count all n-grams (n=2..max_n) across the corpus.
    Returns Counter of (pattern_tuple → count).
    """
    _log.info("Mining n-gram patterns (n=2..%d, min_count=%d)...", max_n, min_count)
    pattern_counts: Counter = Counter()

    for stream in token_streams:
        n = len(stream)
        for ngram_len in range(2, min(max_n + 1, n + 1)):
            for i in range(n - ngram_len + 1):
                pattern = tuple(stream[i:i + ngram_len])
                pattern_counts[pattern] += 1

    # Filter by minimum count
    filtered = Counter({p: c for p, c in pattern_counts.items() if c >= min_count})
    _log.info("Found %d patterns above min_count=%d (from %d total)",
              len(filtered), min_count, len(pattern_counts))
    return filtered


def build_vocab_map(
    pattern_counts: Counter,
    alphabet: Set[str],
    vocab_size: int = 32000,
) -> Dict[str, List[str]]:
    """
    Select top-K patterns and format as vocab_map.json.

    The "centroid" is the pattern string (token types joined by space).
    Members are just the centroid itself (no fuzzy clustering for code — 
    patterns are exact sequences of discrete token types).
    """
    # Reserve space for alphabet (base tokens)
    n_patterns = vocab_size - len(alphabet)
    top = pattern_counts.most_common(n_patterns)

    vocab_map = {}
    for pattern, count in top:
        centroid = " ".join(pattern)
        vocab_map[centroid] = [centroid]  # self-member

    _log.info("Vocab: %d base tokens + %d patterns = %d total",
              len(alphabet), len(vocab_map), len(alphabet) + len(vocab_map))
    return vocab_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build ZEST-Code tokenizer")
    p.add_argument("--codesearchnet-dir", required=True,
                   help="CodeSearchNet data directory")
    p.add_argument("--languages", nargs="+",
                   default=["python", "java", "javascript", "go", "php", "ruby"])
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-ngram", type=int, default=16)
    p.add_argument("--min-count", type=int, default=50)
    p.add_argument("--max-functions", type=int, default=None)
    p.add_argument("--output", required=True, help="Output vocab_map.json path")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    # Step 1: Lex
    streams, alphabet = lex_corpus(
        args.codesearchnet_dir, args.languages, args.max_functions
    )

    # Step 2: Mine patterns
    pattern_counts = mine_ngram_patterns(streams, args.max_ngram, args.min_count)

    # Step 3: Build vocab
    vocab_map = build_vocab_map(pattern_counts, alphabet, args.vocab_size)

    # Step 4: Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(vocab_map, f, indent=2)

    # Also save alphabet for reference
    alpha_path = args.output.replace(".json", "_alphabet.json")
    with open(alpha_path, "w") as f:
        json.dump(sorted(alphabet), f, indent=2)

    print(f"\nZEST-Code tokenizer built:")
    print(f"  Alphabet: {len(alphabet)} lexer token types")
    print(f"  Patterns: {len(vocab_map)}")
    print(f"  Total vocab: {len(alphabet) + len(vocab_map)}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
