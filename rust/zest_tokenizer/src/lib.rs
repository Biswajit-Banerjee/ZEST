use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

const NO_CHILD: u32 = u32::MAX;
const AA_COUNT: usize = 20;
const WS_CODE: u8 = 254;
const UNK_CODE: u8 = 255;

const SPECIAL_TOKENS: [&str; 5] = ["<PAD>", "<UNK>", "<CLS>", "<EOS>", "<MASK>"];
const AMINO_ACIDS: [u8; AA_COUNT] = *b"ACDEFGHIKLMNPQRSTVWY";

/// O(1) ASCII lookup table: standard 20 AAs (upper/lower) -> 0..19,
/// ASCII whitespace -> WS_CODE (254), everything else -> UNK_CODE (255).
const fn build_aa_lut() -> [u8; 256] {
    let mut lut = [UNK_CODE; 256];
    lut[b' ' as usize] = WS_CODE;
    lut[b'\t' as usize] = WS_CODE;
    lut[b'\n' as usize] = WS_CODE;
    lut[b'\r' as usize] = WS_CODE;
    lut[0x0b] = WS_CODE; // vertical tab
    lut[0x0c] = WS_CODE; // form feed

    let mut i = 0;
    while i < AA_COUNT {
        let up = AMINO_ACIDS[i];
        lut[up as usize] = i as u8;
        lut[(up + 32) as usize] = i as u8; // lowercase ASCII
        i += 1;
    }
    lut
}

static AA_LUT: [u8; 256] = build_aa_lut();

#[derive(Clone)]
struct Node {
    children: [u32; AA_COUNT],
    token_id: i32,
}

impl Node {
    #[inline]
    fn new() -> Self {
        Self {
            children: [NO_CHILD; AA_COUNT],
            token_id: -1,
        }
    }
}

/// Fast SplitMix64 PRNG for optional trie dropout during training.
struct SplitMix64(u64);

impl SplitMix64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn gen_range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize % (hi - lo))
    }
}

fn default_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678_9abc_def0)
}

#[pyclass(name = "RustZESTTokenizer")]
pub struct RustZESTTokenizer {
    nodes: Vec<Node>,
    id_to_token: Vec<String>,
    pad_id: u32,
    unk_id: u32,
    cls_id: u32,
    eos_id: u32,
    mask_id: u32,
    vocab_size: usize,
}

impl RustZESTTokenizer {
    fn insert_pattern(&mut self, pattern: &str, token_id: u32) {
        let mut cur: usize = 0;
        for &b in pattern.as_bytes() {
            let code = AA_LUT[b as usize];
            if code >= AA_COUNT as u8 {
                return;
            }
            let next = self.nodes[cur].children[code as usize];
            if next == NO_CHILD {
                let new_idx = self.nodes.len() as u32;
                self.nodes.push(Node::new());
                self.nodes[cur].children[code as usize] = new_idx;
                cur = new_idx as usize;
            } else {
                cur = next as usize;
            }
        }
        self.nodes[cur].token_id = token_id as i32;
    }

    /// Normalize raw input into a compact Vec of AA codes (0..19) and UNK_CODE (255),
    /// skipping ASCII whitespace in a single linear pass.
    #[inline]
    fn normalize_bytes(text: &str, buf: &mut Vec<u8>) {
        buf.clear();
        buf.reserve(text.len());
        for &b in text.as_bytes() {
            let code = AA_LUT[b as usize];
            if code != WS_CODE {
                buf.push(code);
            }
        }
    }

    fn segment_into(
        &self,
        text: &str,
        dropout: f64,
        max_length: Option<usize>,
        rng: &mut SplitMix64,
        syms_buf: &mut Vec<u8>,
        matches_buf: &mut Vec<(u32, u32)>,
    ) -> (Vec<u32>, Vec<u32>) {
        Self::normalize_bytes(text, syms_buf);
        let n = syms_buf.len();
        let cap = n / 3 + 4;
        let mut ids = Vec::with_capacity(cap);
        let mut lens = Vec::with_capacity(cap);
        let limit = max_length.unwrap_or(usize::MAX);

        let mut i: usize = 0;
        if dropout <= 0.0 {
            // Hot inference path: zero extra allocations, flat array trie walk.
            while i < n && ids.len() < limit {
                let mut node_idx: usize = 0;
                let mut best_len: u32 = 0;
                let mut best_tid: i32 = -1;
                let mut j = i;
                while j < n {
                    let c = syms_buf[j];
                    if c >= AA_COUNT as u8 {
                        break;
                    }
                    let next = self.nodes[node_idx].children[c as usize];
                    if next == NO_CHILD {
                        break;
                    }
                    node_idx = next as usize;
                    let tid = self.nodes[node_idx].token_id;
                    if tid >= 0 {
                        best_len = (j - i + 1) as u32;
                        best_tid = tid;
                    }
                    j += 1;
                }
                if best_tid >= 0 {
                    ids.push(best_tid as u32);
                    lens.push(best_len);
                    i += best_len as usize;
                } else {
                    ids.push(self.mask_id);
                    lens.push(1);
                    i += 1;
                }
            }
        } else {
            // Training path with trie dropout (mirrors Python ZESTTokenizer._segment).
            while i < n && ids.len() < limit {
                matches_buf.clear();
                let mut node_idx: usize = 0;
                let mut j = i;
                while j < n {
                    let c = syms_buf[j];
                    if c >= AA_COUNT as u8 {
                        break;
                    }
                    let next = self.nodes[node_idx].children[c as usize];
                    if next == NO_CHILD {
                        break;
                    }
                    node_idx = next as usize;
                    let tid = self.nodes[node_idx].token_id;
                    if tid >= 0 {
                        matches_buf.push(((j - i + 1) as u32, tid as u32));
                    }
                    j += 1;
                }
                if matches_buf.is_empty() {
                    ids.push(self.mask_id);
                    lens.push(1);
                    i += 1;
                } else {
                    // matches_buf is shortest-first; longest is last.
                    let chosen = if matches_buf.len() > 1 && rng.next_f64() < dropout {
                        // Pick uniformly from all shorter matches (0..len-1)
                        let idx = rng.gen_range(0, matches_buf.len() - 1);
                        matches_buf[idx]
                    } else {
                        matches_buf[matches_buf.len() - 1]
                    };
                    ids.push(chosen.1);
                    lens.push(chosen.0);
                    i += chosen.0 as usize;
                }
            }
        }

        (ids, lens)
    }
}

#[pymethods]
impl RustZESTTokenizer {
    #[new]
    fn new(vocab_path: &str) -> PyResult<Self> {
        let raw = fs::read_to_string(vocab_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read {vocab_path}: {e}")))?;
        let parsed: Value = serde_json::from_str(&raw)
            .map_err(|e| PyValueError::new_err(format!("Invalid JSON in {vocab_path}: {e}")))?;
        let map = parsed
            .as_object()
            .ok_or_else(|| PyValueError::new_err("Vocab JSON must be an object {centroid: [members]}"))?;

        let mut tok = Self {
            nodes: Vec::with_capacity(128_000),
            id_to_token: Vec::with_capacity(SPECIAL_TOKENS.len() + AA_COUNT + map.len()),
            pad_id: 0,
            unk_id: 1,
            cls_id: 2,
            eos_id: 3,
            mask_id: 4,
            vocab_size: 0,
        };
        tok.nodes.push(Node::new());

        // Special tokens (IDs 0..4)
        for s in SPECIAL_TOKENS {
            tok.id_to_token.push(s.to_string());
        }

        // Base amino acids (IDs 5..24)
        for &aa in &AMINO_ACIDS {
            let tid = tok.id_to_token.len() as u32;
            let s = (aa as char).to_string();
            tok.id_to_token.push(s.clone());
            tok.insert_pattern(&s, tid);
        }

        // Many-to-one clusters: preserve JSON key order (serde_json preserves order only with preserve_order;
        // wait: we iterate keys in exact JSON order by using preserve_order feature OR scanning raw JSON!).
        // NOTE: Cargo.toml enables serde_json/preserve_order so IDs match Python dict insertion order 100%!
        for (centroid, members_val) in map.iter() {
            let tid = tok.id_to_token.len() as u32;
            tok.id_to_token.push(centroid.clone());
            tok.insert_pattern(centroid, tid);
            if let Some(arr) = members_val.as_array() {
                for mv in arr {
                    if let Some(member) = mv.as_str() {
                        if member != centroid {
                            tok.insert_pattern(member, tid);
                        }
                    }
                }
            }
        }

        tok.vocab_size = tok.id_to_token.len();
        Ok(tok)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[getter]
    fn pad_id(&self) -> u32 {
        self.pad_id
    }

    #[getter]
    fn unk_id(&self) -> u32 {
        self.unk_id
    }

    #[getter]
    fn cls_id(&self) -> u32 {
        self.cls_id
    }

    #[getter]
    fn eos_id(&self) -> u32 {
        self.eos_id
    }

    #[getter]
    fn mask_id(&self) -> u32 {
        self.mask_id
    }

    #[getter]
    fn num_trie_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[pyo3(signature = (text, dropout=0.0, add_special_tokens=false, max_length=None, seed=None))]
    fn encode(
        &self,
        text: &str,
        dropout: f64,
        add_special_tokens: bool,
        max_length: Option<usize>,
        seed: Option<u64>,
    ) -> Vec<u32> {
        let mut rng = SplitMix64::new(seed.unwrap_or_else(default_seed));
        let mut syms = Vec::new();
        let mut matches = Vec::new();
        let content_limit = max_length.map(|m| if add_special_tokens { m.saturating_sub(2) } else { m });
        let (mut ids, _) = self.segment_into(text, dropout, content_limit, &mut rng, &mut syms, &mut matches);
        if add_special_tokens {
            let mut out = Vec::with_capacity(ids.len() + 2);
            out.push(self.cls_id);
            out.append(&mut ids);
            out.push(self.eos_id);
            out
        } else {
            ids
        }
    }

    #[pyo3(signature = (text, dropout=0.0, max_length=None, seed=None))]
    fn segment(
        &self,
        text: &str,
        dropout: f64,
        max_length: Option<usize>,
        seed: Option<u64>,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut rng = SplitMix64::new(seed.unwrap_or_else(default_seed));
        let mut syms = Vec::new();
        let mut matches = Vec::new();
        self.segment_into(text, dropout, max_length, &mut rng, &mut syms, &mut matches)
    }

    #[pyo3(signature = (texts, dropout=0.0, add_special_tokens=false, max_length=None, seed=None))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        dropout: f64,
        add_special_tokens: bool,
        max_length: Option<usize>,
        seed: Option<u64>,
    ) -> Vec<Vec<u32>> {
        let base_seed = seed.unwrap_or_else(default_seed);
        let content_limit = max_length.map(|m| if add_special_tokens { m.saturating_sub(2) } else { m });
        py.allow_threads(|| {
            texts
                .par_iter()
                .enumerate()
                .map(|(idx, text)| {
                    let mut rng = SplitMix64::new(base_seed.wrapping_add(idx as u64));
                    let mut syms = Vec::new();
                    let mut matches = Vec::new();
                    let (mut ids, _) = self.segment_into(text, dropout, content_limit, &mut rng, &mut syms, &mut matches);
                    if add_special_tokens {
                        let mut out = Vec::with_capacity(ids.len() + 2);
                        out.push(self.cls_id);
                        out.append(&mut ids);
                        out.push(self.eos_id);
                        out
                    } else {
                        ids
                    }
                })
                .collect()
        })
    }

    #[pyo3(signature = (texts, dropout=0.0, max_length=None, seed=None))]
    fn segment_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        dropout: f64,
        max_length: Option<usize>,
        seed: Option<u64>,
    ) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let base_seed = seed.unwrap_or_else(default_seed);
        py.allow_threads(|| {
            texts
                .par_iter()
                .enumerate()
                .map(|(idx, text)| {
                    let mut rng = SplitMix64::new(base_seed.wrapping_add(idx as u64));
                    let mut syms = Vec::new();
                    let mut matches = Vec::new();
                    self.segment_into(text, dropout, max_length, &mut rng, &mut syms, &mut matches)
                })
                .unzip()
        })
    }

    #[pyo3(signature = (ids, skip_special=true))]
    fn decode(&self, ids: Vec<u32>, skip_special: bool) -> String {
        let mut out = String::with_capacity(ids.len() * 4);
        for id in ids {
            if skip_special && (id == self.pad_id || id == self.cls_id || id == self.eos_id || id == self.mask_id) {
                continue;
            }
            if let Some(tok) = self.id_to_token.get(id as usize) {
                out.push_str(tok);
            }
        }
        out
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(id as usize).cloned()
    }

    fn get_centroid_vocab(&self) -> HashMap<String, u32> {
        self.id_to_token
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect()
    }
}

#[pymodule]
fn zest_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustZESTTokenizer>()?;
    Ok(())
}
