# model-pipeline

Segmented ONNX export of GPT-2 small (124M) for the in-browser mechanistic
interpretability visualizer. The model is split at residual-stream boundaries
into 14 graphs (`embed`, `block_00..block_11`, `unembed`), each block also
emitting its post-softmax attention pattern. The web app chains the graphs in
JS and holds the residual stream itself, which makes logit lens (run `unembed`
on any intermediate residual) and activation patching (swap residuals at any
boundary) cheap.

## Pipeline

Run with the pinned venv and the HF cache env var:

```sh
# 1. export fp32 segmented graphs (+ manifest.json)
HF_HOME=D:/dev/hf-cache python export.py --out D:/dev/interp-artifacts/onnx/fp32

# 2. derive fp16 and int8 variants (re-run whenever fp32 is re-exported)
HF_HOME=D:/dev/hf-cache python quantize.py \
    --src D:/dev/interp-artifacts/onnx/fp32 \
    --out-root D:/dev/interp-artifacts/onnx

# 3. parity vs TransformerLens -> reports/parity_report.{md,json}; exits 1 on gate fail
HF_HOME=D:/dev/hf-cache python validate.py \
    --artifacts D:/dev/interp-artifacts/onnx \
    --variants fp32 fp16 int8

# 4. stamp per-variant content_hash into each manifest (web cache versioning);
#    always the LAST step -- quantize.py deliberately strips inherited hashes
python stamp_manifests.py --artifacts D:/dev/interp-artifacts/onnx \
    --variants fp32 fp16 int8
```

`python` = `D:/dev/interp-venv/Scripts/python.exe` (Python 3.12, torch 2.12.1+cpu).
Pinned deps in `requirements.lock`.

Precomputed web assets (run after export/quantize):

```sh
HF_HOME=D:/dev/hf-cache python gallery.py --artifacts D:/dev/interp-artifacts/onnx  # out/gallery.json
HF_HOME=D:/dev/hf-cache python token_fixture.py                                     # out/token_fixture.json
```

## Artifact locations

`D:/dev/interp-artifacts/onnx/{fp32,fp16,int8}/` — 14 `.onnx` graphs + `manifest.json` each.

| Variant | embed | block (×12) | unembed | total |
|---|---|---|---|---|
| fp32 | 157.5 MB | 28.4 MB | 154.4 MB | 652 MB |
| fp16 | 78.8 MB | 14.2 MB | 77.2 MB | 326 MB |
| int8 (mixed recipe, see below) | 39.4 MB | 14.2 MB | 41.8 MB | 252 MB |

The token embedding matrix is duplicated in `embed` and `unembed` (GPT-2 ties
them; separate graphs can't share weights), which is why those two graphs
dominate the totals.

Repo-tracked deliverables (committed): `reports/parity_report.{md,json}`,
`out/gallery.json`, `out/token_fixture.json`. The `.onnx` artifacts themselves
live under `D:/dev/` and are not committed.

## Interface contract

seq axis is dynamic (1..1024). fp16/int8 variants keep **float32** graph IO —
the JS glue passes a `Float32Array` at every boundary regardless of variant.

| Graph | Input | Output(s) |
|---|---|---|
| `embed.onnx` | `input_ids` int64 `[1, seq]` | `resid` f32 `[1, seq, 768]` |
| `block_XX.onnx` | `resid` f32 `[1, seq, 768]` | `resid_out` f32 `[1, seq, 768]`, `pattern` f32 `[1, 12, seq, seq]` (post-softmax) |
| `unembed.onnx` | `resid` f32 `[1, seq, 768]` | `logits` f32 `[1, seq, 50257]` |

## The int8 recipe (low-bandwidth WASM variant)

The `int8/` directory (default for non-WebGPU browsers) is a **mixed recipe**,
not naive dynamic quantization:

- **embed**: weight-only int8 `wte`/`wpe` with per-**dim** (axis=1) MSE-optimal
  scales, dequantized (`DequantizeLinear`) before the Gather — compute stays fp32.
- **blocks**: fp16 (identical files to `fp16/`).
- **unembed**: weight-only int8 lm_head with per-dim MSE scales, plus the k=16
  highest-range rows (GPT-2's outlier dims — 138 et al.) kept exactly in fp32
  via a parallel skinny MatMul (LLM.int8-style split).

Standard ONNX ops only (no MatMulInteger/contrib ops), fp32 graph IO preserved.

Why: `quantize_dynamic` is not weight-only — it quantizes *activations* with one
per-tensor scale at runtime (`DynamicQuantizeLinear`), and GPT-2's residual
stream has extreme activation outliers (|x| up to ~370). That is unfixable from
the weight side. Experiment history (parity vs reference; "full" = 16-prompt
validate.py vs TransformerLens, "hard-6" = 6 hardest prompts vs the fp32 chain):

| Recipe | Size | Pattern max\|diff\| | Lens top-1 | Lens top-5 | Suite |
|---|---|---|---|---|---|
| dynamic per-tensor, all graphs (old int8) | 164 MB | 9.68e-01 | 63.12% | 10.23% | full |
| dynamic per-tensor, blocks only | — | 9.99e-01 | 66.48% | 15.53% | full |
| dynamic per-tensor, embed only | — | 1.10e-01 | 97.56% | 87.10% | full |
| dynamic per-tensor, unembed only | — | 8.46e-06 | 85.46% | 46.64% | full |
| dynamic per-channel, all graphs | 164 MB | 8.56e-01 | 71.58% | 20.41% | full |
| dynamic per-channel, blocks only | — | 8.55e-01 | 81.08% | 35.23% | full |
| weight-only int8 all (per-row embed) | 165 MB | 3.54e-01 | 93.99% | 73.92% | hard-6 |
| weight-only int8 blocks only (MSE) | — | 1.66e-01 | 97.56% | 87.28% | hard-6 |
| weight-only int8 embed only (per-dim MSE) | — | 2.52e-02 | 99.62% | 97.78% | hard-6 |
| weight-only int8 unembed only (per-dim MSE) | — | 0 | 97.19% | 91.23% | hard-6 |
| wo8 embed + fp16 blocks + wo8 unembed ("C1") | 248 MB | 2.52e-02 | 97.50% | 90.68% | full |
| wo8 embed + fp16 blocks + fp16 unembed ("C2") | 287 MB | 2.52e-02 | 99.65% | 97.94% | full |
| **C1 + outlier-split unembed (shipped)** | **252 MB** | **2.52e-02** | **98.35%** | **91.06%** | full |

Key findings: int8 *blocks* are ruled out in any form (weight rounding alone
accumulates to 0.15+ pattern diff vs the 5e-2 gate); embed must be scaled
per-dim, not per-row (outlier dimensions); the outlier split buys ~0.9pp lens
top-1 for +3.2 MB. Plain C1 (248 MB) also passes the gates — set
`UNEMBED_OUTLIER_ROWS = 0` in quantize.py — but with half the top-1 margin.

## Parity (latest)

Reference: TransformerLens `from_pretrained_no_processing("gpt2")` over the 16
fixed prompts in `prompts.json`, across all (layer, position) logit-lens cells.

| Variant | Attn pattern max\|diff\| | Final logit max\|diff\| | Lens top-1 | Lens top-5 |
|---|---|---|---|---|
| fp32 | 8.46e-06 | 4.88e-04 | 100.00% | 100.00% |
| fp16 | 1.41e-02 | 4.87e-01 | 99.87% | 99.43% |
| int8 | 2.52e-02 | 3.48e+00 | 98.35% | 91.06% |

Gates (in `validate.py`): all three variants are hard gates. int8 was promoted
from informational when it became the default WASM-path variant (thresholds:
pattern ≤ 5e-2, top-1 ≥ 97%, top-5 ≥ 90%). The fp16 pattern tolerance is 2e-2 —
a single fp16 block matches fp32 to ~6e-4; the larger chain figure is
residual-stream drift accumulated across 12 blocks and does not move behaviour
(fp16 lens top-1 = 99.87%). See the gate comments in `validate.py` for full
justifications.
