# S3 — SAE encoder ONNX export + W_dec blob + parity

Execute **S3 only**. Do not start S4+. No git commands.

## Read first (authoritative, in order)
1. `interp/SAE_EXPLORER_BUILD_BRIEF.md` → "Epic B stories" → S3.
2. `interp/sae-pipeline/BASIS_CONTRACT.md` — the S2 gate result. **Authoritative.**
3. `interp/sae-pipeline/basis_transform.py` (`to_sae_input()`) and `verify_basis.py` (how the SAE is loaded).
4. `interp/model-pipeline/quantize.py` and `stamp_manifests.py` — reuse the established fp16-conversion and content_hash-stamping patterns.

## Environment (S2 already built it — reuse, do not rebuild)
- Python: `D:/dev/sae-venv/Scripts/python.exe`
- Always set `HF_HOME=D:/dev/hf-cache`
- **C: is ~98% full.** Write all binary artifacts to `D:/dev/sae-artifacts/L8/`. Only code, docs, and small reports go in the repo.
- Own only `interp/sae-pipeline/`. Do NOT touch `web/`, `infra/`, `model-pipeline/`.

## The SAE (from S2, layer 8)
```python
sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre", device="cpu")
```
cfg: `d_in=768`, `d_sae=24576`, `apply_b_dec_to_input=True`, `normalize_activations="none"`.

**Encode math (b_dec IS pre-subtracted):**
```
f = ReLU((x - b_dec) @ W_enc + b_enc)
```
where `x` is already basis-corrected per BASIS_CONTRACT (per-token mean subtracted over the 768 dims). **Do NOT bake the centering into the graph** — the browser applies `to_sae_input()` itself before calling the encoder. The graph's input `x` is post-centering.

## Deliverables

### 1. `export_sae_onnx.py` → `sae_enc.onnx`
- Graph: input `x` float32 `[1, seq, 768]` (dynamic seq) → output `feats` float32 `[1, seq, 24576]`.
- Bake `b_dec` subtraction + `W_enc` matmul + `b_enc` + ReLU into the graph.
- **Float32 graph IO on every variant** (same contract as the model-pipeline graphs) so the JS glue always passes/reads Float32Array.
- Emit fp32 first, then an fp16 variant (reuse `quantize.py`'s `convert_fp16` approach: `keep_io_types=True`, `onnx.checker.check_model`, and the `force_outputs_fp32` trick if the converter leaves an fp16 output).
- **fp16 caution (learned the hard way in model-pipeline):** the WebGPU EP executes fp16 natively; avoid ops/constants that overflow fp16. There is no causal mask or LayerNorm here, so this graph should be safe — but say so explicitly in your report after checking for any large constants.

### 2. `W_dec` as a raw binary blob (NOT an ONNX graph)
Steering is a JS vector add (`resid' = resid + α·W_dec[f]`), so the browser needs indexable rows, not a decoder graph.
- Write `w_dec_fp16.bin`: raw little-endian fp16, shape `[24576, 768]`, C-contiguous (row `f` = bytes `f*768*2 .. (f+1)*768*2`). ~36 MB.
- Also write `b_dec_fp32.bin` (768 floats) — needed if a later story reconstructs `x_hat`.
- Document the exact byte layout + dtype in the manifest and in your report, since S7/S9 read it from JS.
- Sanity-check: reload the blob with numpy, assert it round-trips to the torch `W_dec` within fp16 tolerance, and assert row `f` equals `W_dec[f]`.

### 3. Parity test: `test_encoder_parity.py`
- Reference: PyTorch `sae.encode(to_sae_input(resid))` (or the explicit formula — assert both agree first).
- Candidate: onnxruntime CPU EP on `sae_enc.onnx`, fed the SAME basis-corrected input.
- Use real residuals (reuse `verify_basis.py`'s extraction) over the S2 prompt set, not just random vectors.
- **Gates:** fp32 max-abs-diff < 1e-4. fp16 max-abs-diff < 1e-2.
- Also report: L0 (mean active features/token) for torch vs onnx fp32 vs onnx fp16 — they should agree closely; a large L0 divergence means ReLU threshold crossings and is worth flagging.
- Exit non-zero on gate failure.

### 4. Manifest + content_hash
- `D:/dev/sae-artifacts/L8/manifest.json`: model/sae ids, layer, `d_in`, `d_sae`, `apply_b_dec_to_input`, the basis transform name ("center: subtract per-token mean over d_model"), file sizes, W_dec byte layout, and a `content_hash` (reuse `stamp_manifests.py`'s scheme — hash the artifact bytes).
- The browser cache-versions on this hash, exactly like the model weights.

### 5. Report + docs
Update `sae-pipeline/README.md` with the S3 commands. In your final report state: artifact sizes (fp32/fp16 encoder, W_dec blob), the parity table (max-abs-diff + L0 for torch/onnx-fp32/onnx-fp16), the content_hash, the exact W_dec byte layout for JS, whether int8 is worth pursuing for the encoder, and anything S7 (browser encoder) must know.

## Notes
- Do NOT ship an int8 encoder unless fp16 parity is comfortable AND you measure int8 separately; report a recommendation rather than assuming.
- Do not upload to HF Hub — that is S6. Local artifacts only.
