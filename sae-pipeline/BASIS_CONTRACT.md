# SAE Basis Contract — S2 correctness gate (PASS)

**The one line the browser must obey (S7):**

```
sae_input[L] = residualsEnteringBlocks()[L] − mean(residualsEnteringBlocks()[L] over the 768 d_model dims)
```

i.e. mean-center each token's residual vector over d_model, then feed it to the SAE
encoder. Nothing else. No scaling, no LayerNorm, no fold. This is verified below to
recover the SAE's training basis **exactly** (explained variance identical to feeding
the SAE its own native activations; skipping it is catastrophic).

---

## Why a transform is needed at all

- The browser (`web/src/lib/runner.ts::residualsEnteringBlocks()[L]`) produces GPT-2
  `hook_resid_pre` at layer L from the **raw HuggingFace** weights — no TransformerLens
  processing. `model-pipeline/validate.py` already proves the segmented ONNX chain
  equals `HookedTransformer.from_pretrained_no_processing("gpt2")`.
- Joseph Bloom's `gpt2-small-res-jb` SAEs were **not** trained on those raw residuals.
  The SAE's own config declares the model it expects:

  ```
  cfg.model_name                    = "gpt2-small"
  cfg.model_from_pretrained_kwargs  = {"center_writing_weights": true}
  cfg.normalize_activations         = "none"
  cfg.apply_b_dec_to_input          = true
  cfg.prepend_bos                   = true
  cfg.hook_name                     = "blocks.{L}.hook_resid_pre"
  cfg.d_in = 768 , cfg.d_sae = 24576 , architecture = StandardSAE , dtype = float32
  cfg.dataset_path = "Skylion007/openwebtext" , context_size = 128
  ```

  The load recipe is therefore `from_pretrained_no_processing("gpt2",
  center_writing_weights=True)`. The **only** processing flag that is on is
  `center_writing_weights`.

## Why `center_writing_weights` ≡ "subtract the d_model mean" at `hook_resid_pre`

GPT-2 reads from the residual stream only through LayerNorm, whose first step subtracts
the mean over d_model. So the component of the residual along the all-ones direction is
information the model can never read — a free gauge. `center_writing_weights` fixes that
gauge by centering every matrix that *writes* to the residual stream (`W_E`, `W_pos`,
each attention `W_O`, each MLP `W_out`). Because every individual write becomes
mean-zero over d_model, their running sum — the residual stream at every hook, including
`hook_resid_pre` — is mean-zero over d_model too. The raw stream is that same sum without
the centering, so for any position:

```
resid_pre_centered = resid_pre_raw − mean_over_dmodel(resid_pre_raw)
```

The other default-processing flags do **not** change `hook_resid_pre` *values*:
`fold_ln` (moves LN affine params forward, behaviour-preserving), `center_unembed`
(touches the unembed only), `fold_value_biases` (reparametrizes attention values,
behaviour-preserving). So `no_processing + center_writing_weights` and full default
`from_pretrained` give the **identical** `resid_pre`, and both equal `raw − mean`.

## Evidence (measured by `verify_basis.py`, 22 prompts / 348 tokens, CPU)

Three sanity chains, all closed to float32 precision:

1. **Browser == raw HF.** ONNX chain (`embed → block_00..block_{L-1}` from
   `D:/dev/interp-artifacts/onnx/fp32`) vs `from_pretrained_no_processing` `hook_resid_pre`:
   `max|diff|` = 5–7 × 10⁻⁴. The browser really does start from the raw basis.
2. **Transform == center_writing_weights.** `from_pretrained("gpt2")` (default) `resid_pre`
   vs `raw − mean_dmodel(raw)`: `max|diff|` = 2–4 × 10⁻⁴. The mean-subtraction is exactly
   the SAE's expected basis.
3. **Reconstruction recovers.** Feeding the SAE `raw − mean` gives the same explained
   variance as feeding it its own native activations; feeding **raw** (no transform)
   is catastrophic.

| Layer | EV (SAE native basis) | EV (raw, **no** transform) | EV (browser + transform) | L0 (native) | mean\|resid\| |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 6 | 0.9991 | **−22305.7** | 0.9991 | 48.2 | 267.9 |
| 7 | 0.9988 | **−14729.2** | 0.9988 | 53.6 | 280.4 |
| 8 | 0.9983 | **−12104.7** | 0.9983 | 58.2 | 295.2 |
| 9 | 0.9976 | **−10903.4** | 0.9976 | 62.2 | 314.9 |

`browser + transform` matches `SAE native basis` to < 1e-3 EV at every layer. Without the
transform, explained variance is enormously negative and L0 blows up to ~1600–1900 active
features (vs ~48–62): the SAE fires on garbage. **This is the silent failure the gate
exists to prevent.**

> Metric note — `explained_variance = 1 − mean_t‖x_t−x̂_t‖² / mean_t‖x_t−x̄‖²`
> (variance across tokens, summed over d_model; `x̄` = per-dim mean over tokens). The
> ~0.998 values are on short, in-distribution prompts (exactly the interp site's usage)
> and read higher than the ~0.90 typically quoted for these SAEs over full-length
> OpenWebText at context 128. The **gate conclusion does not depend on the absolute EV** —
> it depends on `raw` being catastrophic while `transform` matches the native basis, which
> holds by a factor of ~10⁴.

## What each downstream story must do

- **S7 (in-browser encoder runtime).** Before calling `sae_enc.onnx`, apply
  `x − mean_dmodel(x)` to `residualsEnteringBlocks()[L]`. Pure per-token op, ~768 adds +
  1 divide per token; trivial on the int8/WASM path. Because `normalize_activations ==
  "none"`, there is **no** scale factor — do not add one. (`basis_transform.to_sae_input`
  is the reference; the `scale` arg stays `None` for this release.)
- **S3 (encoder export).** `apply_b_dec_to_input == true`: this StandardSAE encodes as
  `f = ReLU((x − b_dec) @ W_enc + b_enc)` (b_dec **is** pre-subtracted). Bake that into
  `sae_enc.onnx`. Steering decode is `x̂ = f @ W_dec + b_dec`; the W_dec blob is the raw
  decoder. Parity target: ONNX-encode vs PyTorch `sae.encode` on the S2 basis (the
  centered residual).
- **S9 (steering).** Steering happens in the raw residual stream. The decoder vector
  `W_dec[f]` lives in the centered basis, but since centering only removes the all-ones
  component (which LayerNorm ignores), adding `α·W_dec[f]` to the raw residual and
  continuing the forward is valid — the added mean component is a no-op downstream.

## Layer recommendation

**Layer 8** (the brief's default) — confirmed. All of 6–9 share the identical basis
contract and reconstruct comparably (EV 0.997–0.999); the choice is about feature
interpretability, not correctness. Layer 8 is the community-standard demo layer for
`gpt2-small-res-jb` (late-middle: rich semantic/concept features rather than
token/positional ones), with reasonable sparsity (L0 ≈ 58). If S4 harvest surfaces
nicer curated features at 7, switching is free — the transform is layer-independent.

## Reproduce

```
export HF_HOME=D:/dev/hf-cache
D:/dev/sae-venv/Scripts/python.exe verify_basis.py --layers 6 7 8 9 --out reports/basis_report.json
D:/dev/sae-venv/Scripts/python.exe -m pytest test_basis_transform.py -q
```

Full machine-readable numbers: `reports/basis_report.json`. Confirmed with
sae-lens 6.45.3, transformer-lens 3.5.1, transformers 5.13.0, torch 2.13.0 (CPU).
