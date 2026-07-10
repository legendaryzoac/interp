# S7 — In-browser SAE Token Inspector (Epic C, story 1)

You own `C:/Users/Zack/ClaudeCode/interp/web/`. Do NOT touch `model-pipeline/`, `sae-pipeline/`,
`infra/`, or the root `package.json`. No git commits. Windows machine; the dev server config
`interp-web` (port 5273) already exists in the workspace-root `.claude/launch.json`.

## What S7 delivers
A new **SAE** tab on interp.zackwithers.com that, as the user types a prompt, shows **per-token
the top-k active SAE features** (with their plain-language labels), computed live in the browser.
This is the feature that makes the whole SAE explorer legible. Steering (S9) and feature pages
(S8) come later — do NOT build them; but structure so they can plug in.

## Read first (all exist)
- `web/src/lib/runner.ts` — the GPT-2 runner. It already exposes `residualsEnteringBlocks(tokenIds)`
  → `Float32Array[]` where index L is the residual **entering block L**. Layer-8 `resid_pre`
  (what the SAE consumes) is `residualsEnteringBlocks(tokenIds)[8]`, shape [seq,768] flattened.
  Reuse this; do not re-plumb GPT-2. Also note `detectBackend()` / the WebGPU-vs-WASM variant
  logic and the Cache API versioning (`openVersionedCache`) — mirror that pattern for the SAE files.
- `sae-pipeline/BASIS_CONTRACT.md` — THE transform. Before encoding you MUST apply
  `to_sae_input`: subtract, per token, the mean over the 768 d_model dims. No scale, no LayerNorm.
  `sae_input = resid - resid.mean(over d_model)`. Getting this wrong = silently garbage features.
- `web/src/components/Explainer.tsx` + how the existing tabs (Attention/LogitLens/Compare/Circuits)
  are wired in `App.tsx` — add the SAE tab the same way, with an Explainer. Match the site's voice
  (the help text was just deliberately de-AI-ified — keep it plain, human, understated; no
  "Dive in!", no "Let's explore", no emoji).

## SAE artifacts (on D:, published to HF for prod)
`D:/dev/sae-artifacts/L8/`:
- `sae_enc_fp16.onnx` (37.8 MB) — input `x`: Float32Array [1,seq,768]; output `feats`: [1,seq,24576].
  fp32 IO. This is a plain matmul graph: `feats = ReLU((x - b_dec) @ W_enc + b_enc)` already baked in.
- `dashboards/manifest.json` — has `content_hash` (use for Cache API versioning, like the model graphs).
- `dashboards/index.json` — `{layer, sae_release, d_sae:24576, total_nonbos_tokens, histogram_bins,
  features:{ "<id>": { freq, chunk|null, hasDashboard, label, label_confidence } }}`. 24576 entries;
  384 have `hasDashboard:true` + a real `label`. Features without a label just show as
  "feature #<id>" (unlabeled is fine — most are).
- `dashboards/features_0000.json` (256) / `features_0001.json` (128) — full per-feature dashboards
  (top_examples, histogram, logit_lens). S7 only needs `index.json` for labels; the chunk files are
  for S8. You MAY prefetch the two chunks (they're small, 1.6MB) to get labels, OR just use
  index.json's per-feature `label`. Prefer index.json labels — one fetch.

## Dev serving (mirror the model-graph pattern)
The model ONNX graphs are dev-served from `D:/dev/interp-artifacts/onnx` via a Vite middleware.
Add the SAE artifacts the same way: serve `D:/dev/sae-artifacts/L8` at `/sae` in the dev server.
`VITE_SAE_BASE_URL` defaults to `/sae` (dev); in production it will be the HF dataset URL
`https://huggingface.co/datasets/<user>/interp-sae-gpt2-L8/resolve/main/L8` (wired into deploy.yml
later by Zack — just read it from `import.meta.env.VITE_SAE_BASE_URL` with the `/sae` default, and
document the prod value in a comment). The app fetches `${base}/sae_enc_fp16.onnx`,
`${base}/dashboards/manifest.json`, `${base}/dashboards/index.json`.

## Build
1. `src/lib/sae.ts` — load `sae_enc_fp16.onnx` (ORT session, cached & content_hash-versioned),
   load index.json labels. `encode(residLayer8: Float32Array, seq): Float32Array` that applies the
   `to_sae_input` centering then runs the graph → feats [seq,24576]. A `topKPerToken(feats, seq, k)`
   returning, per token, the k highest-activation feature ids + values + labels. Reuse the model
   runner to get `residualsEnteringBlocks(ids)[8]`. (Encoder runs on the SAME backend/EP the model
   picked — respect the shader-f16 gate below.)
2. `src/components/SaeInspector.tsx` (+ subcomponents) — prompt input (can reuse the shared prompt
   bar/tokenizer), a token strip where each token shows its top-k feature chips (label + strength
   bar); click a token to expand its full top-k; click a feature chip → for now a lightweight panel
   showing the label + freq + "feature #id" (S8 will make this a full page — leave a clear seam,
   e.g. an `onFeatureClick(id)` prop that currently opens the panel).
3. Wire the SAE tab into `App.tsx` with an `Explainer` ("Every token, GPT-2's internal state gets
   decomposed into a handful of interpretable features — this shows which ones light up on each word."
   — but in the site's plain voice; rewrite as you see fit, keep it un-hyped).
4. Debounce recompute as the user types; keep the UI responsive.

## CRITICAL: fp16 encoder on native WebGPU (the shader-f16 lesson)
The model pipeline learned the hard way that fp16 ONNX graphs produce **NaN garbage on WebGPU
adapters lacking the `shader-f16` feature** (e.g. this dev box's GTX 1070). `runner.ts`'s
`detectBackend` already gates the model's fp16/WebGPU path on `adapter.features.has('shader-f16')`.
The SAE encoder is fp16 too, so it has the SAME risk. You MUST:
- Verify the encoder produces correct (non-NaN, sane-magnitude) features on whatever EP the app
  actually uses on this machine (which will be WASM, since the 1070 lacks shader-f16 — so the WASM
  path is your required proof).
- Ensure that when the model runs on WASM/int8, the SAE encoder also runs on a path that works
  (WASM). Do NOT force the SAE onto WebGPU independently. If shader-f16 is absent, the encoder must
  run on WASM like the model does. Confirm no NaNs.
- Sanity check: for the prompt "The cat sat on the mat", some features should activate with positive
  magnitudes and stable labels; feature activations must not be all-zero or NaN.

## Verify before reporting (required)
- `npm run build -w web` clean; `npx vitest run` green (add a small unit test for the centering
  transform and topK).
- Preview (`interp-web`): open the SAE tab, type a prompt, screenshot the per-token feature chips
  with real labels. Confirm via console: no NaNs, encoder ran on the expected EP, labels resolve.
  Test a labeled feature you can eyeball (e.g. a citation/number feature on a prompt with "[12]").
- Mobile 375px: no horizontal overflow; chips wrap.
- Report: files added, the EP the encoder ran on + why, a screenshot description, the centering
  unit-test result, and any seams left for S8/S9.
