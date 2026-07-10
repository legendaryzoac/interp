# SAE Feature Explorer — Build Brief (revised)

Supersedes `Downloads/SAE_EXPLORER_PLAN.md`. That plan was written without knowledge of
the interp repo's current state; this brief reconciles it. Read the "What already exists"
section before writing any code — it deletes a large fraction of the original plan's work.

**Goal (in priority order):** (1) make interp genuinely *explainable* to a layperson —
SAE features + steering are the most intuitive interp artifact if framed as "concepts the
model uses" and "watch me change its mind"; (2) a strong AI-safety/interp resume piece.
Metrics (L0, explained variance, reconstruction loss) are garnish — never the headline.

**Scope = MVP.** Build S1–S10 (S1 done). Explicitly DEFERRED, do not build in v1: full
24k-feature dashboards, self-trained SAE + comparison view, exhaustive RESULTS.md.

---

## What already exists in the repo (reconcile against the old plan)

- **Per-block segmented ONNX graphs** (`embed.onnx`, `block_00..11.onnx`, `unembed.onnx`)
  on HF Hub, cache-versioned by manifest `content_hash`. This is FINER than the old plan's
  proposed front/back split — no new GPT-2 graph export is needed for steering.
- **`web/src/lib/runner.ts` already exposes the steering/encode primitives** (built for
  M1.3 activation patching):
  - `embed(tokenIds)` → residual entering block 0.
  - `residualsEnteringBlocks(tokenIds)` → the 12 residuals entering each block. **The SAE
    reads `residualsEnteringBlocks()[L]`** — that is GPT-2's `hook_resid_pre` at layer L.
  - `continueFromBlock(resid, startBlock, seq)` → runs blocks startBlock..11 + unembed;
    returns the final-position logits. **This IS the steering forward pass.**
  - Currently `continueFromBlock` returns only the final position (patching needed only
    that). Steered *generation* needs the full last-token logits each step — confirm it
    returns what sampling needs, extend minimally if not. Still NO KV cache (v1 recomputes
    the sequence per generated token — acceptable, bounded token budget).
- **Parity harness** (`model-pipeline/validate.py`) + the `content_hash` stamping pattern
  (`stamp_manifests.py`) — reuse for encoder parity and dashboard cache-versioning.
- **Tokenizer**: `gpt-tokenizer` r50k_base (GPT-2 BPE) with `out/token_fixture.json`. Do
  NOT reintroduce a different tokenizer.
- **HF Hub hosting**: model weights load from a HF repo via `VITE_MODEL_BASE_URL`. SAE
  artifacts + dashboards go to a HF **dataset** repo the same way (the old plan's "600MB
  static dir" worry is moot — HF CDN, lazy-fetched, cache-versioned).
- **Hardware reality**: the dev GTX 1070 LACKS WebGPU `shader-f16`, so its default path is
  **WASM/int8**. `runner.ts detectBackend()` gates fp16/WebGPU on the `shader-f16` feature.
  Steering perf MUST be measured on the int8/WASM path, not assumed on WebGPU.

---

## THE correctness gate — S2, do this first, everything depends on it

**Basis/normalization match.** Joseph Bloom's `gpt2-small-res-jb` SAEs were trained on
TransformerLens residuals with default processing (which *centers writing weights* — shifts
residual values across d_model). Our segmented graphs are **raw HuggingFace GPT-2** (no
centering, no LN fold). If the SAE is fed raw residuals when it expects centered ones,
every feature and label is silently wrong.

S2 must, before any UI work:
1. Load `gpt2-small-res-jb` at the chosen layer via current SAELens ([VERIFY] exact release
   name + loading API — SAELens has drifted).
2. Extract layer-L `resid_pre` **the exact way the browser does** — i.e. reproduce
   `residualsEnteringBlocks()[L]` numerically in Python from the same HF/ONNX graphs (or
   assert the HF resid_pre equals the ONNX chain's residual at that boundary — the parity
   suite already proves ONNX≈HF).
3. Encode those residuals and measure reconstruction / explained variance. If poor, find
   the transform the SAE expects (centering and/or a fixed input scale — SAELens exposes
   the training normalization) and apply it. Re-verify. **Document the exact residual basis
   contract** so the browser encoder applies the identical transform.
4. [DECIDE] layer: default **layer 8 resid_pre**; eyeball feature quality at 6–9 and confirm.

Acceptance: a written basis-contract + reconstruction numbers; a Python function that turns
a browser-equivalent layer-L residual into SAE-ready input, unit-tested.

---

## Epic B stories (offline pipeline)

New Python project `interp/sae-pipeline/` (own venv on D:, mirror the model-pipeline uv
pattern; do NOT reuse the parity venv). Pinned deps: `sae-lens`, `transformer-lens`,
`torch`, `datasets`, `anthropic`, `onnx`, `onnxruntime`, `numpy`.

- **S2 — SAE load + basis verification** (the gate above). [VERIFY] SAELens release/API.
- **S3 — Encoder export + W_dec blob.** Export `sae_enc.onnx` (fp16): `f = ReLU((x - b_dec?)
  @ W_enc + b_enc)` — [VERIFY] whether this release pre-subtracts `b_dec` (some do, some
  don't). Export `W_dec` as a raw fp16 binary (`[d_sae, 768]`, or just the rows for
  steerable features) — NO decoder ONNX graph; steering is a JS vector add. Parity: ONNX
  encode vs PyTorch encode, max-abs-diff < 1e-2 fp16, on the S2 basis. Consider int8 encoder
  only if parity holds.
- **S4 — Bounded harvest → curated feature set.** Stream ~2–5M tokens of OpenWebText via
  `datasets`, batch through GPT-2 on the 1070, encode to features. Track per-feature: top-K
  activating snippets (heap + context window), histogram, activation frequency. **Ship
  dashboards for only a few hundred features** ([DECIDE] count; my rec ~300–500 = highest
  frequency + a hand-picked interesting set). Emit per-feature JSON (§ schema below).
  Estimate + report wall-clock on the 1070.
- **S5 — Haiku auto-labeling.** Label ONLY the shipped features (hundreds, not 24k → keeps
  cost in single-digit dollars). Prompt = top ~10 activating snippets with activating tokens
  marked → ≤8-word label + confidence. [VERIFY] current Bedrock/Anthropic model id (Haiku
  4.5 = `us.anthropic.claude-haiku-4-5-20251001-v1:0` on Bedrock; or Anthropic API — reuse
  whichever the account already has). Resumable, disk-cached, `--limit`.
- **S6 — Dashboard artifacts + index → HF dataset repo.** `index.json` (`{feature_id →
  {label, freq, chunk, hasDashboard}}` for ALL features so the Inspector can label live
  activations even for un-dashboarded features), chunked `features_XXXX.json`. Publish to a
  HF **dataset** repo; verify CORS from the interp origin (same check we did for weights).
  Stamp a `content_hash` for cache-versioning.

### Per-feature dashboard JSON
```json
{
  "id": 1234,
  "label": "mentions of legal proceedings",
  "label_confidence": "high|medium|low",
  "freq": 0.0012,
  "histogram": {"bins": [], "counts": []},
  "top_examples": [{"tokens": [], "acts": [], "max_act": 7.2}],
  "logit_lens": {"promoted": [["token", 1.9]], "suppressed": [["token", -1.4]]}
}
```
Bound size: ≤12 examples × ≤24 tokens, activations 8-bit-quantized. Logit lens =
`W_dec[f] @ W_U` ([DECIDE] apply `ln_f` approximation or skip — document choice), top/bottom
10 tokens.

---

## Epic C stories (in-browser) — briefed later, listed for continuity

- **S7 — SAE encoder runtime + Token Inspector.** ORT session for `sae_enc.onnx`; encode
  `residualsEnteringBlocks()[L]` (applying the S2 basis transform); per-token top-k features
  with labels from `index.json`; works for ANY feature (live) even without a dashboard;
  debounced; runs on int8/WASM + fp32 paths; measured latency.
- **S8 — Feature pages + curated gallery.** 10–15 hand-curated strong features on the SAE
  landing; detail page (max-activating examples w/ token highlighting, histogram via
  existing D3, promoted/suppressed tokens). Lazy-load heavy JSON.
- **S9 — Steering playground.** Reuse `continueFromBlock`; `resid' = resid + α·W_dec[f]`
  added in JS from the W_dec blob; baseline vs steered side-by-side; ablation toggle.
  **Measure generation on int8/WASM first**; hard token budget (≤32 generated) + progress
  bar; option to run the continuation fp32 for steering quality; no-intervention path must
  match baseline generation.
- **S10 — SAE tab explainer + degradation.** Plain-language intro ("features are concepts
  the model uses; steer one to change its mind"), reuse the S1 `Explainer` component; honest
  perf/WebGPU messaging; mobile non-broken.

---

## Orchestration notes
- S2 is a hard gate — one agent, must finish and document the basis contract before S3+.
- S3–S6 are largely sequential (each consumes the prior's output) → one pipeline agent,
  or S3 ∥ S4-harvest then S5/S6.
- Epic C (S7–S10) parallels a web agent once S6's `index.json` schema is frozen — same
  interface-freezing pattern that worked for M1.x (freeze the JSON schema, both sides build
  against it).
- Every heavy artifact lazy-loads and is cache-versioned by content_hash. Never bloat the
  initial bundle.
- Keep the metrics quiet; lead every surface with the plain-language payoff.
```
