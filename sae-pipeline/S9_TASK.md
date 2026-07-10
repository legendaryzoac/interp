# S9 — SAE Steering Playground (Epic C, story 3) — the payoff milestone

You own `C:/Users/Zack/ClaudeCode/interp/web/`. Do NOT touch `model-pipeline/`, `sae-pipeline/`,
`infra/`, or root `package.json`. No git. Dev server `interp-web` (port 5273) exists.

## What S9 delivers
Pick an SAE feature, crank a slider, and watch GPT-2's generated text bend toward (or away from) that
feature's concept — live, in the browser. This is the "Golden Gate Bridge" demo, scaled to GPT-2: the
visceral proof that these features are causal, not just correlational. Baseline vs steered completions
side by side from the same prompt.

## The mechanism (mostly already built — reuse, don't re-plumb)
Steering = add a scaled decoder vector to the residual stream at layer 8, then continue the forward pass.
The primitives EXIST in `web/src/lib/runner.ts` (built for M1.3 activation patching):
- `residualsEnteringBlocks(tokenIds)` → `Float32Array[]`; index 8 is the residual entering block 8
  (layer-8 `resid_pre`, what the SAE operates on).
- `continueFromBlock(resid, startBlock, seq)` → runs blocks startBlock..11 then unembed, returns the
  **final-position** logit row (that's what patching needed). For generation you need the next-token
  logits at the LAST position each step — which is exactly what this returns. Reuse it.
- Read `runner.ts` fully first; respect `detectBackend()` and the EP it picks.

The decoder vector: `${SAE_BASE}/w_dec_fp16.bin` (already served by the vite `/sae` middleware) is
`[24576, 768]` little-endian fp16, C-contiguous row-major. Feature f's decoder row = bytes
`f*1536 .. (f+1)*1536`, upcast fp16→fp32. Per `sae-pipeline/BASIS_CONTRACT.md` §S9, steering is applied
in the RAW residual basis: `resid'[pos] = resid[pos] + alpha * W_dec[f]` at every position (or generated
positions — default every position; make it simple). NO centering on the steering add (centering is only
for the SAE *encoder* input; the decoder row lives in raw residual space — re-read BASIS_CONTRACT to
confirm). `b_dec` is NOT added for steering (it's a constant offset that cancels in the intervention).

## Generation loop (no KV cache — accepted for v1)
Greedy or temperature sampling, recompute the full sequence each generated token (GPT-2 small at short
lengths on the app's EP is fine; the model pipeline showed ~1s/forward on WASM int8). Per step:
1. tokenize prompt+generated-so-far → ids.
2. `resid = residualsEnteringBlocks(ids)[8]`.
3. steered: clone resid, add `alpha * W_dec[f]` to every position's 768-vector.
4. `logits = continueFromBlock(steeredResid, 8, seq)` → sample next token (temperature/greedy).
5. append, repeat to a cap (~40 new tokens; make it configurable, keep it modest for perf).
Do the SAME loop with `alpha=0` for the baseline (or just skip the add) so baseline vs steered use the
identical sampling path/seed. Show tokens streaming in as they generate (don't freeze the UI — yield
between steps like the circuits progress fix did). Cap sequence length ~128.

## Load the decoder efficiently
You only need ONE row per chosen feature (1536 bytes), but the slider lets the user pick arbitrary
features, so fetching the whole 37.7MB `w_dec_fp16.bin` once (cached, content_hash-versioned like the
encoder) is acceptable — OR use HTTP Range requests to fetch just the needed row (the HF Xet CDN supports
ranges; dev middleware may not — so full-file-with-cache is the safe default; Range is a nice-to-have,
don't block on it). Decode the row to a Float32Array[768].

## UI — `src/components/SteeringPlayground.tsx`
- Feature picker: reuse the label search / let the user arrive here via the S8 FeaturePage "Steer with
  this feature" button (wire the `onSteer(id)` seam: thread it from SaeInspector → FeaturePage; enable
  the currently-disabled button). Also allow picking a feature by searching labels directly in the
  playground.
- Prompt input (reuse the shared prompt bar if practical) + a **strength slider** (alpha). Range: tune
  empirically per this SAE's decoder norm — start something like -30..+30 or -8..+8 and CALIBRATE by
  testing (report the range you chose and why; too small = no visible effect, too large = degenerate
  repetition/garbage). Include alpha=0 marker. Negative alpha = suppress.
- **Side-by-side**: baseline completion (alpha 0) vs steered completion, same prompt, from the same
  point. Label which feature + alpha. An "ablation" toggle (subtract the feature's own contribution) is
  OPTIONAL — skip if time-constrained; the toward/away slider is the core.
- A small explainer (plain voice, matching the site) + a few **preset "crowd-pleaser" features** the user
  can one-click (pick 2-3 features whose labels + effects are strong/legible from your own testing — e.g. a
  vivid topical or stylistic feature that visibly bends the text; you must actually test which ones work).
- Wire the tab into `App.tsx` alongside the SAE tab (or as a mode within it — your call; keep the S7
  inspector and S8 pages working).

## CRITICAL correctness + perf checks
- **Steering must visibly work.** On at least one feature at a moderate alpha, the steered completion must
  differ meaningfully from baseline in the direction of the feature's concept, WITHOUT collapsing into
  pure repetition/garbage. If you can't find any feature/alpha that visibly steers without breaking, STOP
  and report — that's a real finding (maybe the alpha range or the basis is off), not something to paper over.
- **fp16/shader-f16 lesson:** the model runs on WASM on this box (1070 lacks shader-f16). The generation
  reuses the model's EP via runner — fine. No NaNs in logits; sanity-check the sampled tokens are real
  words, not token 0 / "!".
- **Basis sanity:** if steered output is garbage at ALL alphas including tiny ones, suspect the steering
  is in the wrong basis — recheck BASIS_CONTRACT (the encoder needs centering; the decoder add does NOT).
- Perf: ~40 tokens × 2 (baseline+steered) forwards on WASM ≈ a minute+. Show a progress/streaming state so
  it never looks hung. Consider a shorter default (e.g. 24 tokens) and let the user extend.

## Verify before reporting (screenshots BROKEN — use DOM/console/network)
- `npm run build -w web` clean; `npx vitest run` green (add unit tests: fp16-row decode from a synthetic
  buffer; the steering vector-add on a residual; alpha=0 == baseline).
- Preview: open the steering playground, pick a preset feature, generate, and DOM-verify BOTH baseline and
  steered completions render and DIFFER; confirm via console no NaNs and the EP used. Report the actual
  steered vs baseline text you observed for one feature (this is the key evidence), the alpha range chosen,
  the token cap, and per-token generation latency.
- Mobile 375px: no horizontal overflow; the side-by-side stacks.
- Report: files added/changed, how you wired the S8 onSteer seam + the FeaturePage button enablement,
  the calibrated alpha range, the observed baseline-vs-steered example, perf numbers, and any correctness
  caveats.
