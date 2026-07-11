# S9-fix — Steering sampler hygiene (kill the � and the "…" repetition)

You own `C:/Users/Zack/ClaudeCode/interp/web/`. No git. No other dirs. Dev server `interp-web`
(port 5273) exists. This is a focused QUALITY fix to the already-working S9 steering playground.

## The bug (diagnosed — confirm, then fix)
The steering demo works (steered text visibly bends toward the feature's concept), but the generated
text shows two artifacts:
1. A `�` replacement glyph mid-completion.
2. A degenerate tail like `… … … … …` (or `... ... ...`) repeating.

Root cause is in `web/src/lib/steering.ts` → `sampleToken`: it does a **bare temperature-softmax over
the full 50257-token vocab with NO top-k / top-p truncation and NO repetition penalty**. Consequences:
- No nucleus/top-k ⇒ the long tail (including GPT-2's lone byte-fragment tokens that aren't valid
  standalone UTF-8) can be sampled — steering shifts the distribution so this happens more — rendering
  as `�`. (Note: `decodeIds` in `tokenizer.ts` decodes the whole array at once and is CORRECT; the �
  comes from the model genuinely emitting an invalid partial-byte token, not from decoding.)
- No repetition penalty ⇒ once GPT-2 falls into an ellipsis/word loop, nothing breaks it out.

Confirm this reading by reading `steering.ts` (`sampleToken`, the `generate` loop) and
`components/SteeringPlayground.tsx` (how `baselineIds`/`steeredIds` are built and rendered via
`decodeIds`). If the real cause differs, report it before changing course.

## The fix
In `steering.ts` (and thread through `SteeringPlayground.tsx`'s generate loop):
1. **Nucleus (top-p) sampling**, default p≈0.9 (or top-k≈40 — your call; top-p is preferred). Truncate
   to the smallest set of highest-prob tokens whose cumulative prob ≥ p, renormalize, sample within it.
   This removes the garbage tail (kills most �) and cuts degeneration. Keep it numerically stable.
2. **Repetition penalty** (CTRL-style divide-logits-by-penalty for already-generated ids, penalty≈1.3,
   or a no-repeat-bigram guard) applied over the tokens generated SO FAR in that sequence. This kills
   the `…`/word loops. Apply to the logits before sampling.
3. Keep **greedy** (temperature ≤ 0) deterministic, and keep the **alpha=0 == baseline** invariant:
   baseline and steered must use the IDENTICAL sampler + same seed each step so the only difference is
   the steering vector (otherwise the side-by-side comparison is unfair). The existing seeded RNG per
   step must be preserved.
4. Belt-and-suspenders display guard: ensure the completion still renders cleanly if a stray
   invalid-byte token ever slips through (e.g. decode is already whole-array; optionally strip a
   trailing incomplete multibyte sequence so the LAST token can't show a dangling �). Don't over-engineer
   — the sampler fix is the real remedy.

Sensible defaults, exposed minimally: don't add a pile of new sliders. top-p and repetition-penalty can
be constants (documented) or one modest "less repetitive" default; keep the UI uncluttered. Do NOT change
the steering math (`addSteeringVector`), the basis, or the alpha range — only the sampling.

## Verify before reporting (screenshots BROKEN in this pane — DOM/console/network; generation is slow
under a backgrounded pane, be patient / use a short new-token count)
- `npm run build -w web` clean; `npx vitest run` green (currently 67). Update/add sampler tests:
  top-p truncation picks only the nucleus; repetition penalty lowers a repeated token's odds;
  greedy still deterministic; **alpha-0 path still == baseline** (same seed ⇒ identical ids).
- In preview: pick the **NFL preset (9127)** on a neutral prompt (e.g. "My favorite thing to talk about
  is"), generate ~16–24 tokens, and DOM-verify: (a) NO `�` in the steered output, (b) NO runaway
  `…`/repeated-token tail, (c) steering STILL visibly works (NFL vocabulary appears), (d) baseline is
  still coherent English. Report the actual baseline vs steered text you observed.
- Report: confirmed root cause, exact sampler change (top-p value, penalty value/scheme), the
  before/after observed text, test results, and confirmation the alpha-0==baseline invariant holds.
