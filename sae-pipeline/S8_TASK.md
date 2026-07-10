# S8 — SAE Feature Pages (Epic C, story 2)

You own `C:/Users/Zack/ClaudeCode/interp/web/`. Do NOT touch `model-pipeline/`, `sae-pipeline/`,
`infra/`, or the root `package.json`. No git. Dev server `interp-web` (port 5273) already exists.

## What S8 delivers
When a user clicks a feature (from the S7 Token Inspector chip, or later the S10 gallery), open a
**full feature page**: what the feature detects (label + confidence), its strongest real-text
examples with per-token activation highlighting, its activation-frequency histogram, and the tokens
it pushes the model toward / away from (logit lens). This is what turns "feature #4078" from an
opaque id into something a visitor understands.

## Read first (all exist)
- `web/src/components/SaeInspector.tsx` and `web/src/lib/sae.ts` (from S7). SaeInspector already
  exposes an `onFeatureClick(id)` seam that currently opens a lightweight inline panel — S8 replaces
  that panel's contents with the real feature page (or routes to it). Keep S7's live inspector intact.
- `web/src/lib/sae.ts` — it already loads `index.json` (has per-feature `{freq, chunk, hasDashboard,
  label, label_confidence}`). Reuse its base-URL resolution + Cache API pattern for fetching chunks.
- Any existing D3 histogram/heatmap helpers in the codebase (the logit-lens / attention views use D3)
  — reuse the project's D3 idioms and design tokens; don't introduce a new chart lib.
- The site's plain, un-hyped copy voice (recently de-AI-ified) — match it.

## Data: the per-feature dashboard (already served at `${SAE_BASE}/dashboards/`)
`index.json.features["<id>"].chunk` gives the chunk number; fetch
`${SAE_BASE}/dashboards/features_{chunk:0000}.json` (256 or 128 features; already fetched files are
`features_0000.json` = chunk 0, `features_0001.json` = chunk 1). Each is an array of feature objects:
```
{
  id, freq, max_act, n_active, selection_reason,
  label, label_confidence,                         // added in S5
  histogram: { bins: [21 floats], counts: [20 ints] },   // shared-log-bin activation histogram
  top_examples: [ {                                  // up to 12, strongest first
    tokens: [ up to 24 token strings ],              // GPT-2 BPE strings, leading-space form e.g. " the"
    acts:   [ uint8 per token ],                     // DECODE: act = acts[i]/255 * max_act
    max_act: float,                                  // this example's peak activation
    act_index: int                                   // index of the peak token within tokens[]
  } ],
  logit_lens: { promoted: [ ["token", weight], x10 ], suppressed: [ ["token", weight], x10 ] }
}
```
Only ~384 features (`hasDashboard:true`) have a chunk + full dashboard. For a feature WITHOUT a
dashboard (chunk null / most of the 24576), show a graceful minimal page: label if any (else
"feature #id · unlabeled"), its `freq` from index.json, and a short "no detailed dashboard for this
feature" note. Never fetch a null chunk; never throw on a missing feature.

## Build
1. `src/lib/featurePage.ts` — `loadFeature(id)` → resolves the chunk from index.json, fetches +
   caches the chunk (Cache API, versioned on dashboards `content_hash` like S7), returns the feature
   object (or a minimal stub for no-dashboard features). Memoize per chunk so opening several
   features in the same chunk is one fetch. Pure helpers: `decodeActs(acts, max_act)` →
   Float32Array; `histogramToBars(histogram)`.
2. `src/components/FeaturePage.tsx`:
   - Header: label (large) + confidence pill (high/medium/low) + `feature #id` + freq ("fires on
     X% of tokens").
   - **Max-activating examples**: each example a line of its tokens with per-token background
     shading proportional to decoded activation (0 → transparent, max_act → accent). Mark/emphasize
     the peak token (`act_index`). Show whitespace markers like the token strips elsewhere (␣).
     This is the centerpiece — make it readable.
   - **Histogram**: small D3 bar chart of the activation distribution (log-bin). Reuse project D3.
   - **Logit lens**: two short columns — "pushes toward" (promoted) and "pushes away" (suppressed)
     tokens, weight-shaded. One-line plain caption explaining it.
   - A "Steer with this feature" button that is present but disabled/"coming soon" for now (S9 wires
     it) — leave an `onSteer?(id)` prop seam.
3. Wire it to the S7 inspector: clicking a feature chip opens the FeaturePage (modal/overlay or a
   routed panel — your call, but it must be dismissible and not lose the inspector state behind it).
4. Loading + error states: a spinner while the chunk fetches; a clean message if a fetch fails.

## Verify before reporting (required — screenshots are BROKEN in this pane, use DOM/console/network)
- `npm run build -w web` clean; `npx vitest run` green (unit-test `decodeActs` against a hand case
  e.g. acts=[0,128,255], max_act=8 → [0, ~4.01, 8]; and the chunk-resolution/memoization).
- Preview (`interp-web`): run a prompt in the SAE tab, click a labeled feature that HAS a dashboard
  (e.g. one whose chunk is 0/1), confirm via DOM read that the feature page renders: the label,
  ≥1 example row with shaded tokens, the histogram element, and promoted/suppressed token lists.
  Then open a feature WITHOUT a dashboard and confirm the graceful minimal page (no throw, no null
  fetch — check network shows no request for `features_null.json`). Confirm dismiss returns to the
  inspector with its tokens intact. Check console: no errors, no NaNs.
- Mobile 375px: feature page readable, example rows wrap or scroll within their container (no page
  horizontal overflow) — verify `documentElement.scrollWidth === clientWidth`.
- Report: files added/changed, how you resolved chunk→page, the decodeActs test result, DOM-verified
  description of a real feature page you opened (which feature id, its label, an example), the
  no-dashboard path, and the S9 `onSteer` seam left in place.
