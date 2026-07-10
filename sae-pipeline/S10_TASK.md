# S10 — Featured Gallery (Epic C, story 4 — the landing payoff)

You own `C:/Users/Zack/ClaudeCode/interp/web/`. Do NOT touch `model-pipeline/`, `sae-pipeline/`,
`infra/`, or root `package.json`. No git. Dev server `interp-web` (port 5273) exists. This is the
LAST story of the SAE epic.

## What S10 delivers
A curated **featured gallery** on the SAE tab's landing (before the user has typed/run anything) so a
first-time visitor gets immediate payoff: a grid of ~10–15 hand-picked, legible features, each a card
showing its label + a taste of its top example, that a visitor can click to jump straight into either
its Feature Page (S8) or the Steering Playground (S9). It turns "here's a blank prompt box" into
"here are the coolest things this model learned — go look."

## Read first (all exist)
- `web/src/components/SaeInspector.tsx` — the SAE tab. The gallery should show on the SAE landing
  (when there's no active per-token result yet) and get out of the way once the user runs a prompt.
- `web/src/components/FeaturePage.tsx` — clicking a gallery card should open this (reuse the existing
  open-feature flow / `onFeatureClick`).
- `web/src/components/SteeringPlayground.tsx` + `App.tsx`'s `handleSteer`/`onSteer` seam — a card's
  "Steer" affordance routes here with the feature preselected (reuse, don't rebuild).
- `web/src/lib/sae.ts` + `featurePage.ts` — label/index loading + chunk loading. Gallery cards need
  label + freq (from index.json) and optionally a top example snippet (from the feature's dashboard
  chunk). Reuse the memoized loaders; don't refetch.
- The site's plain, un-hyped voice (recently de-AI-ified). No hype, no emoji.

## Curation — start from EVIDENCE, not guesses
S9 already identified features that steer strongly AND have legible labels — use these as the spine of
the gallery (all have `hasDashboard:true`, chunk 0 or 1):
- **9127** — NFL / pro football (quarterbacks, rookies, the offseason)
- **11270** — Philosophy (philosophers, empirical, conceptual reasoning)
- **19948** — Early-1900s history
- **9025** — UK politics
- **21934** — Commas and conjunctions in formal prose (a good "syntactic, not topical" example)
Then EXPAND to ~10–15 by browsing `dashboards/index.json` (features with `hasDashboard:true` and a
`label_confidence` of "high", ~176 of them) and `features_0000.json`/`features_0001.json`. Pick a MIX
that shows range: topical/semantic (sports, philosophy, history, politics, maybe science/place/finance
if present), plus a couple of structural/linguistic ones (punctuation, discourse markers, a
morphological one like the hyphenated-compound feature 14498). Prefer high-confidence labels whose top
examples you can eyeball as coherent. AVOID: dead/rare features, polysemantic high-frequency ones with
muddy examples, and anything whose label reads awkwardly. You must actually open the candidates and
sanity-check their examples before including them — curate hard, this is the shop window.

Store the curated set as a small committed config (e.g. `src/data/featuredFeatures.ts`) — an array of
`{ id, blurb? }` where `blurb` is an optional one-line human note if the auto-label is weak. Keep the
list data-driven so it's easy to edit.

## Build
1. `src/data/featuredFeatures.ts` — the curated id list (with any override blurbs).
2. `src/components/FeatureGallery.tsx` — a responsive card grid. Each card: the feature's label
   (or blurb), a small "fires on X%" / confidence hint, and a one-line taste of its strongest example
   with the peak token emphasized (pull from the dashboard chunk; lazy/graceful if not yet loaded).
   Two actions per card: open Feature Page, and "Steer" (routes to the playground preselected).
   Loading skeleton while chunks fetch; never block the whole tab on it.
3. Wire into `SaeInspector` (or the SAE landing area) so the gallery shows before a run and collapses
   (or moves below) once per-token results exist. A short plain-voice intro line above it.
4. Make sure the existing S7 inspector, S8 pages, and S9 steering all still work unchanged.

## Verify before reporting (screenshots BROKEN in this pane — use DOM/console/network)
- `npm run build -w web` clean; `npx vitest run` green (62 currently; add a small test if you add
  logic worth testing, e.g. the curated list is non-empty and ids are unique).
- Preview (`interp-web`): open the SAE tab WITHOUT running a prompt; DOM-verify the gallery renders
  ~10–15 cards with real labels and example snippets; click a card → Feature Page opens; use a card's
  "Steer" → lands in the Steering Playground with that feature selected. Confirm the gallery yields to
  the per-token view after a prompt is run. Console: no errors, no NaNs, chunk fetches 200.
- Mobile 375px: cards reflow to 1–2 columns, no horizontal overflow (`scrollWidth === clientWidth`).
- Report: the final curated feature list (ids + labels + why each made the cut, and any you rejected
  and why), files added/changed, and DOM-verified confirmation of card → page and card → steer.
