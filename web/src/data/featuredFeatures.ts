/**
 * featuredFeatures.ts — the hand-picked set that seeds the SAE tab's landing
 * gallery (S10). This is the shop window: a first-time visitor who hasn't run
 * anything sees a grid of legible features they can open (S8 feature page) or
 * steer with (S9 playground).
 *
 * How this list was chosen (curate from EVIDENCE, not guesses):
 *   - Every id below has `hasDashboard: true` and a `high` (or, for 21934,
 *     `medium`) label confidence in dashboards/index.json.
 *   - Each one's top max-activating examples were opened and eyeballed for
 *     coherence before inclusion — dead/rare features, muddy-polysemantic
 *     high-frequency ones, and anything with broken/garbage examples were
 *     rejected (e.g. 4715 language-names had broken-unicode examples; 13215
 *     legal-citations peaked on function words; 13481/5449 had junk logit
 *     lenses; 22588/21334 were coherent but too grim for a landing page).
 *   - The set is a deliberate MIX: topical/semantic features (sports,
 *     philosophy, history, politics, law, health, finance, geography, register)
 *     plus structural/linguistic ones (punctuation, a morphological compound
 *     feature, a grammatical superlative feature) to show the SAE's range.
 *
 * The gallery is data-driven: it loads each feature's real label, firing rate
 * and strongest example at runtime via `loadFeature` (featurePage.ts). Only the
 * id and an optional human `blurb` live here, so editing the shop window is a
 * one-line change. `blurb` overrides the auto-generated label on the card when
 * that label is clunky or narrower than what the feature actually does.
 */
export interface FeaturedFeature {
  /** SAE feature id (0..24575). Must have a dashboard for the example taste. */
  id: number
  /**
   * Optional human title/note shown on the card instead of the auto-label,
   * for features whose auto-generated label is weak, long, or too narrow.
   */
  blurb?: string
}

/**
 * ~13 curated features, roughly topical-first then structural. Order here is
 * the display order in the grid.
 */
export const FEATURED_FEATURES: FeaturedFeature[] = [
  // --- topical / semantic ---
  { id: 9127, blurb: 'NFL and pro football' }, // auto: "Sports rankings and superlatives in NFL context"
  { id: 11270, blurb: 'Philosophy and abstract reasoning' }, // auto: "Philosophical discourse and abstract reasoning"
  { id: 19948 }, // "American historical events and dates"
  { id: 9025, blurb: 'UK politics and constituencies' }, // auto: "UK parliamentary constituency names and locations"
  { id: 16656 }, // "Legal/formal proceedings and court contexts"
  { id: 13265, blurb: 'US health agencies — in practice, the CDC' }, // auto: "Government agency or research center names"
  { id: 12628 }, // "Large monetary amounts in dollars or currency"
  { id: 24063 }, // "Country names in alphabetical lists"
  { id: 13976 }, // "South Asian place names and demonyms"
  { id: 23688 }, // "Archaic or poetic language markers"

  // --- structural / linguistic ---
  { id: 21934 }, // "Commas and conjunctions in formal prose" (syntactic, not topical)
  { id: 14498, blurb: 'Hyphenated compound adjectives' }, // auto: "...or adjectival suffixes"
  { id: 14888 }, // "Superlative adjectives after 'most'"
]
