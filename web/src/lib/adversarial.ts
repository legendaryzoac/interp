/**
 * adversarial.ts — optional adversarial-suffix loader for the Compare tab.
 *
 * If `/adversarial.json` exists it supplies the suffix chips the Compare view
 * appends to a base prompt. Like `gallery.ts`, a missing or malformed file must
 * never throw — the app just falls back to a built-in default suffix so Compare
 * is always usable.
 *
 * Schema (FROZEN — the pipeline agent writes GCG entries in this exact shape to
 * a separate file that gets merged in later, so unknown `provenance` values and
 * extra fields must be tolerated, never rejected):
 *   { suffixes: [ {
 *       id: string,
 *       text: string,                              // appended verbatim to the base prompt
 *       provenance: "curated" | "gcg" | string,    // badge; unknown values pass through
 *       note: string,
 *       target: string | null
 *   } ] }
 */

export type SuffixProvenance = 'curated' | 'gcg'

export interface AdversarialSuffix {
  id: string
  text: string
  provenance: SuffixProvenance
  note: string
  target: string | null
}

export interface AdversarialData {
  suffixes: AdversarialSuffix[]
}

/** A hard-coded fallback so Compare works even without /adversarial.json. */
export const FALLBACK_SUFFIX: AdversarialSuffix = {
  id: 'repeat-storm',
  text: ' ! ! ! ! ! ! ! ! ! !',
  provenance: 'curated',
  note: 'Built-in fallback: a run of repeated exclamation tokens.',
  target: null,
}

/** Human label for the provenance badge. Unknown values render verbatim. */
export function provenanceLabel(p: string): string {
  if (p === 'gcg') return 'GCG optimized'
  if (p === 'curated') return 'curated'
  return p
}

/** Coerce one raw entry into a well-formed suffix, or null if unusable. */
function normalizeSuffix(raw: unknown): AdversarialSuffix | null {
  if (raw == null || typeof raw !== 'object') return null
  const o = raw as Record<string, unknown>
  const id = typeof o.id === 'string' ? o.id : null
  const text = typeof o.text === 'string' ? o.text : null
  // A suffix with no text is meaningless; id is needed as a React key.
  if (!id || text == null || text.length === 0) return null
  const provenance =
    o.provenance === 'gcg' || o.provenance === 'curated'
      ? o.provenance
      : // Tolerate unknown provenance from future pipeline outputs; keep the
        // raw string so the badge still says something sensible.
        ((typeof o.provenance === 'string' ? o.provenance : 'curated') as SuffixProvenance)
  return {
    id,
    text,
    provenance,
    note: typeof o.note === 'string' ? o.note : '',
    target: typeof o.target === 'string' ? o.target : null,
  }
}

/**
 * Fetch /adversarial.json. Never throws: returns the parsed, filtered suffixes,
 * always including at least the fallback so the caller has something to show.
 * If the file is present but has extra `gcg` entries later, they flow through.
 */
export async function loadAdversarial(): Promise<AdversarialData> {
  try {
    const res = await fetch('/adversarial.json', { cache: 'no-cache' })
    if (!res.ok) return { suffixes: [FALLBACK_SUFFIX] }
    const json = (await res.json()) as unknown
    const rawList =
      json && typeof json === 'object' && Array.isArray((json as { suffixes?: unknown }).suffixes)
        ? (json as { suffixes: unknown[] }).suffixes
        : []
    const suffixes = rawList
      .map(normalizeSuffix)
      .filter((s): s is AdversarialSuffix => s != null)
    return { suffixes: suffixes.length > 0 ? suffixes : [FALLBACK_SUFFIX] }
  } catch {
    return { suffixes: [FALLBACK_SUFFIX] }
  }
}
