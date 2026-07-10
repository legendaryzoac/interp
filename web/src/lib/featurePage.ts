/**
 * featurePage.ts — loads the per-feature dashboard that powers the S8 feature
 * page (label, max-activating examples, activation histogram, logit lens).
 *
 * S7's `sae.ts` already fetches `dashboards/index.json` for the live inspector's
 * labels. This module reuses that same index (memoized here) to resolve which
 * chunk a feature lives in, then fetches `dashboards/features_{chunk:0000}.json`
 * — a 256- or 128-feature array — caching it with the SAME versioned Cache API
 * pattern the encoder uses (namespaced on the dashboards `content_hash`, so a
 * re-export evicts the stale copy). Chunks are memoized per session, so opening
 * several features from one chunk is a single fetch.
 *
 * Only ~384 of the 24576 features have a dashboard (`hasDashboard`, non-null
 * `chunk`). For every other feature we return a minimal stub — its label (if
 * any) and firing rate from the index — and NEVER fetch a null chunk. Missing
 * data never throws; the page degrades to the minimal view.
 */
import {
  SAE_BASE_URL,
  fetchManifestVersion,
  openVersionedCache,
} from './sae'

const INDEX_URL = `${SAE_BASE_URL}/dashboards/index.json`

function chunkUrl(chunk: number): string {
  return `${SAE_BASE_URL}/dashboards/features_${String(chunk).padStart(4, '0')}.json`
}

// ---------------------------------------------------------------------------
// Public view types (camelCase; the on-disk dashboard is snake_case)
// ---------------------------------------------------------------------------

export interface FeatureHistogram {
  /** 21 log-spaced activation bin edges. */
  bins: number[]
  /** 20 token counts, one per bin. */
  counts: number[]
}

export interface FeatureExample {
  /** GPT-2 BPE token strings in leading-space form, e.g. " the". */
  tokens: string[]
  /** uint8 per token; decode with `act = acts[i]/255 * maxAct`. */
  acts: number[]
  /** This example's peak activation (the value acts=255 decodes to). */
  maxAct: number
  /** Index of the peak token within `tokens`. */
  actIndex: number
}

export interface LogitLensPair {
  token: string
  weight: number
}

export type LabelConfidence = 'high' | 'medium' | 'low'

interface FeatureBase {
  id: number
  label: string | null
  labelConfidence: LabelConfidence | null
  /** Fraction of tokens on which the feature fires. */
  freq: number
}

/** A feature with a full curated dashboard. */
export interface FullFeature extends FeatureBase {
  kind: 'full'
  /** Peak activation seen across the harvest. */
  maxAct: number
  /** Number of tokens the feature fired on in the harvest. */
  nActive: number
  histogram: FeatureHistogram
  /** Up to 12 max-activating examples, strongest first. */
  examples: FeatureExample[]
  /** Tokens this feature pushes the next-token prediction toward. */
  promoted: LogitLensPair[]
  /** Tokens it pushes away from. */
  suppressed: LogitLensPair[]
}

/** A feature with no dashboard — only index-level metadata is known. */
export interface MinimalFeature extends FeatureBase {
  kind: 'minimal'
}

export type FeatureView = FullFeature | MinimalFeature

// ---------------------------------------------------------------------------
// On-disk shapes (snake_case, as served)
// ---------------------------------------------------------------------------

interface IndexFeature {
  freq: number
  chunk: number | null
  hasDashboard: boolean
  label?: string
  label_confidence?: LabelConfidence
}
interface SaeIndex {
  features: Record<string, IndexFeature>
}
interface RawExample {
  tokens: string[]
  acts: number[]
  max_act: number
  act_index: number
}
interface RawDashboardFeature {
  id: number
  freq: number
  max_act: number
  n_active: number
  label?: string
  label_confidence?: LabelConfidence
  histogram: FeatureHistogram
  top_examples: RawExample[]
  logit_lens: { promoted: [string, number][]; suppressed: [string, number][] }
}

// ---------------------------------------------------------------------------
// Pure helpers — unit-tested
// ---------------------------------------------------------------------------

/**
 * Decode a per-token uint8 activation strip back to real activations:
 *   act = acts[i] / 255 * maxAct
 * (The encoder quantized each example's activations against its own peak, so
 * the peak token — acts = 255 — decodes to exactly `maxAct`.)
 */
export function decodeActs(
  acts: ArrayLike<number>,
  maxAct: number,
): Float32Array {
  const out = new Float32Array(acts.length)
  const s = maxAct / 255
  for (let i = 0; i < acts.length; i++) out[i] = acts[i] * s
  return out
}

export interface HistBar {
  /** Left activation edge of this bin. */
  x0: number
  /** Right activation edge. */
  x1: number
  /** Token count in this bin. */
  count: number
}

/**
 * Turn the shared-log-bin histogram ({bins:[21], counts:[20]}) into 20 bars,
 * each spanning [bins[i], bins[i+1]] with height `counts[i]`. Tolerates a short
 * or missing array by clamping to the counts length.
 */
export function histogramToBars(h: FeatureHistogram | undefined): HistBar[] {
  if (!h || !h.counts || !h.bins) return []
  const n = Math.min(h.counts.length, Math.max(0, h.bins.length - 1))
  const bars: HistBar[] = []
  for (let i = 0; i < n; i++) {
    bars.push({ x0: h.bins[i], x1: h.bins[i + 1], count: h.counts[i] })
  }
  return bars
}

// ---------------------------------------------------------------------------
// Index + chunk loading (memoized)
// ---------------------------------------------------------------------------

let indexPromise: Promise<SaeIndex | null> | null = null
let versionPromise: Promise<string> | null = null
const chunkPromises = new Map<number, Promise<Map<number, RawDashboardFeature>>>()

/** Reset all module-level memoization. Test-only. */
export function __resetFeaturePageCache(): void {
  indexPromise = null
  versionPromise = null
  chunkPromises.clear()
}

function getIndex(): Promise<SaeIndex | null> {
  if (!indexPromise) {
    indexPromise = fetch(INDEX_URL, { cache: 'no-cache' })
      .then((res) => (res.ok ? (res.json() as Promise<SaeIndex>) : null))
      .catch(() => null)
  }
  return indexPromise
}

function getVersion(): Promise<string> {
  if (!versionPromise) versionPromise = fetchManifestVersion()
  return versionPromise
}

/**
 * Fetch (or cache-hit) one dashboard chunk and index it by feature id. Memoized
 * per chunk for the session, so several features in the same chunk share one
 * fetch. A failed fetch rejects the memoized promise — callers treat that as a
 * graceful minimal page.
 */
function loadChunk(chunk: number): Promise<Map<number, RawDashboardFeature>> {
  const existing = chunkPromises.get(chunk)
  if (existing) return existing

  const p = (async () => {
    const url = chunkUrl(chunk)
    const version = await getVersion()
    const cache = await openVersionedCache(version)

    let arr: RawDashboardFeature[] | null = null
    if (cache) {
      const hit = await cache.match(url)
      if (hit) arr = (await hit.json()) as RawDashboardFeature[]
    }
    if (!arr) {
      const res = await fetch(url)
      if (!res.ok)
        throw new Error(`fetch ${url} failed: ${res.status} ${res.statusText}`)
      if (cache) {
        try {
          await cache.put(url, res.clone())
        } catch {
          /* quota / private mode — non-fatal */
        }
      }
      arr = (await res.json()) as RawDashboardFeature[]
    }

    const map = new Map<number, RawDashboardFeature>()
    for (const f of arr) map.set(f.id, f)
    return map
  })()

  chunkPromises.set(chunk, p)
  return p
}

// ---------------------------------------------------------------------------
// loadFeature
// ---------------------------------------------------------------------------

/**
 * Resolve a feature id to a full or minimal view. Never fetches a null chunk;
 * never throws on a feature that is missing from the index or its chunk.
 */
export async function loadFeature(id: number): Promise<FeatureView> {
  const index = await getIndex()
  const meta = index?.features?.[String(id)]

  const base: FeatureBase = {
    id,
    label: meta?.label ?? null,
    labelConfidence: meta?.label_confidence ?? null,
    freq: meta?.freq ?? 0,
  }

  // No index entry, no dashboard, or a null chunk → minimal page, no fetch.
  if (!meta || meta.chunk == null || !meta.hasDashboard) {
    return { ...base, kind: 'minimal' }
  }

  let raw: RawDashboardFeature | undefined
  try {
    const map = await loadChunk(meta.chunk)
    raw = map.get(id)
  } catch {
    // Chunk fetch failed — fall back to the minimal page rather than throw.
    return { ...base, kind: 'minimal' }
  }
  if (!raw) return { ...base, kind: 'minimal' }

  return {
    kind: 'full',
    id,
    // Prefer the chunk's own label; fall back to the index's.
    label: raw.label ?? base.label,
    labelConfidence: raw.label_confidence ?? base.labelConfidence,
    freq: raw.freq ?? base.freq,
    maxAct: raw.max_act,
    nActive: raw.n_active,
    histogram: raw.histogram,
    examples: (raw.top_examples ?? []).map((e) => ({
      tokens: e.tokens,
      acts: e.acts,
      maxAct: e.max_act,
      actIndex: e.act_index,
    })),
    promoted: (raw.logit_lens?.promoted ?? []).map(([token, weight]) => ({
      token,
      weight,
    })),
    suppressed: (raw.logit_lens?.suppressed ?? []).map(([token, weight]) => ({
      token,
      weight,
    })),
  }
}
