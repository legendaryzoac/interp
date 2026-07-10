/**
 * steering.ts — SAE feature steering for the S9 playground.
 *
 * Steering = add a scaled decoder vector to GPT-2's layer-8 residual stream and
 * continue the forward pass. Per BASIS_CONTRACT.md §S9 the add happens in the
 * RAW residual basis:
 *     resid'[pos] = resid[pos] + alpha * W_dec[f]
 * at every position. Crucially there is NO centering on the steering add — the
 * SAE encoder needs the centered basis (see sae.ts::centerRows), but the decoder
 * row lives in raw residual space, and the all-ones component centering removes
 * is a gauge GPT-2's LayerNorms ignore. `b_dec` is a constant offset that cancels
 * in the intervention, so it is not added.
 *
 * The decoder blob `${SAE_BASE}/w_dec_fp16.bin` is [24576, 768] little-endian
 * fp16, C-contiguous row-major (feature f's row = bytes f*1536 .. (f+1)*1536).
 * The rows are unit-normalised (‖W_dec[f]‖ ≈ 1), so `alpha` is directly the L2
 * magnitude of the nudge; features fire up to ~7-12 in the harvest, which sets
 * the natural steering scale.
 */
import {
  SAE_BASE_URL,
  fetchManifestVersion,
  openVersionedCache,
} from './sae'

export const D_MODEL = 768
export const D_SAE = 24576

/** Bytes per decoder row: 768 fp16 halves. */
const ROW_BYTES = D_MODEL * 2
const W_DEC_URL = `${SAE_BASE_URL}/w_dec_fp16.bin`

// ---------------------------------------------------------------------------
// fp16 → fp32 (pure, unit-tested)
// ---------------------------------------------------------------------------

/**
 * Decode one IEEE-754 half-precision value (given as its 16 raw bits) to a JS
 * number. Handles subnormals, ±Inf and NaN. Used to upcast the fp16 decoder
 * blob — `Float16Array` isn't available everywhere, and one row (768 halves) is
 * decoded rarely (once per feature selection), so a plain loop is plenty.
 */
export function halfToFloat(h: number): number {
  const sign = (h & 0x8000) >> 15
  const exp = (h & 0x7c00) >> 10
  const frac = h & 0x03ff
  let val: number
  if (exp === 0) {
    // subnormal (or zero): no implicit leading 1
    val = Math.pow(2, -14) * (frac / 1024)
  } else if (exp === 0x1f) {
    val = frac ? NaN : Infinity
  } else {
    val = Math.pow(2, exp - 15) * (1 + frac / 1024)
  }
  return sign ? -val : val
}

/**
 * Decode feature `featureId`'s decoder row out of the full fp16 blob into a
 * fresh Float32Array[768]. Little-endian, row-major. Throws if the row would
 * run past the buffer (guards a bad feature id / truncated download).
 */
export function decodeDecoderRow(
  buf: ArrayBuffer,
  featureId: number,
  d = D_MODEL,
): Float32Array {
  if (featureId < 0 || !Number.isInteger(featureId))
    throw new Error(`bad featureId ${featureId}`)
  const byteOffset = featureId * d * 2
  if (byteOffset + d * 2 > buf.byteLength)
    throw new Error(
      `decoder row ${featureId} out of range (blob is ${buf.byteLength} bytes, ` +
        `need ${byteOffset + d * 2})`,
    )
  const dv = new DataView(buf)
  const out = new Float32Array(d)
  for (let i = 0; i < d; i++) {
    out[i] = halfToFloat(dv.getUint16(byteOffset + i * 2, /* littleEndian */ true))
  }
  return out
}

// ---------------------------------------------------------------------------
// The steering add (pure, unit-tested)
// ---------------------------------------------------------------------------

/**
 * Return a NEW residual with `alpha * vec` added to every position's 768-vector.
 * `resid` is a flat [1, seq, d] buffer; `vec` is one decoder row [d]. The input
 * is never mutated (the runner reuses its residual buffers). With `alpha === 0`
 * this is an exact clone of `resid` — so the alpha-0 steered path is bit-for-bit
 * the baseline path, which is exactly the invariant the playground relies on.
 */
export function addSteeringVector(
  resid: Float32Array,
  vec: Float32Array,
  alpha: number,
  seq: number,
  d = D_MODEL,
): Float32Array {
  if (resid.length !== seq * d)
    throw new Error(`resid length ${resid.length} != seq*${d} (${seq * d})`)
  if (vec.length !== d)
    throw new Error(`steering vec length ${vec.length} != ${d}`)
  const out = new Float32Array(resid) // clone; baseline == clone when alpha 0
  if (alpha === 0) return out
  for (let t = 0; t < seq; t++) {
    const base = t * d
    for (let i = 0; i < d; i++) out[base + i] += alpha * vec[i]
  }
  return out
}

// ---------------------------------------------------------------------------
// Sampling from a single next-token logit row (pure, unit-tested)
// ---------------------------------------------------------------------------

/** True iff at least one logit is finite — a cheap NaN/degeneracy guard. */
export function logitsAreFinite(logits: Float32Array): boolean {
  for (let v = 0; v < logits.length; v++) if (Number.isFinite(logits[v])) return true
  return false
}

/**
 * Pick the next token from a logit row.
 *   temperature <= 0  → greedy argmax (deterministic; NaNs never win a `>`).
 *   temperature  > 0  → temperature-softmax sample using `rng()` in [0,1).
 * Numerically stable (subtract max before exp).
 */
export function sampleToken(
  logits: Float32Array,
  temperature: number,
  rng: () => number = Math.random,
): number {
  const n = logits.length
  // max for stability / argmax
  let max = -Infinity
  let argmax = 0
  for (let v = 0; v < n; v++) {
    const x = logits[v]
    if (x > max) {
      max = x
      argmax = v
    }
  }
  if (temperature <= 0) return argmax

  // temperature softmax, single exp pass into a scratch buffer
  const probs = new Float64Array(n)
  let sum = 0
  for (let v = 0; v < n; v++) {
    const e = Math.exp((logits[v] - max) / temperature)
    probs[v] = e
    sum += e
  }
  const target = rng() * sum
  let acc = 0
  for (let v = 0; v < n; v++) {
    acc += probs[v]
    if (acc >= target) return v
  }
  return argmax // numerical fallthrough
}

/**
 * A macrotask yield that lets the browser paint the just-streamed token but is
 * NOT clamped the way `setTimeout(…, 0)` is in a hidden/background tab (Chrome
 * "intensive throttling" limits background timers to ~1/minute, which would
 * freeze streaming if the user tabs away). A MessagePort message is exempt from
 * that throttling — this is the same trick React's scheduler uses. Falls back to
 * setTimeout where MessageChannel is unavailable.
 */
export function yieldToUI(): Promise<void> {
  if (typeof MessageChannel !== 'undefined') {
    return new Promise<void>((resolve) => {
      const ch = new MessageChannel()
      ch.port1.onmessage = () => {
        ch.port1.close()
        resolve()
      }
      ch.port2.postMessage(0)
    })
  }
  return new Promise<void>((resolve) => setTimeout(resolve, 0))
}

/**
 * mulberry32 — a tiny seeded PRNG so a temperature run is reproducible and the
 * baseline / steered streams can share a seed ("identical sampling path").
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return function () {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// ---------------------------------------------------------------------------
// Decoder blob loader (mirrors sae.ts encoder loading + versioned cache)
// ---------------------------------------------------------------------------

export interface SteeringLoadProgress {
  loadedBytes: number
  totalBytes: number
  /** true once the blob is downloaded and rows are decodable. */
  ready: boolean
}

async function fetchDecoderBlob(
  cache: Cache | null,
  onProgress?: (p: SteeringLoadProgress) => void,
): Promise<ArrayBuffer> {
  if (cache) {
    const hit = await cache.match(W_DEC_URL)
    if (hit) {
      const buf = await hit.arrayBuffer()
      onProgress?.({ loadedBytes: buf.byteLength, totalBytes: buf.byteLength, ready: true })
      return buf
    }
  }

  const res = await fetch(W_DEC_URL)
  if (!res.ok)
    throw new Error(`fetch ${W_DEC_URL} failed: ${res.status} ${res.statusText}`)
  const total = Number(res.headers.get('Content-Length') ?? 0)

  let buf: ArrayBuffer
  if (res.body) {
    const reader = res.body.getReader()
    const chunks: Uint8Array[] = []
    let loaded = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      loaded += value.byteLength
      onProgress?.({ loadedBytes: loaded, totalBytes: total || loaded, ready: false })
    }
    const merged = new Uint8Array(loaded)
    let off = 0
    for (const c of chunks) {
      merged.set(c, off)
      off += c.byteLength
    }
    buf = merged.buffer
  } else {
    buf = await res.arrayBuffer()
  }

  if (cache) {
    try {
      await cache.put(
        W_DEC_URL,
        new Response(buf.slice(0), {
          headers: { 'Content-Type': 'application/octet-stream' },
        }),
      )
    } catch {
      /* quota / private mode — non-fatal */
    }
  }
  onProgress?.({ loadedBytes: buf.byteLength, totalBytes: buf.byteLength, ready: true })
  return buf
}

/**
 * Holds the fp16 decoder blob and hands out per-feature decoder rows (cached).
 * Created once via `SteeringDecoder.create`; the 37.7MB blob is shared through
 * the same versioned Cache API namespace the encoder uses, so it downloads once
 * and survives reloads.
 */
export class SteeringDecoder {
  private rowCache = new Map<number, Float32Array>()

  private constructor(private readonly blob: ArrayBuffer) {}

  static async create(
    onProgress?: (p: SteeringLoadProgress) => void,
  ): Promise<SteeringDecoder> {
    const version = await fetchManifestVersion()
    const cache = await openVersionedCache(version)
    const blob = await fetchDecoderBlob(cache, onProgress)
    if (blob.byteLength < D_SAE * ROW_BYTES) {
      throw new Error(
        `decoder blob too small: ${blob.byteLength} bytes (expected ` +
          `${D_SAE * ROW_BYTES} for [${D_SAE}, ${D_MODEL}] fp16)`,
      )
    }
    return new SteeringDecoder(blob)
  }

  /** Decoder row for a feature, memoised. */
  row(featureId: number): Float32Array {
    const hit = this.rowCache.get(featureId)
    if (hit) return hit
    const r = decodeDecoderRow(this.blob, featureId)
    this.rowCache.set(featureId, r)
    return r
  }
}

// ---------------------------------------------------------------------------
// Labeled-feature index (for the playground's search + presets)
// ---------------------------------------------------------------------------

export interface LabeledFeature {
  id: number
  label: string
  labelConfidence: 'high' | 'medium' | 'low' | null
  freq: number
  hasDashboard: boolean
}

interface RawIndexFeature {
  freq: number
  chunk: number | null
  hasDashboard: boolean
  label?: string
  label_confidence?: 'high' | 'medium' | 'low'
}
interface RawIndex {
  features: Record<string, RawIndexFeature>
}

const INDEX_URL = `${SAE_BASE_URL}/dashboards/index.json`

let labeledPromise: Promise<LabeledFeature[]> | null = null

/**
 * Fetch the SAE dashboards index once and return every feature that carries a
 * curated label, for the playground's label-search picker. Memoised per session;
 * a failed fetch yields an empty list (search just returns nothing) rather than
 * throwing.
 */
export function loadLabeledFeatures(): Promise<LabeledFeature[]> {
  if (!labeledPromise) {
    labeledPromise = fetch(INDEX_URL, { cache: 'no-cache' })
      .then((res) => (res.ok ? (res.json() as Promise<RawIndex>) : null))
      .then((idx) => {
        if (!idx?.features) return []
        const out: LabeledFeature[] = []
        for (const [k, v] of Object.entries(idx.features)) {
          if (!v.label) continue
          out.push({
            id: Number(k),
            label: v.label,
            labelConfidence: v.label_confidence ?? null,
            freq: v.freq ?? 0,
            hasDashboard: v.hasDashboard ?? false,
          })
        }
        return out
      })
      .catch(() => [])
  }
  return labeledPromise
}

/** Case-insensitive label substring search, ranked by dashboard + frequency. */
export function searchLabeledFeatures(
  all: LabeledFeature[],
  query: string,
  limit = 30,
): LabeledFeature[] {
  const q = query.trim().toLowerCase()
  if (!q) return []
  // numeric query → treat as a direct feature-id lookup as well
  const asId = /^\d+$/.test(q) ? Number(q) : null
  const hits = all.filter(
    (f) => f.label.toLowerCase().includes(q) || (asId != null && f.id === asId),
  )
  hits.sort((a, b) => {
    if (a.hasDashboard !== b.hasDashboard) return a.hasDashboard ? -1 : 1
    return b.freq - a.freq
  })
  return hits.slice(0, limit)
}
