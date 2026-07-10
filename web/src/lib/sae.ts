/**
 * sae.ts — in-browser sparse-autoencoder encoder for GPT-2 layer-8 resid_pre.
 *
 * The SAE decomposes GPT-2's internal state at a layer into ~24k features, most
 * of which fire rarely and, when they do, on a human-recognisable thing (a kind
 * of punctuation, a topic, a syntactic role). This module runs the encoder in
 * the browser so the SAE tab can show which features light up on each token.
 *
 * The encoder graph (`sae_enc_fp16.onnx`, exported in S3) bakes in the
 * StandardSAE encode:
 *     feats = ReLU((x - b_dec) @ W_enc + b_enc)
 * but it expects `x` in the SAE's *centered* basis. Per BASIS_CONTRACT.md the
 * one transform the browser must apply first is: subtract, per token, the mean
 * over the 768 d_model dims. No scale, no LayerNorm. Skipping it does not error
 * — it silently produces garbage (L0 explodes, features fire on noise) — so
 * `centerRows` runs on every encode.
 *
 * fp16 hazard: like the model pipeline's fp16 graphs, this encoder returns NaN
 * on a WebGPU adapter that lacks the `shader-f16` feature. We therefore run it
 * on the SAME execution provider the model runner picked (see runner.ts
 * `detectBackend`, which already gates WebGPU on shader-f16) — WASM on adapters
 * without it — and never force the encoder onto WebGPU on its own.
 */
import * as ort from 'onnxruntime-web'
// Same Vite-fingerprinted ORT wasm binaries the model runner uses. ORT reads
// `ort.env.wasm.wasmPaths` at session-create time; the runner sets it too, but
// we set it here as well so the encoder is correct even if it is created first.
import ortWasmThreaded from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import ortWasmThreadedMjs from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import ortWasmJsep from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url'
import ortWasmJsepMjs from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url'
import type { BackendChoice } from './runner'

export const D_MODEL = 768
export const D_SAE = 24576

/**
 * Base URL for the SAE artifacts.
 *   dev  — `/sae`, served from D:/dev/sae-artifacts/L8 by the Vite middleware.
 *   prod — the HF dataset resolve URL, wired into deploy.yml by Zack, e.g.
 *          https://huggingface.co/datasets/<user>/interp-sae-gpt2-L8/resolve/main/L8
 * The app fetches `${base}/sae_enc_fp16.onnx`, `${base}/dashboards/manifest.json`
 * (version pointer) and `${base}/dashboards/index.json` (feature labels).
 */
export const SAE_BASE_URL: string =
  (import.meta.env.VITE_SAE_BASE_URL as string | undefined) ?? '/sae'

const ENCODER_FILE = 'sae_enc_fp16.onnx'
const MANIFEST_URL = `${SAE_BASE_URL}/dashboards/manifest.json`
const INDEX_URL = `${SAE_BASE_URL}/dashboards/index.json`
const ENCODER_URL = `${SAE_BASE_URL}/${ENCODER_FILE}`

// ---------------------------------------------------------------------------
// Feature metadata / results
// ---------------------------------------------------------------------------

export interface FeatureMeta {
  id: number
  /** Fraction of tokens on which this feature fires (from the harvest). */
  freq: number
  /** Curated plain-language label, or null for the (majority) unlabeled ones. */
  label: string | null
  labelConfidence: 'high' | 'medium' | 'low' | null
  /** True if a full S8 dashboard exists for this feature. */
  hasDashboard: boolean
}

/** One active feature on one token: id, activation value, resolved label. */
export interface FeatureHit {
  id: number
  value: number
  /** Real curated label, or `feature #<id>` for the unlabeled majority. */
  label: string
  /** True only when a curated label exists (drives styling). */
  labeled: boolean
  labelConfidence: 'high' | 'medium' | 'low' | null
  freq: number
  hasDashboard: boolean
}

/** Per token, the top-k active features sorted by descending activation. */
export type TokenTopK = FeatureHit[]

interface IndexFeature {
  freq: number
  chunk: number | null
  hasDashboard: boolean
  label?: string
  label_confidence?: 'high' | 'medium' | 'low'
}
interface SaeIndex {
  layer: number
  sae_release: string
  d_sae: number
  total_nonbos_tokens: number
  histogram_bins: number[]
  features: Record<string, IndexFeature>
}
interface DashboardsManifest {
  layer: number
  sae_release: string
  d_sae: number
  content_hash?: string
}

// ---------------------------------------------------------------------------
// The S2 basis transform (centering) — pure, unit-tested
// ---------------------------------------------------------------------------

/**
 * The one line the browser must obey (BASIS_CONTRACT.md): mean-center each
 * token's residual vector over the d_model dimension. No scale, no LayerNorm.
 *
 * Returns a NEW Float32Array and leaves the input untouched — the model runner
 * reuses its residual buffers elsewhere, so we must not mutate them.
 */
export function centerRows(
  x: Float32Array,
  seq: number,
  d = D_MODEL,
): Float32Array {
  if (x.length !== seq * d)
    throw new Error(`centerRows: length ${x.length} != seq*${d} (${seq * d})`)
  const out = new Float32Array(seq * d)
  for (let t = 0; t < seq; t++) {
    const base = t * d
    let sum = 0
    for (let i = 0; i < d; i++) sum += x[base + i]
    const mean = sum / d
    for (let i = 0; i < d; i++) out[base + i] = x[base + i] - mean
  }
  return out
}

// ---------------------------------------------------------------------------
// Top-k per token — pure, unit-tested
// ---------------------------------------------------------------------------

/**
 * Per-token top-k active features from a flat feats buffer [1, seq, D_SAE].
 * Allocation-light: keeps a k-sized scratch buffer per token (k is tiny).
 * Feats are a ReLU output, so a value of 0 means inactive and is skipped.
 *
 * Optional `resolve` attaches curated labels/metadata; without it every hit is
 * reported as an unlabeled `feature #<id>` (that path is what the unit test
 * pins, and it is also correct at runtime before labels have loaded).
 */
export function topKPerToken(
  feats: Float32Array,
  seq: number,
  k: number,
  resolve?: (id: number) => FeatureMeta,
): TokenTopK[] {
  const out: TokenTopK[] = []
  for (let t = 0; t < seq; t++) {
    const base = t * D_SAE
    const idxBuf: number[] = []
    const valBuf: number[] = []
    for (let f = 0; f < D_SAE; f++) {
      const v = feats[base + f]
      if (v <= 0) continue // inactive
      if (idxBuf.length < k) {
        idxBuf.push(f)
        valBuf.push(v)
      } else {
        let minI = 0
        for (let j = 1; j < valBuf.length; j++)
          if (valBuf[j] < valBuf[minI]) minI = j
        if (v > valBuf[minI]) {
          idxBuf[minI] = f
          valBuf[minI] = v
        }
      }
    }
    const hits: FeatureHit[] = idxBuf.map((id, j) => {
      const meta = resolve?.(id)
      const label = meta?.label ?? null
      return {
        id,
        value: valBuf[j],
        label: label ?? `feature #${id}`,
        labeled: label != null,
        labelConfidence: meta?.labelConfidence ?? null,
        freq: meta?.freq ?? 0,
        hasDashboard: meta?.hasDashboard ?? false,
      }
    })
    hits.sort((a, b) => b.value - a.value)
    out.push(hits)
  }
  return out
}

// ---------------------------------------------------------------------------
// Cache-API versioning (mirrors runner.ts::openVersionedCache)
// ---------------------------------------------------------------------------

const CACHE_PREFIX = 'interp-sae'

/**
 * Open the encoder cache, namespaced by the dashboards manifest content hash so
 * a re-exported artifact evicts the stale one instead of being shadowed forever
 * by whatever the Cache API stored first. Stale versions are deleted.
 */
export async function openVersionedCache(version: string): Promise<Cache | null> {
  if (!('caches' in globalThis)) return null
  const name = `${CACHE_PREFIX}-${version}`
  try {
    const keys = await caches.keys()
    await Promise.all(
      keys
        .filter((k) => k.startsWith(`${CACHE_PREFIX}-`) && k !== name)
        .map((k) => caches.delete(k)),
    )
    return await caches.open(name)
  } catch {
    return null
  }
}

export async function fetchManifestVersion(): Promise<string> {
  try {
    // no-cache: the manifest is the version pointer, so it must revalidate.
    const res = await fetch(MANIFEST_URL, { cache: 'no-cache' })
    if (!res.ok) return 'unversioned'
    const m = (await res.json()) as DashboardsManifest
    return m.content_hash ?? 'unversioned'
  } catch {
    return 'unversioned'
  }
}

export interface SaeLoadProgress {
  loadedBytes: number
  totalBytes: number
  /** true once the graph is downloaded and the session is instantiating. */
  instantiating: boolean
}

/** Fetch the encoder graph, preferring the Cache API; stream for progress. */
async function fetchEncoder(
  cache: Cache | null,
  onProgress?: (p: SaeLoadProgress) => void,
): Promise<ArrayBuffer> {
  if (cache) {
    const hit = await cache.match(ENCODER_URL)
    if (hit) {
      const buf = await hit.arrayBuffer()
      onProgress?.({
        loadedBytes: buf.byteLength,
        totalBytes: buf.byteLength,
        instantiating: true,
      })
      return buf
    }
  }

  const res = await fetch(ENCODER_URL)
  if (!res.ok)
    throw new Error(
      `fetch ${ENCODER_URL} failed: ${res.status} ${res.statusText}`,
    )
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
      onProgress?.({
        loadedBytes: loaded,
        totalBytes: total || loaded,
        instantiating: false,
      })
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
    onProgress?.({
      loadedBytes: buf.byteLength,
      totalBytes: buf.byteLength,
      instantiating: false,
    })
  }

  if (cache) {
    try {
      await cache.put(
        ENCODER_URL,
        new Response(buf.slice(0), {
          headers: { 'Content-Type': 'application/octet-stream' },
        }),
      )
    } catch {
      /* quota / private mode — non-fatal */
    }
  }
  onProgress?.({
    loadedBytes: buf.byteLength,
    totalBytes: buf.byteLength,
    instantiating: true,
  })
  return buf
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/**
 * Holds the instantiated encoder session + the feature label index. Created
 * once via `SaeEncoder.create`, reused for every `encode`.
 */
export class SaeEncoder {
  private constructor(
    /** The EP the encoder actually runs on (matches the model runner). */
    readonly ep: Backend,
    private readonly session: ort.InferenceSession,
    private readonly index: SaeIndex | null,
    private readonly inputName: string,
    private readonly outputName: string,
  ) {}

  static async create(
    backend: BackendChoice,
    onProgress?: (p: SaeLoadProgress) => void,
  ): Promise<SaeEncoder> {
    ort.env.wasm.wasmPaths =
      backend.ep === 'webgpu'
        ? { wasm: ortWasmJsep, mjs: ortWasmJsepMjs }
        : { wasm: ortWasmThreaded, mjs: ortWasmThreadedMjs }

    const version = await fetchManifestVersion()
    const cache = await openVersionedCache(version)
    const buf = await fetchEncoder(cache, onProgress)

    // Same EP the model picked. detectBackend only returns 'webgpu' when the
    // adapter has shader-f16, so the fp16 graph is numerically safe there; on
    // this box (GTX 1070, no shader-f16) the runner picks 'wasm' and so do we.
    const session = await ort.InferenceSession.create(new Uint8Array(buf), {
      executionProviders: backend.ep === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
      graphOptimizationLevel: 'all',
    })

    // Labels are best-effort: a missing/failed index just renders every feature
    // as `feature #<id>`. Never let it break the encoder.
    let index: SaeIndex | null = null
    try {
      const res = await fetch(INDEX_URL, { cache: 'no-cache' })
      if (res.ok) index = (await res.json()) as SaeIndex
    } catch {
      /* labels are optional */
    }

    return new SaeEncoder(
      backend.ep,
      session,
      index,
      session.inputNames[0] ?? 'x',
      session.outputNames[0] ?? 'feats',
    )
  }

  /** Whether curated feature labels loaded. */
  get hasLabels(): boolean {
    return this.index != null
  }

  /** Metadata for a feature id (unlabeled → label null, freq 0). */
  labelFor = (id: number): FeatureMeta => {
    const f = this.index?.features?.[String(id)]
    return {
      id,
      freq: f?.freq ?? 0,
      label: f?.label ?? null,
      labelConfidence: f?.label_confidence ?? null,
      hasDashboard: f?.hasDashboard ?? false,
    }
  }

  /**
   * Encode a layer-8 resid_pre buffer (flat [1, seq, 768], from the model
   * runner's `residualsEnteringBlocks(ids)[8]`) into feats (flat [1, seq,
   * 24576]). Applies the S2 centering transform, then runs the graph.
   */
  async encode(residLayer8: Float32Array, seq: number): Promise<Float32Array> {
    const x = centerRows(residLayer8, seq, D_MODEL)
    const input = new ort.Tensor('float32', x, [1, seq, D_MODEL])
    const out = await this.session.run({ [this.inputName]: input })
    const feats = out[this.outputName] as ort.Tensor
    return feats.data as Float32Array
  }

  /** Convenience: encode + top-k with labels attached, in one call. */
  topK(feats: Float32Array, seq: number, k: number): TokenTopK[] {
    return topKPerToken(feats, seq, k, this.labelFor)
  }
}

type Backend = BackendChoice['ep']
