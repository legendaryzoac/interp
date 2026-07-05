/**
 * runner.ts — chains the segmented GPT-2 ONNX graphs entirely client-side.
 *
 * Pipeline (frozen tensor contract, see model-pipeline/validate.py):
 *   embed.onnx      feed input_ids int64 [1, seq]  -> resid   f32 [1, seq, 768]
 *   block_XX.onnx   feed resid                      -> resid_out f32 [1, seq, 768]
 *                                                      pattern   f32 [1, 12, seq, seq]
 *   unembed.onnx    feed any residual               -> logits  f32 [1, seq, 50257]
 *
 * Chain: embed -> block_00..11 (collect pattern + resid per layer) ->
 * unembed on every layer's resid for the logit lens. The final layer's lens
 * output equals the model's actual next-token logits.
 *
 * Design notes for later milestones (prompt-comparison, activation patching):
 * a run returns the full per-layer residual stream, so a caller can re-feed a
 * (possibly patched) residual into any block or into unembed without re-running
 * from the embedding.
 */
import * as ort from 'onnxruntime-web'

// Resolve the ORT wasm binaries through Vite so they are fingerprinted, served
// same-origin (COEP-safe), and versioned exactly with the JS. We map the file
// names ORT requests to their built URLs. Both the plain threaded wasm (wasm
// EP) and the jsep build (webgpu EP) are covered.
import ortWasmThreaded from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import ortWasmThreadedMjs from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import ortWasmJsep from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url'
import ortWasmJsepMjs from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url'

export const N_LAYERS = 12
export const N_HEADS = 12
export const D_MODEL = 768
export const VOCAB_SIZE = 50257

export type Variant = 'fp32' | 'fp16' | 'int8'
export type Backend = 'webgpu' | 'wasm'

export interface BackendChoice {
  ep: Backend
  variant: Variant
  /** Human label for the badge, e.g. "WebGPU · fp16". */
  label: string
}

/** One graph to fetch + instantiate. Order matters for the chain. */
const GRAPH_NAMES = [
  'embed',
  ...Array.from({ length: N_LAYERS }, (_, i) => `block_${String(i).padStart(2, '0')}`),
  'unembed',
] as const

export interface DownloadProgress {
  /** Bytes downloaded across all graphs so far. */
  loadedBytes: number
  /** Total bytes to download (from manifest). */
  totalBytes: number
  /** Files fully downloaded. */
  loadedFiles: number
  totalFiles: number
  /** Name of the file currently downloading. */
  current: string
  /** True once every file is downloaded and served from cache/network. */
  done: boolean
}

export interface RunResult {
  seq: number
  /** Per-layer post-softmax attention, each Float32Array of [1,12,seq,seq]. */
  patterns: Float32Array[]
  /** Per-layer logit-lens logits, each [1,seq,50257]. Last === final logits. */
  lensLogits: Float32Array[]
  /** Per-layer residual stream after each block, each [1,seq,768]. */
  resids: Float32Array[]
}

export interface TopKEntry {
  token: number
  logit: number
  prob: number
}

interface Manifest {
  model_id: string
  n_layers: number
  n_heads: number
  d_model: number
  vocab_size: number
  files: Record<string, number>
  /** Content fingerprint stamped by model-pipeline/stamp_manifests.py; versions the Cache API namespace. */
  content_hash?: string
}

const MODEL_BASE_URL: string =
  (import.meta.env.VITE_MODEL_BASE_URL as string | undefined) ?? '/models'

/** Feature-detect WebGPU. Presence of navigator.gpu is the cheap gate; we also
 *  try requestAdapter so a stub (no real adapter) falls back to wasm.
 *  A `?backend=wasm|webgpu` query param forces the choice (handy for testing
 *  the fallback path on a WebGPU-capable machine). */
export async function detectBackend(): Promise<BackendChoice> {
  const params =
    typeof location !== 'undefined'
      ? new URLSearchParams(location.search)
      : new URLSearchParams()
  const forced = params.get('backend')
  // ?variant= decouples precision from EP — diagnostic only (e.g. fp16 on wasm
  // isolates "bad graphs" from "bad EP numerics"; fp32 on webgpu isolates the EP).
  const forcedVariant = (['fp32', 'fp16', 'int8'] as const).find(
    (v) => v === params.get('variant'),
  )
  if (forced === 'wasm') {
    const variant = forcedVariant ?? 'int8'
    return { ep: 'wasm', variant, label: `WASM · ${variant}` }
  }
  if (forced === 'webgpu') {
    const variant = forcedVariant ?? 'fp16'
    return { ep: 'webgpu', variant, label: `WebGPU · ${variant}` }
  }

  const gpu = (navigator as Navigator).gpu as
    | {
        requestAdapter?: () => Promise<{
          features?: { has?: (f: string) => boolean }
        } | null>
      }
    | undefined
  if (gpu && typeof gpu.requestAdapter === 'function') {
    try {
      const adapter = await gpu.requestAdapter()
      // The fp16 graphs need native f16 shaders. On adapters without the
      // shader-f16 feature (e.g. Pascal-era GPUs), every f16 compute pipeline
      // fails Dawn validation — sessions still "run" but return garbage
      // buffers, rendering a NaN-degenerate lens. An adapter alone is not
      // enough; require the feature. WASM·int8 is the honest fallback, and
      // fp32-on-WebGPU stays reachable via ?backend=webgpu&variant=fp32.
      if (adapter && adapter.features?.has?.('shader-f16')) {
        return { ep: 'webgpu', variant: 'fp16', label: 'WebGPU · fp16' }
      }
    } catch {
      /* fall through to wasm */
    }
  }
  return { ep: 'wasm', variant: 'int8', label: 'WASM · int8' }
}

const CACHE_PREFIX = 'interp-onnx'
const LEGACY_CACHE = 'interp-onnx-v1'

/** Tiny FNV-1a — enough to fingerprint a manifest for cache versioning. */
function fnv1a(s: string): string {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0).toString(16)
}

/**
 * Open the graph cache for a variant, versioned by manifest content, evicting
 * stale versions of the same variant (other variants' caches are kept — a user
 * switching backends shouldn't re-download). Without versioning, re-exported
 * artifacts are shadowed forever by whatever the Cache API stored first.
 */
async function openVersionedCache(
  variant: Variant,
  manifest: Manifest | null,
): Promise<Cache | null> {
  if (!('caches' in globalThis)) return null
  const version = manifest
    ? (manifest.content_hash ?? fnv1a(JSON.stringify(manifest)))
    : 'unversioned'
  const name = `${CACHE_PREFIX}-${variant}-${version}`
  try {
    const keys = await caches.keys()
    await Promise.all(
      keys
        .filter(
          (k) =>
            (k.startsWith(`${CACHE_PREFIX}-${variant}-`) && k !== name) ||
            k === LEGACY_CACHE,
        )
        .map((k) => caches.delete(k)),
    )
    return await caches.open(name)
  } catch {
    return null
  }
}

function graphUrl(variant: Variant, name: string): string {
  return `${MODEL_BASE_URL}/${variant}/${name}.onnx`
}

function manifestUrl(variant: Variant): string {
  return `${MODEL_BASE_URL}/${variant}/manifest.json`
}

/**
 * Fetch a single graph, preferring the Cache API (keyed on variant+filename).
 * Streams the response so we can report byte-level progress; caches the full
 * ArrayBuffer on a miss.
 */
async function fetchGraph(
  cache: Cache | null,
  url: string,
  onBytes: (delta: number) => void,
): Promise<ArrayBuffer> {
  if (cache) {
    const hit = await cache.match(url)
    if (hit) {
      const buf = await hit.arrayBuffer()
      onBytes(buf.byteLength)
      return buf
    }
  }

  const res = await fetch(url)
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status} ${res.statusText}`)

  // Stream for progress; fall back to a plain arrayBuffer if no reader.
  let buf: ArrayBuffer
  if (res.body) {
    const reader = res.body.getReader()
    const chunks: Uint8Array[] = []
    let total = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      total += value.byteLength
      onBytes(value.byteLength)
    }
    const merged = new Uint8Array(total)
    let off = 0
    for (const c of chunks) {
      merged.set(c, off)
      off += c.byteLength
    }
    buf = merged.buffer
  } else {
    buf = await res.arrayBuffer()
    onBytes(buf.byteLength)
  }

  if (cache) {
    try {
      await cache.put(
        url,
        new Response(buf.slice(0), {
          headers: { 'Content-Type': 'application/octet-stream' },
        }),
      )
    } catch {
      /* cache put can fail on quota / private mode — non-fatal */
    }
  }
  return buf
}

async function fetchManifest(variant: Variant): Promise<Manifest | null> {
  try {
    // no-cache: the manifest is the version pointer — it must always revalidate,
    // otherwise cache versioning can never see a new release.
    const res = await fetch(manifestUrl(variant), { cache: 'no-cache' })
    if (!res.ok) return null
    return (await res.json()) as Manifest
  } catch {
    return null
  }
}

/**
 * Holds the 14 instantiated sessions. Created once via `Runner.create`, reused
 * for every `run`. Sessions are the expensive part; keep this alive.
 */
export class Runner {
  private constructor(
    readonly backend: BackendChoice,
    private readonly sessions: Record<string, ort.InferenceSession>,
  ) {}

  static async create(
    backend: BackendChoice,
    onProgress: (p: DownloadProgress) => void,
  ): Promise<Runner> {
    // Configure ORT wasm paths to load the Vite-fingerprinted binaries.
    ort.env.wasm.numThreads = Math.min(
      4,
      Math.max(1, (navigator.hardwareConcurrency ?? 4) - 1),
    )
    // ORT picks exactly one wasm/mjs pair per session: the JSEP build for the
    // WebGPU EP, the plain threaded build for the WASM EP. Point at whichever
    // this backend needs (Vite-fingerprinted, same-origin).
    ort.env.wasm.wasmPaths =
      backend.ep === 'webgpu'
        ? { wasm: ortWasmJsep, mjs: ortWasmJsepMjs }
        : { wasm: ortWasmThreaded, mjs: ortWasmThreadedMjs }

    const { variant, ep } = backend

    const manifest = await fetchManifest(variant)
    const sizes = manifest?.files ?? {}
    const totalBytes = GRAPH_NAMES.reduce(
      (sum, n) => sum + (sizes[`${n}.onnx`] ?? 0),
      0,
    )

    const cache = await openVersionedCache(variant, manifest)

    const progress: DownloadProgress = {
      loadedBytes: 0,
      totalBytes,
      loadedFiles: 0,
      totalFiles: GRAPH_NAMES.length,
      current: '',
      done: false,
    }
    const emit = () => onProgress({ ...progress })

    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: ep === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
      graphOptimizationLevel: 'all',
    }

    const sessions: Record<string, ort.InferenceSession> = {}
    // Download + instantiate sequentially: keeps peak memory bounded and gives
    // a clean per-file progress story. The UI thread stays responsive because
    // every await yields.
    for (const name of GRAPH_NAMES) {
      progress.current = name
      emit()
      const buf = await fetchGraph(cache, graphUrl(variant, name), (delta) => {
        progress.loadedBytes += delta
        emit()
      })
      sessions[name] = await ort.InferenceSession.create(
        new Uint8Array(buf),
        sessionOptions,
      )
      progress.loadedFiles += 1
      emit()
    }

    progress.done = true
    progress.current = ''
    emit()

    return new Runner(backend, sessions)
  }

  /**
   * Run the full chain on a token-id sequence.
   * Collects per-layer pattern + residual, and runs the logit lens on every
   * layer's residual.
   */
  async run(tokenIds: number[]): Promise<RunResult> {
    const seq = tokenIds.length
    if (seq === 0) throw new Error('empty token sequence')

    const ids = BigInt64Array.from(tokenIds, (t) => BigInt(t))
    const inputIds = new ort.Tensor('int64', ids, [1, seq])

    // embed
    let residTensor = (await this.sessions.embed.run({ input_ids: inputIds }))
      .resid as ort.Tensor

    const patterns: Float32Array[] = []
    const resids: Float32Array[] = []

    for (let i = 0; i < N_LAYERS; i++) {
      const name = `block_${String(i).padStart(2, '0')}`
      const out = await this.sessions[name].run({ resid: residTensor })
      const residOut = out.resid_out as ort.Tensor
      const pattern = out.pattern as ort.Tensor
      resids.push(residOut.data as Float32Array)
      patterns.push(pattern.data as Float32Array)
      residTensor = residOut
    }

    // Logit lens: unembed each layer's residual. Reuse a fresh Tensor per layer
    // built on the stored Float32Array (typed-array view, no copy).
    const lensLogits: Float32Array[] = []
    for (let i = 0; i < N_LAYERS; i++) {
      const r = new ort.Tensor('float32', resids[i], [1, seq, D_MODEL])
      const out = await this.sessions.unembed.run({ resid: r })
      lensLogits.push((out.logits as ort.Tensor).data as Float32Array)
    }

    return { seq, patterns, lensLogits, resids }
  }

  /**
   * Embed-only forward: returns the residual stream ENTERING block 0
   * (token + position embeddings), flat [1, seq, 768]. This is the piece
   * `run()` never surfaced — its `resids` are per-block *outputs*, i.e. the
   * input to block L+1, so the block-0 input had no home. Activation patching
   * needs the residual entering *every* block, and block 0's entry is exactly
   * this. (Additive: run() is untouched.)
   */
  async embed(tokenIds: number[]): Promise<Float32Array> {
    const seq = tokenIds.length
    if (seq === 0) throw new Error('empty token sequence')
    const ids = BigInt64Array.from(tokenIds, (t) => BigInt(t))
    const inputIds = new ort.Tensor('int64', ids, [1, seq])
    const out = await this.sessions.embed.run({ input_ids: inputIds })
    return (out.resid as ort.Tensor).data as Float32Array
  }

  /**
   * The residual stream ENTERING each of the 12 blocks, flat [1,seq,768] each:
   *   entering[0] = embed output
   *   entering[L] = output of block L-1  (for L >= 1)
   * This is the natural coordinate for activation patching: to patch "the
   * residual entering block L at position p" you overwrite entering[L] and
   * continue the forward from block L with `continueFromBlock`.
   */
  async residualsEnteringBlocks(tokenIds: number[]): Promise<Float32Array[]> {
    const seq = tokenIds.length
    if (seq === 0) throw new Error('empty token sequence')
    const embedResid = await this.embed(tokenIds)
    const entering: Float32Array[] = [embedResid]
    let residTensor: ort.Tensor = new ort.Tensor('float32', embedResid, [
      1,
      seq,
      D_MODEL,
    ])
    // Run blocks 0..10; each output is the input to the next block. Block 11's
    // output is the final residual (fed to unembed), not an entry — so we stop
    // collecting after 12 entries (embed + outputs of blocks 0..10).
    for (let i = 0; i < N_LAYERS - 1; i++) {
      const name = `block_${String(i).padStart(2, '0')}`
      const out = await this.sessions[name].run({ resid: residTensor })
      const residOut = out.resid_out as ort.Tensor
      entering.push(residOut.data as Float32Array)
      residTensor = residOut
    }
    return entering
  }

  /**
   * Continue a forward from a residual ENTERING block `startBlock`: run blocks
   * startBlock..11, unembed, and return the final next-token logits row
   * (Float32Array of length VOCAB_SIZE) at position `seq - 1`. Used by
   * activation patching — feed a (patched) residual and read the recovered
   * logits. `resid` is a flat [1, seq, 768] buffer.
   *
   * Returning only the last row keeps the patching sweep light: the caller
   * measures a logit-diff at the final position over ~12*seq forwards.
   */
  async continueFromBlock(
    resid: Float32Array,
    startBlock: number,
    seq: number,
  ): Promise<Float32Array> {
    if (startBlock < 0 || startBlock >= N_LAYERS)
      throw new Error(`startBlock out of range: ${startBlock}`)
    if (resid.length !== seq * D_MODEL)
      throw new Error(
        `resid length ${resid.length} != seq*${D_MODEL} (${seq * D_MODEL})`,
      )
    let residTensor: ort.Tensor = new ort.Tensor('float32', resid, [
      1,
      seq,
      D_MODEL,
    ])
    for (let i = startBlock; i < N_LAYERS; i++) {
      const name = `block_${String(i).padStart(2, '0')}`
      const out = await this.sessions[name].run({ resid: residTensor })
      residTensor = out.resid_out as ort.Tensor
    }
    const logitsTensor = (
      await this.sessions.unembed.run({ resid: residTensor })
    ).logits as ort.Tensor
    const all = logitsTensor.data as Float32Array
    // Slice out the final position's vocab row.
    const base = (seq - 1) * VOCAB_SIZE
    return all.slice(base, base + VOCAB_SIZE)
  }
}

/**
 * Softmax + top-k over a single position's logit row.
 * `logits` is the full [1,seq,vocab] flat array; we index the row for `pos`.
 * Returns entries sorted by descending prob. Numerically stable.
 */
export function topKAtPosition(
  logits: Float32Array,
  pos: number,
  seq: number,
  k: number,
): TopKEntry[] {
  const base = pos * VOCAB_SIZE
  // max for stability
  let max = -Infinity
  for (let v = 0; v < VOCAB_SIZE; v++) {
    const x = logits[base + v]
    if (x > max) max = x
  }
  // exp + sum
  let sum = 0
  // We only need the top-k, so track a small candidate list while summing.
  const topIdx: number[] = []
  const topVal: number[] = []
  for (let v = 0; v < VOCAB_SIZE; v++) {
    const e = Math.exp(logits[base + v] - max)
    sum += e
    // maintain a tiny top-k by raw logit (monotonic with prob)
    const lg = logits[base + v]
    if (topIdx.length < k) {
      topIdx.push(v)
      topVal.push(lg)
    } else {
      // find current min in the small buffer
      let minI = 0
      for (let j = 1; j < topVal.length; j++) if (topVal[j] < topVal[minI]) minI = j
      if (lg > topVal[minI]) {
        topIdx[minI] = v
        topVal[minI] = lg
      }
    }
  }
  const out: TopKEntry[] = topIdx.map((token, j) => ({
    token,
    logit: topVal[j],
    prob: Math.exp(topVal[j] - max) / sum,
  }))
  out.sort((a, b) => b.logit - a.logit)
  void seq
  return out
}

/** Convenience: a typed-array view of one layer/head attention matrix
 *  [seq, seq] out of the flat [1,12,seq,seq] pattern buffer — no copy. */
export function headMatrix(
  pattern: Float32Array,
  head: number,
  seq: number,
): Float32Array {
  const per = seq * seq
  const off = head * per
  return pattern.subarray(off, off + per)
}
