/**
 * compare.ts — pure compute for the adversarial Compare tab.
 *
 * Given a `base` run and a `perturbed` run (= base prompt + adversarial suffix),
 * derive three views of how the suffix bends the model's internals:
 *
 *   1. Per-layer logit-lens KL divergence at the last shared base position.
 *   2. Per-head attention mass on the suffix tokens (12x12).
 *   3. Per-layer logit-lens top-1 diff at the final base position.
 *
 * All functions are pure and typed-array based so they can be unit-tested
 * without a model and reused for a precomputed path later.
 */
import { N_LAYERS, N_HEADS, VOCAB_SIZE, type RunResult } from './runner'
import { labelForId } from './tokenizer'

/** One layer's top-1 lens prediction for base vs perturbed, at a fixed pos. */
export interface LensDiffRow {
  layer: number
  baseToken: string
  baseProb: number
  perturbedToken: string
  perturbedProb: number
  /** True when the suffix flipped the layer's top-1 prediction. */
  changed: boolean
}

export interface CompareResult {
  /** Sequence length of the base run (# of shared prompt tokens). */
  baseSeq: number
  /** Sequence length of the perturbed run (base + suffix). */
  perturbedSeq: number
  /** Number of suffix tokens = perturbedSeq - baseSeq. */
  suffixLen: number
  /** Position in the BASE run the lens comparison is taken at = baseSeq - 1. */
  basePos: number
  /** Position in the PERTURBED run the lens comparison is taken at = perturbedSeq - 1.
   *  We compare each run's own final position — i.e. its next-token belief —
   *  because with a suffix appended, the shared base positions are causally
   *  identical (attention is causal, the suffix is strictly later), so KL there
   *  is exactly 0 and carries no signal. */
  perturbedPos: number
  /** Per-layer KL(base@basePos || perturbed@perturbedPos), full-vocab. length N_LAYERS. */
  kl: number[]
  /** Per-(layer,head) fraction of base-query attention landing on suffix keys.
   *  Flat [N_LAYERS * N_HEADS], row-major (layer outer, head inner). */
  suffixAttn: Float32Array
  /** Per-layer top-1 lens diff: base@basePos vs perturbed@perturbedPos. length N_LAYERS. */
  lensDiff: LensDiffRow[]
}

/**
 * Numerically-stable softmax of one logit row out of a flat [1,seq,vocab]
 * buffer. Writes into `out` (length VOCAB_SIZE) to avoid per-call allocation.
 */
export function softmaxRow(
  logits: Float32Array,
  pos: number,
  out: Float32Array,
): void {
  const base = pos * VOCAB_SIZE
  let max = -Infinity
  for (let v = 0; v < VOCAB_SIZE; v++) {
    const x = logits[base + v]
    if (x > max) max = x
  }
  let sum = 0
  for (let v = 0; v < VOCAB_SIZE; v++) {
    const e = Math.exp(logits[base + v] - max)
    out[v] = e
    sum += e
  }
  const inv = 1 / sum
  for (let v = 0; v < VOCAB_SIZE; v++) out[v] *= inv
}

/**
 * KL(p || q) = sum_v p_v * log(p_v / q_v), for two probability vectors.
 * A small epsilon on q guards log(0)/divide-by-zero when a base-supported token
 * has vanishing perturbed probability (common under int8). Returns nats.
 */
export function klDivergence(p: Float32Array, q: Float32Array): number {
  const eps = 1e-12
  let kl = 0
  for (let v = 0; v < p.length; v++) {
    const pv = p[v]
    if (pv <= eps) continue
    const qv = q[v] > eps ? q[v] : eps
    kl += pv * Math.log(pv / qv)
  }
  // Clamp tiny negatives from float error; KL is non-negative in theory.
  return kl < 0 ? 0 : kl
}

/**
 * Per-layer logit-lens KL between two runs, each read at its own position.
 * Softmaxes the full-vocab lens row for each layer of each run and returns the
 * 12-point KL(base@basePos || perturbed@perturbedPos) curve. Positions are
 * per-run so we compare next-token beliefs, not the causally-identical shared
 * base positions.
 */
export function perLayerLensKL(
  base: RunResult,
  perturbed: RunResult,
  basePos: number,
  perturbedPos: number,
): number[] {
  const pBuf = new Float32Array(VOCAB_SIZE)
  const qBuf = new Float32Array(VOCAB_SIZE)
  const kl: number[] = []
  for (let l = 0; l < N_LAYERS; l++) {
    softmaxRow(base.lensLogits[l], basePos, pBuf)
    softmaxRow(perturbed.lensLogits[l], perturbedPos, qBuf)
    kl.push(klDivergence(pBuf, qBuf))
  }
  return kl
}

/**
 * Per-head fraction of attention, from the SUFFIX query positions of the
 * perturbed run, that lands on the suffix key positions — i.e. how much the
 * adversarial suffix captures its own attention (and thus hijacks that head).
 *
 * The perturbed pattern is [1, N_HEADS, pSeq, pSeq]. For each head we average,
 * over query rows q in [baseSeq, pSeq) (the suffix tokens, including the final
 * position that drives the next-token prediction), the summed weight over key
 * columns k in [baseSeq, pSeq). Each attention row is a distribution summing to
 * ~1, so this reads directly as "what share of the suffix tokens' attention
 * stays on the suffix". Returns a flat [N_LAYERS*N_HEADS] array.
 *
 * NB: attention is causal, so base query positions [0, baseSeq) can never
 * attend to the strictly-later suffix keys — measuring from the suffix queries
 * is the only well-defined direction for "suffix attention mass".
 */
export function suffixAttentionMass(
  perturbed: RunResult,
  baseSeq: number,
): Float32Array {
  const pSeq = perturbed.seq
  const out = new Float32Array(N_LAYERS * N_HEADS)
  const suffixLen = pSeq - baseSeq
  if (suffixLen <= 0 || baseSeq <= 0) return out
  const perHead = pSeq * pSeq
  for (let l = 0; l < N_LAYERS; l++) {
    const pat = perturbed.patterns[l]
    for (let h = 0; h < N_HEADS; h++) {
      const headBase = h * perHead
      let acc = 0
      for (let q = baseSeq; q < pSeq; q++) {
        const rowBase = headBase + q * pSeq
        let onSuffix = 0
        for (let k = baseSeq; k < pSeq; k++) onSuffix += pat[rowBase + k]
        acc += onSuffix
      }
      out[l * N_HEADS + h] = acc / suffixLen
    }
  }
  return out
}

/** Argmax + its softmax prob of one logit row (full vocab). */
function top1(logits: Float32Array, pos: number): { id: number; prob: number } {
  const buf = new Float32Array(VOCAB_SIZE)
  softmaxRow(logits, pos, buf)
  let bestI = 0
  let bestP = -Infinity
  for (let v = 0; v < VOCAB_SIZE; v++) {
    if (buf[v] > bestP) {
      bestP = buf[v]
      bestI = v
    }
  }
  return { id: bestI, prob: bestP }
}

/**
 * Per-layer top-1 logit-lens diff: what the base run's layer predicts (at its
 * final position) vs what the perturbed run's layer predicts (at its final
 * position), and whether the suffix flipped it.
 */
export function perLayerLensDiff(
  base: RunResult,
  perturbed: RunResult,
  basePos: number,
  perturbedPos: number,
): LensDiffRow[] {
  const rows: LensDiffRow[] = []
  for (let l = 0; l < N_LAYERS; l++) {
    const b = top1(base.lensLogits[l], basePos)
    const p = top1(perturbed.lensLogits[l], perturbedPos)
    rows.push({
      layer: l,
      baseToken: labelForId(b.id),
      baseProb: b.prob,
      perturbedToken: labelForId(p.id),
      perturbedProb: p.prob,
      changed: b.id !== p.id,
    })
  }
  return rows
}

/**
 * Full Compare computation from two runs. The perturbed run must be strictly
 * longer than the base run (it carries the appended suffix). KL and lens-diff
 * compare each run at its own final position (its next-token belief); the
 * suffix-attention mass measures how much of the base tokens' attention the
 * suffix positions capture in the perturbed run.
 */
export function computeCompare(base: RunResult, perturbed: RunResult): CompareResult {
  const baseSeq = base.seq
  const perturbedSeq = perturbed.seq
  const basePos = baseSeq - 1
  const perturbedPos = perturbedSeq - 1
  return {
    baseSeq,
    perturbedSeq,
    suffixLen: perturbedSeq - baseSeq,
    basePos,
    perturbedPos,
    kl: perLayerLensKL(base, perturbed, basePos, perturbedPos),
    suffixAttn: suffixAttentionMass(perturbed, baseSeq),
    lensDiff: perLayerLensDiff(base, perturbed, basePos, perturbedPos),
  }
}
