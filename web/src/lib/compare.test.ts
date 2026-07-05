/**
 * compare.test.ts — validates the adversarial-Compare compute (compare.ts)
 * with synthetic RunResults, so the KL / suffix-attention / lens-diff math is
 * pinned without needing the ONNX model.
 */
import { describe, it, expect } from 'vitest'
import { N_HEADS, N_LAYERS, VOCAB_SIZE, type RunResult } from './runner'
import {
  softmaxRow,
  klDivergence,
  perLayerLensKL,
  suffixAttentionMass,
  computeCompare,
} from './compare'

/** Build a lensLogits buffer [1,seq,VOCAB] where every position/layer favors a
 *  given token id with a chosen logit bump, rest zeros. */
function lensBuf(seq: number, favored: number, bump: number): Float32Array {
  const buf = new Float32Array(seq * VOCAB_SIZE)
  for (let pos = 0; pos < seq; pos++) buf[pos * VOCAB_SIZE + favored] = bump
  return buf
}

/** A synthetic RunResult with controllable lens + attention. */
function makeRun(opts: {
  seq: number
  favored: number
  bump: number
  /** pattern[l] provider: (l,h,q,k) -> weight; rows need not be normalized. */
  attn?: (l: number, h: number, q: number, k: number) => number
}): RunResult {
  const { seq, favored, bump, attn } = opts
  const lensLogits: Float32Array[] = []
  const patterns: Float32Array[] = []
  const resids: Float32Array[] = []
  for (let l = 0; l < N_LAYERS; l++) {
    lensLogits.push(lensBuf(seq, favored, bump))
    const pat = new Float32Array(N_HEADS * seq * seq)
    if (attn) {
      for (let h = 0; h < N_HEADS; h++)
        for (let q = 0; q < seq; q++)
          for (let k = 0; k < seq; k++)
            pat[h * seq * seq + q * seq + k] = attn(l, h, q, k)
    }
    patterns.push(pat)
    resids.push(new Float32Array(1))
  }
  return { seq, patterns, lensLogits, resids }
}

describe('softmaxRow', () => {
  it('normalizes to a probability distribution', () => {
    const buf = new Float32Array(VOCAB_SIZE)
    buf[5] = 10
    buf[9] = 10
    const out = new Float32Array(VOCAB_SIZE)
    softmaxRow(buf, 0, out)
    let sum = 0
    for (let v = 0; v < VOCAB_SIZE; v++) sum += out[v]
    expect(sum).toBeCloseTo(1, 5)
    // The two tied top logits get equal mass and each dominates any single
    // background (logit-0) token by a factor of exp(10).
    expect(out[5]).toBeCloseTo(out[9], 6)
    expect(out[5] / out[0]).toBeCloseTo(Math.exp(10), 2)
    expect(out[5]).toBeGreaterThan(out[0] * 1000)
  })
})

describe('klDivergence', () => {
  it('is zero for identical distributions', () => {
    const p = new Float32Array([0.25, 0.25, 0.25, 0.25])
    expect(klDivergence(p, p)).toBeCloseTo(0, 10)
  })
  it('is positive when distributions differ', () => {
    const p = new Float32Array([0.9, 0.1])
    const q = new Float32Array([0.1, 0.9])
    expect(klDivergence(p, q)).toBeGreaterThan(0)
  })
  it('matches a hand-computed value', () => {
    // KL([1,0] || [0.5,0.5]) = 1*ln(1/0.5) = ln 2
    const p = new Float32Array([1, 0])
    const q = new Float32Array([0.5, 0.5])
    expect(klDivergence(p, q)).toBeCloseTo(Math.log(2), 6)
  })
})

describe('perLayerLensKL', () => {
  it('is ~zero per layer when both runs favor the same token (per-run final pos)', () => {
    const base = makeRun({ seq: 4, favored: 100, bump: 8 })
    const pert = makeRun({ seq: 6, favored: 100, bump: 8 })
    const kl = perLayerLensKL(base, pert, base.seq - 1, pert.seq - 1)
    expect(kl).toHaveLength(N_LAYERS)
    for (const k of kl) expect(k).toBeLessThan(1e-6)
  })
  it('is positive per layer when the perturbed run favors a different token', () => {
    const base = makeRun({ seq: 4, favored: 100, bump: 8 })
    const pert = makeRun({ seq: 6, favored: 200, bump: 8 })
    const kl = perLayerLensKL(base, pert, base.seq - 1, pert.seq - 1)
    for (const k of kl) expect(k).toBeGreaterThan(0.1)
  })
})

describe('suffixAttentionMass', () => {
  it('is ~1 per head when suffix queries attend entirely to the suffix', () => {
    const baseSeq = 3
    const pSeq = 5 // suffix positions (queries and keys) = {3,4}
    // Each suffix query puts weight 0.5 on each of the two suffix keys =>
    // per-row suffix mass = 1.0.
    const pert = makeRun({
      seq: pSeq,
      favored: 0,
      bump: 0,
      attn: (_l, _h, _q, k) => (k >= baseSeq ? 0.5 : 0),
    })
    const mass = suffixAttentionMass(pert, baseSeq)
    expect(mass).toHaveLength(N_LAYERS * N_HEADS)
    for (let i = 0; i < mass.length; i++) expect(mass[i]).toBeCloseTo(1, 6)
  })
  it('is 0 when suffix queries attend only back into the base span', () => {
    const baseSeq = 3
    const pSeq = 5
    // Suffix queries (q>=3) attend only to base keys (k<3): zero suffix mass.
    const pert = makeRun({
      seq: pSeq,
      favored: 0,
      bump: 0,
      attn: (_l, _h, q, k) => (q >= baseSeq && k < baseSeq ? 1 / baseSeq : 0),
    })
    const mass = suffixAttentionMass(pert, baseSeq)
    for (let i = 0; i < mass.length; i++) expect(mass[i]).toBe(0)
  })
  it('is 0.5 when suffix queries split attention evenly between base and suffix', () => {
    const baseSeq = 4
    const pSeq = 6 // suffix keys {4,5}
    // Each suffix query puts 0.5 total on the two suffix keys, 0.5 on base.
    const pert = makeRun({
      seq: pSeq,
      favored: 0,
      bump: 0,
      attn: (_l, _h, _q, k) => (k >= baseSeq ? 0.25 : 0.125),
    })
    const mass = suffixAttentionMass(pert, baseSeq)
    for (let i = 0; i < mass.length; i++) expect(mass[i]).toBeCloseTo(0.5, 6)
  })
  it('returns all zeros when there is no suffix', () => {
    const pert = makeRun({ seq: 4, favored: 0, bump: 0, attn: () => 0.25 })
    const mass = suffixAttentionMass(pert, 4)
    for (let i = 0; i < mass.length; i++) expect(mass[i]).toBe(0)
  })
})

describe('computeCompare', () => {
  it('reports the right shapes, compare position, and suffix length', () => {
    const base = makeRun({ seq: 4, favored: 100, bump: 8 })
    const pert = makeRun({
      seq: 7,
      favored: 200,
      bump: 8,
      attn: (_l, _h, _q, k) => (k >= 4 ? 1 / 3 : 0),
    })
    const res = computeCompare(base, pert)
    expect(res.baseSeq).toBe(4)
    expect(res.perturbedSeq).toBe(7)
    expect(res.suffixLen).toBe(3)
    expect(res.basePos).toBe(3) // baseSeq - 1
    expect(res.perturbedPos).toBe(6) // perturbedSeq - 1
    expect(res.kl).toHaveLength(N_LAYERS)
    expect(res.suffixAttn).toHaveLength(N_LAYERS * N_HEADS)
    expect(res.lensDiff).toHaveLength(N_LAYERS)
    // Every layer flipped top-1 (100 -> 200) and captured all suffix mass.
    for (const row of res.lensDiff) expect(row.changed).toBe(true)
    for (let i = 0; i < res.suffixAttn.length; i++)
      expect(res.suffixAttn[i]).toBeCloseTo(1, 6)
  })
})
