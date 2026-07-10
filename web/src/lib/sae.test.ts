/**
 * sae.test.ts — pins the two pure pieces of the SAE encoder that would fail
 * silently if wrong: the S2 centering transform (BASIS_CONTRACT.md — feed the
 * encoder the wrong basis and it produces garbage without erroring) and the
 * per-token top-k selection.
 */
import { describe, it, expect } from 'vitest'
import {
  centerRows,
  topKPerToken,
  D_SAE,
  type FeatureMeta,
} from './sae'

describe('centerRows (S2 basis transform)', () => {
  it('subtracts each token row mean over d_model, per hand computation', () => {
    // seq=2, d=4. Row 0 mean = 2.5; row 1 is constant so centers to all zeros.
    const x = new Float32Array([1, 2, 3, 4, 10, 10, 10, 10])
    const out = centerRows(x, 2, 4)
    expect(Array.from(out.slice(0, 4))).toEqual([-1.5, -0.5, 0.5, 1.5])
    expect(Array.from(out.slice(4, 8))).toEqual([0, 0, 0, 0])
  })

  it('makes every token row sum to ~0', () => {
    const seq = 3
    const d = 5
    const x = new Float32Array(seq * d)
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i) * 7 + i
    const out = centerRows(x, seq, d)
    for (let t = 0; t < seq; t++) {
      let sum = 0
      for (let i = 0; i < d; i++) sum += out[t * d + i]
      expect(sum).toBeCloseTo(0, 5)
    }
  })

  it('does not mutate the input buffer', () => {
    const x = new Float32Array([1, 2, 3, 4])
    const snapshot = Array.from(x)
    centerRows(x, 1, 4)
    expect(Array.from(x)).toEqual(snapshot)
  })

  it('throws on a length that does not match seq*d', () => {
    expect(() => centerRows(new Float32Array(7), 2, 4)).toThrow()
  })
})

describe('topKPerToken', () => {
  /** Build a feats buffer [1, seq, D_SAE] and set (token, feature) -> value. */
  function makeFeats(seq: number, sets: [number, number, number][]): Float32Array {
    const buf = new Float32Array(seq * D_SAE)
    for (const [t, f, v] of sets) buf[t * D_SAE + f] = v
    return buf
  }

  it('returns the k highest active features per token, sorted descending', () => {
    const feats = makeFeats(1, [
      [0, 100, 5],
      [0, 200, 3],
      [0, 300, 9],
    ])
    const top = topKPerToken(feats, 1, 2)
    expect(top).toHaveLength(1)
    expect(top[0].map((h) => h.id)).toEqual([300, 100])
    expect(top[0].map((h) => h.value)).toEqual([9, 5])
  })

  it('skips inactive (<=0) features and never pads to k', () => {
    const feats = makeFeats(1, [
      [0, 10, 4],
      [0, 20, 0], // ReLU-inactive
    ])
    const top = topKPerToken(feats, 1, 5)
    expect(top[0]).toHaveLength(1)
    expect(top[0][0].id).toBe(10)
  })

  it('labels every hit as an unlabeled feature #id when no resolver is given', () => {
    const feats = makeFeats(1, [[0, 42, 1]])
    const top = topKPerToken(feats, 1, 3)
    expect(top[0][0].label).toBe('feature #42')
    expect(top[0][0].labeled).toBe(false)
  })

  it('attaches curated labels via the resolver', () => {
    const feats = makeFeats(1, [[0, 18, 7]])
    const resolve = (id: number): FeatureMeta => ({
      id,
      freq: 0.0012,
      label: id === 18 ? 'Closing bracket or end of citation' : null,
      labelConfidence: 'high',
      hasDashboard: true,
    })
    const top = topKPerToken(feats, 1, 3, resolve)
    expect(top[0][0].labeled).toBe(true)
    expect(top[0][0].label).toBe('Closing bracket or end of citation')
    expect(top[0][0].freq).toBeCloseTo(0.0012)
  })

  it('handles multiple tokens independently', () => {
    const feats = makeFeats(2, [
      [0, 5, 2],
      [1, 6, 8],
      [1, 7, 1],
    ])
    const top = topKPerToken(feats, 2, 1)
    expect(top[0][0].id).toBe(5)
    expect(top[1][0].id).toBe(6)
  })
})
