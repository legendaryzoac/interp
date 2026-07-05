/**
 * circuits.test.ts — validates the Circuits compute (induction scoring, head
 * ranking, logit-diff, and the precomputed-JSON flatteners) with synthetic
 * data, so the mechanistic math is pinned without the ONNX model.
 */
import { describe, it, expect } from 'vitest'
import { N_HEADS, N_LAYERS, type RunResult } from './runner'
import {
  makeRepeatedSequence,
  inductionScores,
  rankHeads,
  logitDiffRow,
  inductionScoresFromJson,
} from './circuits'

/** Build a RunResult whose attention pattern is set by `attn(l,h,q,k)`. */
function makeRun(
  seq: number,
  attn: (l: number, h: number, q: number, k: number) => number,
): RunResult {
  const patterns: Float32Array[] = []
  for (let l = 0; l < N_LAYERS; l++) {
    const pat = new Float32Array(N_HEADS * seq * seq)
    for (let h = 0; h < N_HEADS; h++)
      for (let q = 0; q < seq; q++)
        for (let k = 0; k < seq; k++)
          pat[h * seq * seq + q * seq + k] = attn(l, h, q, k)
    patterns.push(pat)
  }
  return { seq, patterns, lensLogits: [], resids: [] }
}

describe('makeRepeatedSequence', () => {
  it('repeats a random unit once, giving 2*unitLen aligned tokens', () => {
    const seq = makeRepeatedSequence(5, () => 0.5)
    expect(seq).toHaveLength(10)
    // second half must equal the first half token-for-token
    for (let i = 0; i < 5; i++) expect(seq[i + 5]).toBe(seq[i])
  })
  it('draws ids inside the safe mid-vocab band', () => {
    const seq = makeRepeatedSequence(50, Math.random)
    for (const id of seq) {
      expect(id).toBeGreaterThanOrEqual(1000)
      expect(id).toBeLessThan(40000)
    }
  })
})

describe('inductionScores', () => {
  const unitLen = 4
  const seq = 2 * unitLen // 8

  it('scores 1.0 for a perfect induction head, 0 for others', () => {
    // Induction head (h=0): each 2nd-occurrence query i in [4,8) attends fully
    // to target = i - unitLen + 1. Everything else attends elsewhere.
    const run = makeRun(seq, (_l, h, q, k) => {
      if (h === 0) {
        const target = q - unitLen + 1
        return q >= unitLen && k === target ? 1 : 0
      }
      // non-induction head: attend to the diagonal (self)
      return q === k ? 1 : 0
    })
    const scores = inductionScores(run, unitLen)
    for (let l = 0; l < N_LAYERS; l++) {
      expect(scores[l * N_HEADS + 0]).toBeCloseTo(1, 6) // induction head
      expect(scores[l * N_HEADS + 1]).toBeCloseTo(0, 6) // self-attn head
    }
  })

  it('is 0 when the sequence is too short', () => {
    const run = makeRun(2, () => 1)
    const scores = inductionScores(run, 25)
    for (let i = 0; i < scores.length; i++) expect(scores[i]).toBe(0)
  })

  it('averages partial induction attention', () => {
    // Induction head puts 0.5 on the induction target for every 2nd-occ query.
    const run = makeRun(seq, (_l, h, q, k) => {
      if (h === 3) {
        const target = q - unitLen + 1
        return q >= unitLen && k === target ? 0.5 : 0
      }
      return 0
    })
    const scores = inductionScores(run, unitLen)
    expect(scores[3]).toBeCloseTo(0.5, 6)
  })
})

describe('rankHeads', () => {
  it('returns the top heads by score, descending', () => {
    const scores = new Float32Array(N_LAYERS * N_HEADS)
    scores[5 * N_HEADS + 1] = 0.9 // L5 H1
    scores[5 * N_HEADS + 5] = 0.8 // L5 H5
    scores[0 * N_HEADS + 0] = 0.1
    const top = rankHeads(scores, 3)
    expect(top[0]).toEqual({ layer: 5, head: 1, score: expect.closeTo(0.9, 6) })
    expect(top[1]).toEqual({ layer: 5, head: 5, score: expect.closeTo(0.8, 6) })
    expect(top[2].score).toBeCloseTo(0.1, 6)
  })
})

describe('logitDiffRow', () => {
  it('is logit[a] - logit[b] at the given ids', () => {
    const row = new Float32Array(100)
    row[10] = 4
    row[20] = 1.5
    expect(logitDiffRow(row, 10, 20)).toBeCloseTo(2.5, 6)
    expect(logitDiffRow(row, 20, 10)).toBeCloseTo(-2.5, 6)
  })
})

describe('inductionScoresFromJson', () => {
  it('flattens a [layer][head] matrix into the flat scores array', () => {
    const matrix = Array.from({ length: N_LAYERS }, (_, l) =>
      Array.from({ length: N_HEADS }, (_, h) => l * 100 + h),
    )
    const flat = inductionScoresFromJson(matrix)
    expect(flat).toHaveLength(N_LAYERS * N_HEADS)
    expect(flat[5 * N_HEADS + 1]).toBe(501)
  })
  it('tolerates a ragged / short matrix without throwing', () => {
    const flat = inductionScoresFromJson([[1, 2], [3]])
    expect(flat[0]).toBe(1)
    expect(flat[1]).toBe(2)
    expect(flat[N_HEADS]).toBe(3)
    // untouched cells stay zero
    expect(flat[N_HEADS + 5]).toBe(0)
  })
})
