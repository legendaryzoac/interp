/**
 * steering.test.ts — pins the pure pieces of S9 feature steering that would fail
 * silently if wrong: fp16 decoder-row decode, the raw-basis steering add (and
 * the alpha=0 == baseline invariant), and next-token sampling.
 */
import { describe, it, expect } from 'vitest'
import {
  halfToFloat,
  decodeDecoderRow,
  addSteeringVector,
  sampleToken,
  logitsAreFinite,
  mulberry32,
  D_MODEL,
  DEFAULT_TOP_P,
  DEFAULT_REPETITION_PENALTY,
  DEFAULT_NO_REPEAT_NGRAM,
} from './steering'

// Encode a JS number as the raw 16 bits of its nearest fp16 (round-to-nearest),
// so tests can build synthetic decoder blobs without a Float16Array.
function floatToHalf(f: number): number {
  const fbuf = new Float32Array([f])
  const x = new Uint32Array(fbuf.buffer)[0]
  const sign = (x >> 16) & 0x8000
  let mant = x & 0x007fffff
  let exp = (x >> 23) & 0xff
  if (exp === 0xff) {
    // Inf / NaN
    return sign | 0x7c00 | (mant ? 0x200 : 0)
  }
  exp = exp - 127 + 15
  if (exp >= 0x1f) return sign | 0x7c00 // overflow → Inf
  if (exp <= 0) {
    // subnormal / underflow to zero
    if (exp < -10) return sign
    mant = mant | 0x00800000
    const shift = 14 - exp
    const half = mant >> shift
    return sign | half
  }
  return sign | (exp << 10) | (mant >> 13)
}

/** Little-endian fp16 blob from a flat number array. */
function makeBlob(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(values.length * 2)
  const dv = new DataView(buf)
  values.forEach((v, i) => dv.setUint16(i * 2, floatToHalf(v), true))
  return buf
}

describe('halfToFloat', () => {
  it('round-trips exact fp16-representable values', () => {
    for (const v of [0, 1, -1, 0.5, -0.5, 2, 0.25, 1024, -3.5, 0.125]) {
      expect(halfToFloat(floatToHalf(v))).toBeCloseTo(v, 6)
    }
  })

  it('decodes Inf and NaN', () => {
    expect(halfToFloat(0x7c00)).toBe(Infinity)
    expect(halfToFloat(0xfc00)).toBe(-Infinity)
    expect(Number.isNaN(halfToFloat(0x7e00))).toBe(true)
  })

  it('decodes a subnormal (smallest positive fp16)', () => {
    // 0x0001 = 2^-24 ≈ 5.96e-8
    expect(halfToFloat(0x0001)).toBeCloseTo(Math.pow(2, -24), 12)
  })
})

describe('decodeDecoderRow', () => {
  it('extracts the right row from a multi-row fp16 blob', () => {
    const d = 4
    // row 0 = [1,2,3,4], row 1 = [-1,-2,-3,-4], row 2 = [0.5,0.25,0.125,-0.5]
    const blob = makeBlob([1, 2, 3, 4, -1, -2, -3, -4, 0.5, 0.25, 0.125, -0.5])
    expect(Array.from(decodeDecoderRow(blob, 0, d))).toEqual([1, 2, 3, 4])
    expect(Array.from(decodeDecoderRow(blob, 1, d))).toEqual([-1, -2, -3, -4])
    expect(Array.from(decodeDecoderRow(blob, 2, d))).toEqual([0.5, 0.25, 0.125, -0.5])
  })

  it('throws when the row runs past the blob', () => {
    const blob = makeBlob([1, 2, 3, 4]) // one row of d=4
    expect(() => decodeDecoderRow(blob, 1, 4)).toThrow()
    expect(() => decodeDecoderRow(blob, -1, 4)).toThrow()
  })

  it('defaults to D_MODEL-wide rows', () => {
    const row0 = new Array(D_MODEL).fill(0).map((_, i) => (i % 7) - 3)
    const row1 = new Array(D_MODEL).fill(0).map((_, i) => ((i % 5) - 2) * 0.5)
    const blob = makeBlob([...row0, ...row1])
    const dec = decodeDecoderRow(blob, 1)
    expect(dec.length).toBe(D_MODEL)
    for (let i = 0; i < D_MODEL; i++) expect(dec[i]).toBeCloseTo(row1[i], 5)
  })
})

describe('addSteeringVector', () => {
  it('adds alpha*vec to every position, returning a new buffer', () => {
    const d = 3
    const seq = 2
    const resid = new Float32Array([1, 1, 1, 2, 2, 2])
    const vec = new Float32Array([10, 0, -10])
    const out = addSteeringVector(resid, vec, 2, seq, d)
    expect(Array.from(out)).toEqual([21, 1, -19, 22, 2, -18])
    // input untouched
    expect(Array.from(resid)).toEqual([1, 1, 1, 2, 2, 2])
  })

  it('alpha=0 returns an exact clone == baseline (bit-for-bit)', () => {
    const d = 4
    const seq = 3
    const resid = new Float32Array(seq * d)
    for (let i = 0; i < resid.length; i++) resid[i] = Math.sin(i) * 13
    const vec = new Float32Array([5, -5, 2, -2])
    const out = addSteeringVector(resid, vec, 0, seq, d)
    expect(out).not.toBe(resid) // a distinct buffer
    expect(Array.from(out)).toEqual(Array.from(resid)) // identical values
  })

  it('negative alpha suppresses (subtracts the feature direction)', () => {
    const out = addSteeringVector(
      new Float32Array([0, 0, 0]),
      new Float32Array([1, 2, 3]),
      -4,
      1,
      3,
    )
    expect(Array.from(out)).toEqual([-4, -8, -12])
  })

  it('throws on shape mismatch', () => {
    expect(() =>
      addSteeringVector(new Float32Array(6), new Float32Array(3), 1, 3, 3),
    ).toThrow()
    expect(() =>
      addSteeringVector(new Float32Array(9), new Float32Array(2), 1, 3, 3),
    ).toThrow()
  })
})

describe('sampleToken', () => {
  it('greedy (temperature 0) picks the argmax', () => {
    const logits = new Float32Array([0.1, 3.2, -1, 3.1])
    expect(sampleToken(logits, 0)).toBe(1)
  })

  it('greedy skips NaN entries (never returns a NaN slot)', () => {
    const logits = new Float32Array([NaN, 2, NaN, 5, 1])
    expect(sampleToken(logits, 0)).toBe(3)
  })

  it('temperature sampling is reproducible under a seeded rng', () => {
    const logits = new Float32Array([1, 2, 3, 4, 5])
    const a = sampleToken(logits, 0.8, mulberry32(42))
    const b = sampleToken(logits, 0.8, mulberry32(42))
    expect(a).toBe(b)
    expect(a).toBeGreaterThanOrEqual(0)
    expect(a).toBeLessThan(5)
  })

  it('temperature sampling with rng→0 lands on the first mass, →~1 on the last', () => {
    const logits = new Float32Array([0, 0, 0])
    expect(sampleToken(logits, 1, () => 0)).toBe(0)
    expect(sampleToken(logits, 1, () => 0.999999)).toBe(2)
  })
})

describe('sampleToken — nucleus (top-p) truncation', () => {
  it('never samples a token outside the nucleus, even at rng→~1', () => {
    // Two dominant tokens (~0.47 each) + a smaller tail token (~0.06). At temp 1
    // the cumulative mass of {0,1} is ≈0.94 ≥ 0.9, so topP 0.9 excludes token 2 —
    // it must be unreachable however high rng climbs.
    const logits = new Float32Array([2, 2, 0])
    for (const u of [0, 0.25, 0.5, 0.75, 0.999999]) {
      expect(sampleToken(logits, 1, () => u, { topP: 0.9 })).not.toBe(2)
    }
    // The same tail token IS reachable when truncation is disabled (topP ≥ 1) —
    // its ~0.06 mass is non-negligible — proving it's the nucleus doing the
    // excluding, not the token simply having zero probability.
    expect(sampleToken(logits, 1, () => 0.999999, { topP: 1 })).toBe(2)
  })

  it('keeps only the single top token when it already covers topP', () => {
    // token 0 dominates (prob ≈ 1). Nucleus collapses to {0}; every draw picks it.
    const logits = new Float32Array([20, 0, -1, -5])
    for (const u of [0, 0.5, 0.999999]) {
      expect(sampleToken(logits, 1, () => u, { topP: 0.9 })).toBe(0)
    }
  })

  it('defaults to DEFAULT_TOP_P when topP is unspecified', () => {
    expect(DEFAULT_TOP_P).toBeGreaterThan(0)
    expect(DEFAULT_TOP_P).toBeLessThanOrEqual(1)
    const logits = new Float32Array([2, 2, 0])
    // With the default nucleus the ~0.06-mass tail token is still excluded.
    expect(sampleToken(logits, 1, () => 0.999999)).not.toBe(2)
  })
})

describe('sampleToken — repetition penalty', () => {
  it('lowers a repeated token’s odds enough to flip the greedy pick', () => {
    // token 0 leads by a hair; penalising it (it was already generated) hands the
    // pick to token 1.
    const logits = new Float32Array([5, 4.9])
    expect(sampleToken(logits, 0)).toBe(0) // no penalty → argmax is 0
    expect(
      sampleToken(logits, 0, Math.random, { generated: [0], repetitionPenalty: 1.3 }),
    ).toBe(1)
  })

  it('penalises negative logits by multiplying (pushes them further down)', () => {
    // token 2 starts as the argmax at -1; once penalised (×1.3 → -1.3) token 1 wins.
    const logits = new Float32Array([-5, -1.2, -1])
    expect(sampleToken(logits, 0)).toBe(2)
    expect(
      sampleToken(logits, 0, Math.random, { generated: [2], repetitionPenalty: 1.3 }),
    ).toBe(1)
  })

  it('does not mutate the caller’s logit buffer', () => {
    const logits = new Float32Array([5, 4.9])
    const before = Array.from(logits) // fp32-rounded snapshot
    sampleToken(logits, 0, Math.random, { generated: [0], repetitionPenalty: 1.3 })
    expect(Array.from(logits)).toEqual(before) // untouched
  })

  it('is a no-op when nothing has been generated yet', () => {
    const logits = new Float32Array([5, 4.9])
    expect(sampleToken(logits, 0, Math.random, { generated: [] })).toBe(0)
  })

  it('exposes a sane default penalty (>1 so repeats are discouraged)', () => {
    expect(DEFAULT_REPETITION_PENALTY).toBeGreaterThan(1)
  })
})

describe('sampleToken — no-repeat n-gram guard', () => {
  it('masks the token that would complete an already-seen trigram (greedy)', () => {
    // token 0 is the runaway argmax. Once "0 0" has appeared and we're at "…0 0",
    // a third 0 would repeat the trigram (0,0,0) → it's banned, so 1 wins instead.
    const logits = new Float32Array([10, 5, 1])
    // no history → 0
    expect(sampleToken(logits, 0, Math.random, { generated: [], noRepeatNgramSize: 3 })).toBe(0)
    // "0 0" so far → completing (0,0,0) is allowed only if it never appeared; it
    // hasn't yet at this point, so 0 is still fine.
    expect(sampleToken(logits, 0, Math.random, { generated: [0, 0], noRepeatNgramSize: 3 })).toBe(0)
    // "0 0 0" already exists; from "…0 0" another 0 would repeat trigram → banned.
    expect(sampleToken(logits, 0, Math.random, { generated: [0, 0, 0], noRepeatNgramSize: 3 })).toBe(1)
  })

  it('caps a runaway to at most (size) consecutive copies', () => {
    // Feed a fixed distribution that always favours token 0; the guard must break
    // the run so no token id appears 4+ times in a row.
    const logits = new Float32Array([10, 6, 3, 1])
    const gen: number[] = []
    const rng = mulberry32(99)
    for (let i = 0; i < 20; i++) {
      gen.push(sampleToken(logits, 0.3, rng, { generated: gen, noRepeatNgramSize: 3 }))
    }
    let maxRun = 1
    let cur = 1
    for (let i = 1; i < gen.length; i++) {
      cur = gen[i] === gen[i - 1] ? cur + 1 : 1
      if (cur > maxRun) maxRun = cur
    }
    expect(maxRun).toBeLessThanOrEqual(3) // size-3 guard ⇒ never 4 in a row
  })

  it('is disabled by size 0 (a runaway is then possible)', () => {
    const logits = new Float32Array([10, 5, 1])
    // With the guard off, greedy keeps returning the argmax regardless of history.
    expect(
      sampleToken(logits, 0, Math.random, { generated: [0, 0, 0], noRepeatNgramSize: 0 }),
    ).toBe(0)
  })

  it('exposes a sane default n-gram size (≥2)', () => {
    expect(DEFAULT_NO_REPEAT_NGRAM).toBeGreaterThanOrEqual(2)
  })
})

describe('sampleToken — greedy determinism & alpha-0==baseline invariant', () => {
  it('greedy stays deterministic with a repetition context', () => {
    const logits = new Float32Array([1, 3, 2, 4, 0])
    const a = sampleToken(logits, 0, Math.random, { generated: [3, 1] })
    const b = sampleToken(logits, 0, Math.random, { generated: [3, 1] })
    expect(a).toBe(b)
  })

  it('same seed + same logits + same generated ⇒ identical id streams', () => {
    // Simulates baseline vs steered at alpha=0: identical logits every step,
    // separate but same-seeded RNGs, each fed its own generated-so-far ids. The
    // two streams must never diverge — that's the invariant the side-by-side view
    // relies on.
    const logits = new Float32Array([1, 3, 2, 0, 4, 1.5, 2.2])
    const rngA = mulberry32(20260710)
    const rngB = mulberry32(20260710)
    const genA: number[] = []
    const genB: number[] = []
    for (let i = 0; i < 12; i++) {
      genA.push(sampleToken(logits, 0.8, rngA, { generated: genA }))
      genB.push(sampleToken(logits, 0.8, rngB, { generated: genB }))
    }
    expect(genA).toEqual(genB)
  })
})

describe('logitsAreFinite', () => {
  it('is true when any logit is finite, false when all are NaN', () => {
    expect(logitsAreFinite(new Float32Array([NaN, NaN, 1]))).toBe(true)
    expect(logitsAreFinite(new Float32Array([NaN, NaN, NaN]))).toBe(false)
  })
})
