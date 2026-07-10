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

describe('logitsAreFinite', () => {
  it('is true when any logit is finite, false when all are NaN', () => {
    expect(logitsAreFinite(new Float32Array([NaN, NaN, 1]))).toBe(true)
    expect(logitsAreFinite(new Float32Array([NaN, NaN, NaN]))).toBe(false)
  })
})
