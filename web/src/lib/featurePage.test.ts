/**
 * featurePage.test.ts — pins the pure decoders (decodeActs, histogramToBars)
 * and the chunk-resolution + memoization that keeps opening several features
 * from one chunk down to a single fetch, and that NEVER fetches a null chunk.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  decodeActs,
  histogramToBars,
  loadFeature,
  __resetFeaturePageCache,
} from './featurePage'

describe('decodeActs', () => {
  it('decodes acts=[0,128,255], max_act=8 to [0, ~4.01, 8]', () => {
    const out = decodeActs([0, 128, 255], 8)
    expect(out[0]).toBe(0)
    expect(out[1]).toBeCloseTo(4.0157, 3)
    expect(out[2]).toBe(8)
  })

  it('returns a Float32Array of matching length', () => {
    const out = decodeActs([0, 255, 51], 10)
    expect(out).toBeInstanceOf(Float32Array)
    expect(out.length).toBe(3)
    expect(out[2]).toBeCloseTo(2, 5) // 51/255*10
  })
})

describe('histogramToBars', () => {
  it('pairs each count with its [bins[i], bins[i+1]] edges', () => {
    const bars = histogramToBars({ bins: [0, 1, 2], counts: [5, 9] })
    expect(bars).toEqual([
      { x0: 0, x1: 1, count: 5 },
      { x0: 1, x1: 2, count: 9 },
    ])
  })

  it('clamps to the shorter of counts / (bins-1)', () => {
    const bars = histogramToBars({ bins: [0, 1, 2, 3], counts: [7] })
    expect(bars).toHaveLength(1)
    expect(bars[0]).toEqual({ x0: 0, x1: 1, count: 7 })
  })

  it('returns [] for missing input', () => {
    expect(histogramToBars(undefined)).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// chunk resolution + memoization (mocked network; no Cache API in node → the
// versioned-cache helper returns null and every miss goes to fetch)
// ---------------------------------------------------------------------------

const INDEX = {
  features: {
    '18': { freq: 0.0012, chunk: 0, hasDashboard: true, label: 'Closing bracket', label_confidence: 'high' },
    '41': { freq: 0.002, chunk: 0, hasDashboard: true, label: 'Slash separators', label_confidence: 'low' },
    '999': { freq: 0.0013, chunk: null, hasDashboard: false },
  },
}

function rawFeature(id: number, label: string) {
  return {
    id,
    freq: 0.0012,
    max_act: 68,
    n_active: 2000,
    label,
    label_confidence: 'high',
    histogram: { bins: [0, 1, 2], counts: [3, 4] },
    top_examples: [{ tokens: [' a', ' b'], acts: [0, 255], max_act: 5, act_index: 1 }],
    logit_lens: { promoted: [[']', 1.1]], suppressed: [['agra', -0.6]] },
  }
}

const CHUNK0 = [rawFeature(18, 'Closing bracket'), rawFeature(41, 'Slash separators')]

function jsonResponse(body: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => body,
    clone() {
      return this
    },
  }
}

let calls: string[] = []

beforeEach(() => {
  calls = []
  __resetFeaturePageCache()
  vi.stubGlobal('fetch', (input: unknown) => {
    const url = String(input)
    calls.push(url)
    if (url.endsWith('/dashboards/index.json')) return Promise.resolve(jsonResponse(INDEX))
    if (url.endsWith('/dashboards/manifest.json'))
      return Promise.resolve(jsonResponse({ content_hash: 'testhash' }))
    if (url.endsWith('/dashboards/features_0000.json'))
      return Promise.resolve(jsonResponse(CHUNK0))
    return Promise.resolve({ ok: false, status: 404, statusText: 'Not Found', json: async () => null })
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const chunkFetches = () => calls.filter((u) => u.includes('features_0000.json')).length

describe('loadFeature — chunk resolution + memoization', () => {
  it('resolves a dashboard feature to a full view', async () => {
    const f = await loadFeature(18)
    expect(f.kind).toBe('full')
    if (f.kind === 'full') {
      expect(f.label).toBe('Closing bracket')
      expect(f.labelConfidence).toBe('high')
      expect(f.examples).toHaveLength(1)
      expect(f.examples[0].actIndex).toBe(1)
      expect(f.promoted[0]).toEqual({ token: ']', weight: 1.1 })
      expect(f.suppressed[0]).toEqual({ token: 'agra', weight: -0.6 })
    }
  })

  it('fetches the chunk only once for two features in the same chunk', async () => {
    const a = await loadFeature(18)
    const b = await loadFeature(41)
    expect(a.kind).toBe('full')
    expect(b.kind).toBe('full')
    if (b.kind === 'full') expect(b.label).toBe('Slash separators')
    expect(chunkFetches()).toBe(1)
  })

  it('returns a minimal view and NEVER fetches a null chunk', async () => {
    const f = await loadFeature(999)
    expect(f.kind).toBe('minimal')
    expect(f.freq).toBeCloseTo(0.0013)
    // no features_*.json request at all
    expect(calls.some((u) => u.includes('features_'))).toBe(false)
  })

  it('returns a minimal view for a feature absent from the index', async () => {
    const f = await loadFeature(123456)
    expect(f.kind).toBe('minimal')
    expect(f.label).toBeNull()
    expect(chunkFetches()).toBe(0)
  })
})
