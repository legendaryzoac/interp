/**
 * adversarial.test.ts — the suffix loader must be defensive: tolerate absent /
 * malformed files, unknown provenance (future GCG entries), and always yield at
 * least the fallback suffix. We drive normalization through a mocked fetch.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import {
  loadAdversarial,
  provenanceLabel,
  FALLBACK_SUFFIX,
} from './adversarial'

function mockFetch(body: unknown, ok = true) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok,
      json: async () => body,
    })),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('provenanceLabel', () => {
  it('maps known provenances and passes unknown ones through', () => {
    expect(provenanceLabel('gcg')).toBe('GCG optimized')
    expect(provenanceLabel('curated')).toBe('curated')
    expect(provenanceLabel('mystery')).toBe('mystery')
  })
})

describe('loadAdversarial', () => {
  it('falls back to the built-in suffix when the file is absent (404)', async () => {
    mockFetch(null, false)
    const d = await loadAdversarial()
    expect(d.suffixes).toEqual([FALLBACK_SUFFIX])
  })

  it('falls back when fetch throws', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('network down')
      }),
    )
    const d = await loadAdversarial()
    expect(d.suffixes).toEqual([FALLBACK_SUFFIX])
  })

  it('parses curated + gcg entries and keeps unknown provenance', async () => {
    mockFetch({
      suffixes: [
        { id: 'a', text: ' foo', provenance: 'curated', note: 'n', target: null },
        { id: 'b', text: ' bar', provenance: 'gcg', note: '', target: ' Mary' },
        { id: 'c', text: ' baz', provenance: 'future-thing', note: '', target: null },
      ],
    })
    const d = await loadAdversarial()
    expect(d.suffixes.map((s) => s.id)).toEqual(['a', 'b', 'c'])
    expect(d.suffixes[1].provenance).toBe('gcg')
    expect(d.suffixes[1].target).toBe(' Mary')
    // Unknown provenance is preserved verbatim for the badge.
    expect(d.suffixes[2].provenance).toBe('future-thing')
  })

  it('drops entries with no id or empty text, keeping the rest', async () => {
    mockFetch({
      suffixes: [
        { id: 'ok', text: ' hi', provenance: 'curated' },
        { text: ' no-id', provenance: 'curated' },
        { id: 'empty', text: '', provenance: 'curated' },
        'garbage',
      ],
    })
    const d = await loadAdversarial()
    expect(d.suffixes.map((s) => s.id)).toEqual(['ok'])
  })

  it('falls back when suffixes is missing or not an array', async () => {
    mockFetch({ nope: true })
    const d = await loadAdversarial()
    expect(d.suffixes).toEqual([FALLBACK_SUFFIX])
  })
})
