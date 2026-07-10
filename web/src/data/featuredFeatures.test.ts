/**
 * featuredFeatures.test.ts — pins the curated S10 gallery config: it must stay
 * non-empty, have unique in-range ids, and carry only non-blank blurbs. These
 * are the invariants the gallery relies on (one card per id, no dupes) and the
 * cheapest guard against a fat-fingered edit to the shop window.
 */
import { describe, expect, it } from 'vitest'
import { FEATURED_FEATURES } from './featuredFeatures'

describe('FEATURED_FEATURES', () => {
  it('is a curated set of a sensible size', () => {
    expect(FEATURED_FEATURES.length).toBeGreaterThanOrEqual(10)
    expect(FEATURED_FEATURES.length).toBeLessThanOrEqual(15)
  })

  it('has unique feature ids', () => {
    const ids = FEATURED_FEATURES.map((f) => f.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('has integer ids inside the SAE feature range [0, 24575]', () => {
    for (const f of FEATURED_FEATURES) {
      expect(Number.isInteger(f.id)).toBe(true)
      expect(f.id).toBeGreaterThanOrEqual(0)
      expect(f.id).toBeLessThan(24576)
    }
  })

  it('only carries non-blank blurbs when a blurb is present', () => {
    for (const f of FEATURED_FEATURES) {
      if (f.blurb !== undefined) {
        expect(typeof f.blurb).toBe('string')
        expect(f.blurb.trim().length).toBeGreaterThan(0)
      }
    }
  })

  it('keeps the evidence-backed spine features from S9', () => {
    const ids = new Set(FEATURED_FEATURES.map((f) => f.id))
    for (const spine of [9127, 11270, 19948, 9025, 21934]) {
      expect(ids.has(spine)).toBe(true)
    }
  })
})
