/**
 * Validates our gpt-tokenizer wiring against the HF ground-truth fixture the
 * model pipeline produces at model-pipeline/out/token_fixture.json.
 *
 * Fixture is expected to be an array (or {prompts:[...]}) of entries that carry
 * a prompt string and its HF GPT-2 token ids. We accept a few plausible field
 * names so a small schema drift doesn't fail the build. If the file doesn't
 * exist yet (parallel agent hasn't produced it), the test skips gracefully.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { encodeIds } from './tokenizer'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const FIXTURE = path.resolve(
  HERE,
  '../../../model-pipeline/out/token_fixture.json',
)

interface FixtureEntry {
  text?: string
  prompt?: string
  ids?: number[]
  token_ids?: number[]
  input_ids?: number[]
}

function loadFixture(): FixtureEntry[] | null {
  if (!existsSync(FIXTURE)) return null
  try {
    const raw = JSON.parse(readFileSync(FIXTURE, 'utf-8'))
    if (Array.isArray(raw)) return raw as FixtureEntry[]
    if (raw && Array.isArray(raw.prompts)) return raw.prompts as FixtureEntry[]
    return null
  } catch {
    return null
  }
}

const fixture = loadFixture()

// Always-on golden cases (independent of the fixture) that pin the r50k_base
// GPT-2 encoding — guards against the package's default cl100k export sneaking
// back in, which would silently misalign ids with the ONNX embedding table.
describe('gpt-tokenizer GPT-2 (r50k_base) golden cases', () => {
  const GOLDEN: [string, number[]][] = [
    ['Hello world', [15496, 995]],
    ['The quick brown fox jumps over the lazy dog.',
      [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]],
  ]
  for (const [text, ids] of GOLDEN) {
    it(`encodes ${JSON.stringify(text)}`, () => {
      expect(encodeIds(text)).toEqual(ids)
    })
  }
})

describe('gpt-tokenizer parity vs HF fixture', () => {
  if (!fixture) {
    it.skip('token_fixture.json not present yet — skipping parity check', () => {})
    return
  }

  it('matches HF token ids for every fixture prompt', () => {
    let checked = 0
    for (const entry of fixture) {
      const text = entry.text ?? entry.prompt
      const expected = entry.ids ?? entry.token_ids ?? entry.input_ids
      if (text == null || expected == null) continue
      const got = encodeIds(text)
      expect(got, `mismatch for prompt: ${JSON.stringify(text)}`).toEqual(expected)
      checked++
    }
    expect(checked, 'fixture had no usable entries').toBeGreaterThan(0)
  })
})
