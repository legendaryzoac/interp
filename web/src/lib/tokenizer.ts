/**
 * tokenizer.ts — GPT-2 byte-level BPE via `gpt-tokenizer`.
 *
 * We need both the token ids (to feed embed.onnx) and a human-readable label
 * per token for the axis labels in the attention / logit-lens views. GPT-2's
 * byte-level BPE encodes a leading space as the sentinel byte U+0120 (Ġ) and a
 * newline as U+010A (Ċ); we surface those as visible markers (␣ and ⏎) so a
 * viewer can tell " the" from "the".
 *
 * IMPORTANT: GPT-2 uses the r50k_base BPE. The package's default export is the
 * cl100k (GPT-3.5/4) encoding, which produces *different* ids — always import
 * the r50k_base entry point so ids line up with the ONNX embedding table.
 */
import { encode, decode } from 'gpt-tokenizer/encoding/r50k_base'

export interface Token {
  id: number
  /** Decoded text of this single token (may contain a leading space/newline). */
  text: string
  /** Display label with visible whitespace markers. */
  display: string
}

/** Replace leading/embedded whitespace with visible markers for display. */
export function toDisplay(text: string): string {
  if (text.length === 0) return '∅'
  return text
    .replace(/ /g, '␣')
    .replace(/\n/g, '⏎')
    .replace(/\t/g, '⇥')
}

export function encodeIds(text: string): number[] {
  return encode(text)
}

/**
 * Encode a prompt into a list of tokens carrying id + decoded text + display.
 * Each token's text is decoded individually so we can label it precisely.
 */
export function tokenize(text: string): Token[] {
  const ids = encode(text)
  return ids.map((id) => {
    const t = decode([id])
    return { id, text: t, display: toDisplay(t) }
  })
}

export function decodeIds(ids: number[]): string {
  return decode(ids)
}

/** Label for a single token id — used by views that only carry ids. */
export function labelForId(id: number): string {
  return toDisplay(decode([id]))
}
