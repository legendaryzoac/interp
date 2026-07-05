/**
 * viewModel.ts — the shape the UI renders, produced from either a live model
 * run or a precomputed gallery entry. Keeping one shape means the attention /
 * logit-lens components don't care where the numbers came from.
 */
import { N_LAYERS, VOCAB_SIZE, topKAtPosition, type RunResult } from './runner'
import { labelForId } from './tokenizer'
import {
  galleryPatternsToFloat,
  type GalleryPrompt,
} from './gallery'

export interface LensCell {
  /** Top-5 predictions, descending prob. */
  top5: { token: string; prob: number }[]
}

export interface ResultView {
  source: 'live' | 'precomputed'
  seq: number
  /** Display labels per token position. */
  tokens: string[]
  /** Per-layer attention, each [heads, seq, seq] flat Float32Array. */
  patterns: Float32Array[]
  /** lens[layer][pos] top-5 cell. */
  lens: LensCell[][]
}

/** Build a ResultView from a live run + the token labels used to drive it. */
export function viewFromRun(run: RunResult, tokens: string[]): ResultView {
  const { seq, patterns, lensLogits } = run
  const lens: LensCell[][] = []
  for (let l = 0; l < N_LAYERS; l++) {
    const row: LensCell[] = []
    const logits = lensLogits[l]
    for (let pos = 0; pos < seq; pos++) {
      const top = topKAtPosition(logits, pos, seq, 5)
      row.push({
        top5: top.map((e) => ({ token: labelForId(e.token), prob: e.prob })),
      })
    }
    lens.push(row)
  }
  void VOCAB_SIZE
  return { source: 'live', seq, tokens, patterns, lens }
}

/** Build a ResultView from a precomputed gallery prompt. */
export function viewFromGallery(g: GalleryPrompt): ResultView {
  const seq = g.token_ids.length
  const patterns = galleryPatternsToFloat(g.patterns_u8)
  const lens: LensCell[][] = g.lens_top5.map((layer) =>
    layer.map((cell) => ({
      top5: cell.map((c) => ({ token: c.t, prob: c.p })),
    })),
  )
  return { source: 'precomputed', seq, tokens: g.tokens, patterns, lens }
}
