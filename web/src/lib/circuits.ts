/**
 * circuits.ts — compute + loaders for the Circuits tab.
 *
 * Two mechanistic probes:
 *   1. Induction heads — run a repeated random-token sequence and score each
 *      head by how strongly the 2nd occurrence of a token attends to the token
 *      that FOLLOWED its 1st occurrence (the induction / copy signature).
 *   2. Activation patching (IOI) — patch the residual stream entering each
 *      block at each position from a corrupted run with the clean run's value,
 *      and measure how much the clean logit-diff is recovered.
 *
 * An optional `/circuits.json` (produced by the pipeline agent) lets the tab
 * render instantly; a missing/malformed file must never throw.
 */
import {
  N_LAYERS,
  N_HEADS,
  headMatrix,
  type RunResult,
  type Runner,
} from './runner'

// ---------------------------------------------------------------------------
// Induction heads
// ---------------------------------------------------------------------------

export interface InductionResult {
  /** Flat [N_LAYERS * N_HEADS] induction score, row-major (layer outer). */
  scores: Float32Array
  /** The token ids used (repeated sequence), for display / reproducibility. */
  tokenIds: number[]
  /** Length of the repeated unit (so the sequence is 2 * unitLen tokens). */
  unitLen: number
}

/**
 * Build a repeated random-token sequence: `unitLen` random ids repeated once,
 * i.e. [A B C ... A B C ...] of length 2*unitLen. Ids are drawn from a "safe"
 * mid-vocab band to avoid special/byte-fallback tokens skewing attention.
 * `rng` defaults to Math.random but is injectable for deterministic tests.
 */
export function makeRepeatedSequence(
  unitLen = 25,
  rng: () => number = Math.random,
): number[] {
  const lo = 1000
  const hi = 40000 // comfortably inside GPT-2's 50257 vocab, skips the tail
  const unit: number[] = []
  for (let i = 0; i < unitLen; i++) {
    unit.push(lo + Math.floor(rng() * (hi - lo)))
  }
  return [...unit, ...unit]
}

/**
 * Per-head induction score from a run on a repeated sequence of length
 * 2*unitLen. For each 2nd-occurrence position i in [unitLen, 2*unitLen), the
 * "induction target" is the position right after the token's 1st occurrence:
 * firstOcc = i - unitLen, target = firstOcc + 1. We average the attention
 * weight pattern[i][target] over all valid i (target must be < i, always true
 * here since target = i - unitLen + 1 <= i). Returns flat [N_LAYERS*N_HEADS].
 */
export function inductionScores(run: RunResult, unitLen: number): Float32Array {
  const seq = run.seq
  const out = new Float32Array(N_LAYERS * N_HEADS)
  if (seq < 2 * unitLen) return out
  for (let l = 0; l < N_LAYERS; l++) {
    const pat = run.patterns[l]
    for (let h = 0; h < N_HEADS; h++) {
      const m = headMatrix(pat, h, seq) // [seq, seq] view: m[q*seq + k]
      let acc = 0
      let count = 0
      for (let i = unitLen; i < 2 * unitLen; i++) {
        const target = i - unitLen + 1
        if (target >= 0 && target < seq) {
          acc += m[i * seq + target]
          count++
        }
      }
      out[l * N_HEADS + h] = count > 0 ? acc / count : 0
    }
  }
  return out
}

/** Rank heads by induction score, descending. Returns [{layer,head,score}]. */
export function rankHeads(
  scores: Float32Array | number[],
  topN = 8,
): { layer: number; head: number; score: number }[] {
  const list: { layer: number; head: number; score: number }[] = []
  for (let l = 0; l < N_LAYERS; l++)
    for (let h = 0; h < N_HEADS; h++)
      list.push({ layer: l, head: h, score: scores[l * N_HEADS + h] })
  list.sort((a, b) => b.score - a.score)
  return list.slice(0, topN)
}

/**
 * Run the induction probe end-to-end: build a repeated sequence, run it, score.
 */
export async function runInduction(
  runner: Runner,
  unitLen = 25,
  rng: () => number = Math.random,
): Promise<InductionResult> {
  const tokenIds = makeRepeatedSequence(unitLen, rng)
  const run = await runner.run(tokenIds)
  return { scores: inductionScores(run, unitLen), tokenIds, unitLen }
}

// ---------------------------------------------------------------------------
// Activation patching (IOI)
// ---------------------------------------------------------------------------

/** The fixed minimal pair for the IOI patching demo. */
export interface PatchingPair {
  cleanText: string
  corruptText: string
  /** Token id of the correct answer for the clean prompt (e.g. " Mary"). */
  cleanTargetId: number
  /** Token id of the answer the corrupted prompt pushes toward (e.g. " John"). */
  corruptTargetId: number
  /** Display label for the clean target (" Mary"). */
  cleanTargetLabel: string
  corruptTargetLabel: string
}

export interface PatchingResult {
  /** Display tokens of the (clean) prompt — columns of the heatmap. */
  tokens: string[]
  cleanTargetLabel: string
  corruptTargetLabel: string
  /** Logit-diff (clean_target - corrupt_target) at the final position, clean run. */
  logitDiffClean: number
  /** Same for the corrupted run. */
  logitDiffCorrupt: number
  /** heatmap[layer][pos] = recovered logit-diff after patching (L,pos). */
  heatmap: number[][]
  /** Convenience: min/max recovered value for color scaling. */
  minDiff: number
  maxDiff: number
}

/** logit[a] - logit[b] at the final position of a full-vocab final-row buffer. */
export function logitDiffRow(
  finalRow: Float32Array,
  aId: number,
  bId: number,
): number {
  return finalRow[aId] - finalRow[bId]
}

/**
 * Activation patching over the residual stream entering each block at each
 * position. For each (layer L, position p): take the corrupted residual
 * entering block L, overwrite position p with the clean residual entering
 * block L, continue the forward from block L, and record the final-position
 * logit-diff (clean_target - corrupt_target). A value near `logitDiffClean`
 * means position p at layer L carries the information that fixes the answer.
 *
 * This is ~N_LAYERS * seq continuation-forwards — slow on WASM int8 — so the
 * caller should show progress; `onProgress(done, total)` is invoked per cell.
 */
export async function runPatching(
  runner: Runner,
  pair: PatchingPair,
  cleanTokenIds: number[],
  corruptTokenIds: number[],
  cleanTokens: string[],
  onProgress?: (done: number, total: number) => void,
): Promise<PatchingResult> {
  const seq = cleanTokenIds.length
  if (corruptTokenIds.length !== seq)
    throw new Error(
      `clean/corrupt length mismatch: ${seq} vs ${corruptTokenIds.length} — the minimal pair must be token-aligned`,
    )

  const { cleanTargetId, corruptTargetId } = pair

  // Residuals entering each block for both runs (embed + block outputs 0..10).
  const cleanEntering = await runner.residualsEnteringBlocks(cleanTokenIds)
  const corruptEntering = await runner.residualsEnteringBlocks(corruptTokenIds)

  // Baseline logit-diffs (unpatched). Run the corrupted forward from block 0.
  const cleanFinal = await runner.continueFromBlock(cleanEntering[0], 0, seq)
  const corruptFinal = await runner.continueFromBlock(corruptEntering[0], 0, seq)
  const logitDiffClean = logitDiffRow(cleanFinal, cleanTargetId, corruptTargetId)
  const logitDiffCorrupt = logitDiffRow(
    corruptFinal,
    cleanTargetId,
    corruptTargetId,
  )

  const total = N_LAYERS * seq
  let done = 0
  const heatmap: number[][] = []
  let minDiff = Infinity
  let maxDiff = -Infinity

  for (let l = 0; l < N_LAYERS; l++) {
    const row: number[] = []
    const base = corruptEntering[l] // residual entering block L, corrupted
    for (let p = 0; p < seq; p++) {
      // Copy the corrupted entry and splice in the clean vector at position p.
      const patched = new Float32Array(base) // copy
      const d = 768
      patched.set(cleanEntering[l].subarray(p * d, p * d + d), p * d)
      const final = await runner.continueFromBlock(patched, l, seq)
      const diff = logitDiffRow(final, cleanTargetId, corruptTargetId)
      row.push(diff)
      if (diff < minDiff) minDiff = diff
      if (diff > maxDiff) maxDiff = diff
      done++
      onProgress?.(done, total)
    }
    heatmap.push(row)
    // Yield a macrotask each row so React can paint the progress bar. The tight
    // await-loop over WASM forwards only yields microtasks, which starve
    // React's render (macrotask) and freeze the bar; one setTimeout(0) per row
    // (12 total) is negligible overhead and keeps the UI responsive.
    await new Promise<void>((r) => setTimeout(r, 0))
  }

  return {
    tokens: cleanTokens,
    cleanTargetLabel: pair.cleanTargetLabel,
    corruptTargetLabel: pair.corruptTargetLabel,
    logitDiffClean,
    logitDiffCorrupt,
    heatmap,
    minDiff: Number.isFinite(minDiff) ? minDiff : 0,
    maxDiff: Number.isFinite(maxDiff) ? maxDiff : 0,
  }
}

// ---------------------------------------------------------------------------
// Optional precomputed loader (/circuits.json)
// ---------------------------------------------------------------------------

export interface CircuitsJson {
  induction?: { scores: number[][] } // [layer][head]
  patching?: {
    tokens: string[]
    clean_target: string
    corrupt_target: string
    heatmap: number[][] // [layer][position]
    logit_diff_clean: number
    logit_diff_corrupt: number
  }
}

/** Fetch /circuits.json. Returns null on any error/absence — never throws. */
export async function loadCircuits(): Promise<CircuitsJson | null> {
  try {
    const res = await fetch('/circuits.json', { cache: 'no-cache' })
    if (!res.ok) return null
    const json = (await res.json()) as unknown
    if (!json || typeof json !== 'object') return null
    return json as CircuitsJson
  } catch {
    return null
  }
}

/** Flatten a precomputed [layer][head] induction matrix to the flat scores
 *  array the grid consumes. Tolerates ragged/short input defensively. */
export function inductionScoresFromJson(matrix: number[][]): Float32Array {
  const out = new Float32Array(N_LAYERS * N_HEADS)
  for (let l = 0; l < Math.min(N_LAYERS, matrix.length); l++) {
    const rowArr = matrix[l] ?? []
    for (let h = 0; h < Math.min(N_HEADS, rowArr.length); h++) {
      out[l * N_HEADS + h] = rowArr[h]
    }
  }
  return out
}
