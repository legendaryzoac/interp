import { useCallback, useEffect, useMemo, useState } from 'react'
import type { Runner } from '../lib/runner'
import { tokenize, labelForId } from '../lib/tokenizer'
import {
  runInduction,
  rankHeads,
  runPatching,
  loadCircuits,
  inductionScoresFromJson,
  type InductionResult,
  type PatchingResult,
  type PatchingPair,
  type CircuitsJson,
} from '../lib/circuits'
import HeadScoreGrid from './HeadScoreGrid'
import PatchingHeatmap from './PatchingHeatmap'

type Sub = 'induction' | 'patching'

// Fixed IOI minimal pair (token-aligned: both 14 tokens, differ at the names).
const IOI_CLEAN = 'When Mary and John went to the store, John gave a drink to'
const IOI_CORRUPT = 'When John and Mary went to the store, Mary gave a drink to'

/** GPT-2 small's canonical induction heads, for the sanity-check callout. */
const KNOWN_INDUCTION = [
  { layer: 5, head: 1 },
  { layer: 5, head: 5 },
  { layer: 6, head: 9 },
  { layer: 7, head: 10 },
]

export default function CircuitsView({
  getRunner,
  modelReady,
}: {
  getRunner: () => Promise<Runner>
  modelReady: boolean
}) {
  const [sub, setSub] = useState<Sub>('induction')
  const [precomputed, setPrecomputed] = useState<CircuitsJson | null>(null)

  useEffect(() => {
    loadCircuits().then(setPrecomputed)
  }, [])

  return (
    <div className="flex flex-col gap-5">
      <div className="text-muted font-mono text-xs leading-relaxed">
        Two circuit probes on GPT-2 small:{' '}
        <span className="text-accent">induction heads</span> (which heads copy a
        token from where it last appeared) and{' '}
        <span className="text-accent">activation patching</span> (which
        residual-stream locations carry the indirect-object identity in an IOI
        prompt).
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setSub('induction')}
          className={`font-display rounded-lg px-3 py-1.5 text-xs transition-colors ${
            sub === 'induction'
              ? 'bg-accent text-site'
              : 'border-line text-muted hover:text-fg border'
          }`}
        >
          Induction heads
        </button>
        <button
          onClick={() => setSub('patching')}
          className={`font-display rounded-lg px-3 py-1.5 text-xs transition-colors ${
            sub === 'patching'
              ? 'bg-accent text-site'
              : 'border-line text-muted hover:text-fg border'
          }`}
        >
          Activation patching (IOI)
        </button>
      </div>

      {sub === 'induction' ? (
        <InductionPanel
          getRunner={getRunner}
          modelReady={modelReady}
          precomputed={precomputed?.induction ?? null}
        />
      ) : (
        <PatchingPanel
          getRunner={getRunner}
          modelReady={modelReady}
          precomputed={precomputed?.patching ?? null}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Induction sub-panel
// ---------------------------------------------------------------------------

function InductionPanel({
  getRunner,
  modelReady,
  precomputed,
}: {
  getRunner: () => Promise<Runner>
  modelReady: boolean
  precomputed: { scores: number[][] } | null
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [live, setLive] = useState<InductionResult | null>(null)
  const [ms, setMs] = useState<number | null>(null)

  // Prefer the live result; fall back to precomputed JSON if present.
  const scores = useMemo(() => {
    if (live) return live.scores
    if (precomputed) return inductionScoresFromJson(precomputed.scores)
    return null
  }, [live, precomputed])

  const source: 'live' | 'precomputed' | null = live
    ? 'live'
    : precomputed
      ? 'precomputed'
      : null

  const top = useMemo(() => (scores ? rankHeads(scores, 6) : []), [scores])

  const handleRun = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      const runner = await getRunner()
      const t0 = performance.now()
      const res = await runInduction(runner, 25)
      setMs(performance.now() - t0)
      setLive(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [getRunner])

  return (
    <div className="flex flex-col gap-4">
      <div className="border-line bg-panel rounded-xl border p-3">
        <div className="text-muted font-mono text-[0.68rem] leading-relaxed">
          A sequence of 25 random tokens repeated once ([A B C … A B C …]) is fed
          through the model. For each head we score how strongly the{' '}
          <span className="text-fg">2nd</span> occurrence of each token attends
          to the position <span className="text-fg">right after</span> its 1st
          occurrence — the induction signature. GPT-2 small's known induction
          heads (L5H1, L5H5, …) should light up.
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            onClick={handleRun}
            disabled={busy}
            className="bg-accent text-site font-display hover:bg-accent-dim disabled:bg-line disabled:text-muted rounded-lg px-4 py-1.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed"
          >
            {busy
              ? 'Running…'
              : source === 'live'
                ? 'Re-run (new random seq)'
                : precomputed
                  ? 'Recompute live'
                  : 'Run induction probe'}
          </button>
          {source && (
            <span className="border-accent-dim text-accent rounded-full border px-2.5 py-0.5 font-mono text-[0.64rem]">
              {source}
            </span>
          )}
          {ms != null && (
            <span className="border-line bg-panel text-muted rounded-full border px-3 py-1 font-mono text-[0.68rem]">
              <span className="text-accent">{ms.toFixed(0)} ms</span>
            </span>
          )}
          {!modelReady && !busy && !precomputed && (
            <span className="text-muted font-mono text-[0.66rem]">
              first run loads the model
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 font-mono text-xs text-red-300">
          {error}
        </div>
      )}

      {scores ? (
        <div className="flex flex-col gap-4 lg:flex-row">
          <div className="border-line bg-panel/40 rounded-xl border p-4">
            <div className="text-muted mb-2 font-mono text-[0.68rem]">
              induction score — layer ↓ · head → · brighter = stronger
            </div>
            <HeadScoreGrid
              scores={scores}
              valueLabel="induction"
              highlightThreshold={0.5}
            />
          </div>
          <div className="border-line bg-panel/40 min-w-0 flex-1 rounded-xl border p-4">
            <div className="text-muted mb-2 font-mono text-[0.68rem]">
              top induction heads
            </div>
            <div className="flex flex-col gap-1.5">
              {top.map((h) => {
                const known = KNOWN_INDUCTION.some(
                  (k) => k.layer === h.layer && k.head === h.head,
                )
                return (
                  <div
                    key={`${h.layer}-${h.head}`}
                    className="border-line bg-site flex items-center justify-between rounded border px-3 py-1.5 font-mono text-xs"
                  >
                    <span>
                      <span className="text-accent">L{h.layer}</span>
                      <span className="text-muted"> · </span>
                      <span className="text-accent">H{h.head}</span>
                      {known && (
                        <span className="text-warm ml-2 text-[0.62rem]">
                          ✦ known GPT-2 induction head
                        </span>
                      )}
                    </span>
                    <span className="text-muted">{h.score.toFixed(3)}</span>
                  </div>
                )
              })}
            </div>
            <div className="text-muted mt-3 font-mono text-[0.62rem] leading-relaxed">
              GPT-2 small's canonical induction heads sit around layers 5-7
              (notably L5H1 and L5H5). A strong match here is the sanity check.
            </div>
          </div>
        </div>
      ) : (
        <div className="text-muted rounded-lg border border-dashed border-line p-6 text-center font-mono text-sm">
          run the induction probe to score every head
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Patching sub-panel
// ---------------------------------------------------------------------------

function PatchingPanel({
  getRunner,
  modelReady,
  precomputed,
}: {
  getRunner: () => Promise<Runner>
  modelReady: boolean
  precomputed: CircuitsJson['patching'] | null
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(
    null,
  )
  const [live, setLive] = useState<PatchingResult | null>(null)
  const [ms, setMs] = useState<number | null>(null)

  // Precomputed JSON -> the same PatchingResult shape the heatmap consumes.
  const fromJson: PatchingResult | null = useMemo(() => {
    if (!precomputed) return null
    let min = Infinity
    let max = -Infinity
    for (const row of precomputed.heatmap)
      for (const v of row) {
        if (v < min) min = v
        if (v > max) max = v
      }
    return {
      tokens: precomputed.tokens,
      cleanTargetLabel: precomputed.clean_target,
      corruptTargetLabel: precomputed.corrupt_target,
      logitDiffClean: precomputed.logit_diff_clean,
      logitDiffCorrupt: precomputed.logit_diff_corrupt,
      heatmap: precomputed.heatmap,
      minDiff: Number.isFinite(min) ? min : 0,
      maxDiff: Number.isFinite(max) ? max : 0,
    }
  }, [precomputed])

  const result = live ?? fromJson
  const source: 'live' | 'precomputed' | null = live
    ? 'live'
    : fromJson
      ? 'precomputed'
      : null

  const handleRun = useCallback(async () => {
    setBusy(true)
    setError(null)
    setProgress(null)
    try {
      const runner = await getRunner()
      const cleanToks = tokenize(IOI_CLEAN)
      const corruptToks = tokenize(IOI_CORRUPT)
      if (cleanToks.length !== corruptToks.length) {
        throw new Error(
          `IOI pair not token-aligned (${cleanToks.length} vs ${corruptToks.length})`,
        )
      }
      // " Mary" / " John" are each a single token; resolve their ids.
      const cleanTargetIds = tokenize(' Mary')
      const corruptTargetIds = tokenize(' John')
      if (cleanTargetIds.length !== 1 || corruptTargetIds.length !== 1) {
        throw new Error('IOI target names did not tokenize to single tokens')
      }
      const pair: PatchingPair = {
        cleanText: IOI_CLEAN,
        corruptText: IOI_CORRUPT,
        cleanTargetId: cleanTargetIds[0].id,
        corruptTargetId: corruptTargetIds[0].id,
        cleanTargetLabel: labelForId(cleanTargetIds[0].id),
        corruptTargetLabel: labelForId(corruptTargetIds[0].id),
      }
      const t0 = performance.now()
      const res = await runPatching(
        runner,
        pair,
        cleanToks.map((t) => t.id),
        corruptToks.map((t) => t.id),
        cleanToks.map((t) => t.display),
        (done, total) => setProgress({ done, total }),
      )
      setMs(performance.now() - t0)
      setLive(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [getRunner])

  const pct = progress ? (progress.done / progress.total) * 100 : 0

  return (
    <div className="flex flex-col gap-4">
      <div className="border-line bg-panel rounded-xl border p-3">
        <div className="text-muted font-mono text-[0.68rem] leading-relaxed">
          A fixed IOI minimal pair. Clean:{' '}
          <span className="text-fg">
            "When Mary and John … John gave a drink to"
          </span>{' '}
          → <span className="text-accent"> Mary</span>. Corrupt swaps the names →{' '}
          <span className="text-warm"> John</span>. For each (layer, position) we
          splice the clean residual entering that block at that position into the
          corrupted run, continue the forward, and measure how much of the clean
          logit-diff (logit Mary − logit John) is recovered. The bright band
          marks the residual locations that carry the answer.
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            onClick={handleRun}
            disabled={busy}
            className="bg-accent text-site font-display hover:bg-accent-dim disabled:bg-line disabled:text-muted rounded-lg px-4 py-1.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed"
          >
            {busy
              ? 'Patching…'
              : source === 'live'
                ? 'Re-run patching'
                : precomputed
                  ? 'Recompute live'
                  : 'Run patching (slow on WASM)'}
          </button>
          {source && (
            <span className="border-accent-dim text-accent rounded-full border px-2.5 py-0.5 font-mono text-[0.64rem]">
              {source}
            </span>
          )}
          {ms != null && (
            <span className="border-line bg-panel text-muted rounded-full border px-3 py-1 font-mono text-[0.68rem]">
              <span className="text-accent">{(ms / 1000).toFixed(1)} s</span>
            </span>
          )}
          {!modelReady && !busy && !precomputed && (
            <span className="text-muted font-mono text-[0.66rem]">
              ~12×14 forwards — slow on the WASM path
            </span>
          )}
        </div>

        {busy && progress && (
          <div className="mt-3">
            <div className="text-muted mb-1 font-mono text-[0.64rem]">
              patching {progress.done} / {progress.total} cells…
            </div>
            <div className="border-line bg-site h-2 w-full overflow-hidden rounded-full border">
              <div
                className="bg-accent h-full transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 font-mono text-xs text-red-300">
          {error}
        </div>
      )}

      {result ? (
        <div className="border-line bg-panel/40 rounded-xl border p-4">
          <div className="mb-3 flex flex-wrap items-center gap-3 font-mono text-[0.68rem]">
            <span className="text-muted">
              clean logit-diff:{' '}
              <span className="text-accent">
                {result.logitDiffClean.toFixed(2)}
              </span>
            </span>
            <span className="text-muted">
              corrupt logit-diff:{' '}
              <span className="text-warm">
                {result.logitDiffCorrupt.toFixed(2)}
              </span>
            </span>
            <span className="text-muted">
              (
              <span className="text-accent">{result.cleanTargetLabel}</span> −{' '}
              <span className="text-warm">{result.corruptTargetLabel}</span>)
            </span>
          </div>
          <PatchingHeatmap
            heatmap={result.heatmap}
            tokens={result.tokens}
            logitDiffClean={result.logitDiffClean}
            logitDiffCorrupt={result.logitDiffCorrupt}
          />
          <div className="text-muted mt-3 font-mono text-[0.62rem] leading-relaxed">
            Expect the strong recovery band around the name-mover / S-inhibition
            layers (~8-10) at the end positions — that's where the
            indirect-object identity is routed to the final token.
          </div>
        </div>
      ) : (
        <div className="text-muted rounded-lg border border-dashed border-line p-6 text-center font-mono text-sm">
          run the patching sweep to see the recovery heatmap
        </div>
      )}
    </div>
  )
}
