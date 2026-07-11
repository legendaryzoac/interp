import { useCallback, useEffect, useRef, useState } from 'react'
import type { BackendChoice, Runner } from '../lib/runner'
import {
  SteeringDecoder,
  addSteeringVector,
  sampleToken,
  logitsAreFinite,
  mulberry32,
  yieldToUI,
  loadLabeledFeatures,
  searchLabeledFeatures,
  type SteeringLoadProgress,
  type LabeledFeature,
} from '../lib/steering'
import { tokenize, decodeIdsForDisplay } from '../lib/tokenizer'
import Explainer from './Explainer'

// Layer the SAE reads/writes (BASIS_CONTRACT: layer-8 resid_pre).
const STEER_LAYER = 8
// Cap total sequence length so a long prompt + generation stays cheap on WASM.
const MAX_SEQ = 128

// Crowd-pleaser presets — chosen from the harvest's logit-lens "toward" tokens
// and confirmed to bend the text cleanly (see S9 calibration). Each is a
// concrete, legible concept rather than a punctuation/formatting feature.
const PRESETS: { id: number; name: string; blurb: string }[] = [
  { id: 9127, name: 'NFL / pro football', blurb: 'quarterbacks, rookies, the offseason' },
  { id: 11270, name: 'Philosophy', blurb: 'philosophers, empirical, conceptual reasoning' },
  { id: 19948, name: 'Early-1900s history', blurb: 'postwar, 1914, 1936 — dates and eras' },
  { id: 9025, name: 'UK politics', blurb: 'Scotland, Labour, £, British spellings' },
]

function mb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1)
}

/**
 * S9 — the SAE Steering Playground. Pick a feature, set a strength (alpha), and
 * generate: the model completes the prompt twice from the same point — once
 * untouched (baseline) and once with `alpha * W_dec[feature]` added to the
 * layer-8 residual at every position (steered). The two completions stream in
 * side by side so the causal effect of the feature is visible directly.
 *
 * Per BASIS_CONTRACT §S9 the steering add is in the RAW residual basis (NO
 * centering — that's encoder-input-only). Baseline and steered share the same
 * seeded sampler, so any divergence is the steering, not RNG.
 */
export default function SteeringPlayground({
  prompt,
  backend,
  getRunner,
  modelReady,
  initialFeatureId,
}: {
  prompt: string
  backend: BackendChoice | null
  getRunner: () => Promise<Runner>
  modelReady: boolean
  /** Preselected feature when arriving via a FeaturePage "Steer" button. */
  initialFeatureId: number | null
}) {
  const decoderRef = useRef<SteeringDecoder | null>(null)
  const genSeq = useRef(0)

  const [ready, setReady] = useState(false)
  const [busyLoad, setBusyLoad] = useState(false)
  const [loadProg, setLoadProg] = useState<SteeringLoadProgress | null>(null)
  const [ep, setEp] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)

  // Feature selection + label search.
  const [featureId, setFeatureId] = useState<number | null>(initialFeatureId)
  const [labeled, setLabeled] = useState<LabeledFeature[]>([])
  const [query, setQuery] = useState('')

  // Controls.
  const [localPrompt, setLocalPrompt] = useState(prompt)
  const [alpha, setAlpha] = useState(45)
  const [temperature, setTemperature] = useState(0.7)
  const [maxNewTokens, setMaxNewTokens] = useState(24)

  // Generation output.
  const [generating, setGenerating] = useState(false)
  const [baselineIds, setBaselineIds] = useState<number[]>([])
  const [steeredIds, setSteeredIds] = useState<number[]>([])
  const [step, setStep] = useState(0)
  const [perTokenMs, setPerTokenMs] = useState<number | null>(null)
  const [ranAlpha, setRanAlpha] = useState<number | null>(null)
  const [ranFeature, setRanFeature] = useState<number | null>(null)

  // Keep the local prompt in step with an updated shared prompt until the user
  // edits it here (once they've typed, respect their local copy).
  const editedRef = useRef(false)
  useEffect(() => {
    if (!editedRef.current) setLocalPrompt(prompt)
  }, [prompt])

  // A steer request from a FeaturePage picks the feature and shows it here.
  useEffect(() => {
    if (initialFeatureId != null) setFeatureId(initialFeatureId)
  }, [initialFeatureId])

  // Labels for search + preset display (memoised in the lib; empty on failure).
  useEffect(() => {
    void loadLabeledFeatures().then(setLabeled)
  }, [])

  const labelFor = useCallback(
    (id: number): string => {
      const f = labeled.find((x) => x.id === id)
      return f?.label ?? `feature #${id}`
    },
    [labeled],
  )

  const load = useCallback(async () => {
    if (!backend) return
    setBusyLoad(true)
    setErr(null)
    try {
      await getRunner() // downloads/instantiates GPT-2 if not already loaded
      if (!decoderRef.current) {
        decoderRef.current = await SteeringDecoder.create(setLoadProg)
      }
      setEp(backend.ep)
      setReady(true)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyLoad(false)
    }
  }, [backend, getRunner])

  // Auto-pull the decoder once the model is ready (mirrors the SAE tab).
  useEffect(() => {
    if (modelReady && !ready && !busyLoad && backend) void load()
  }, [modelReady, ready, busyLoad, backend, load])

  const stop = useCallback(() => {
    genSeq.current++ // any in-flight loop sees a stale id and bails
    setGenerating(false)
  }, [])

  const generate = useCallback(async () => {
    const decoder = decoderRef.current
    if (!decoder || featureId == null) return
    const toks = tokenize(localPrompt)
    if (toks.length === 0) {
      setErr('prompt produced no tokens')
      return
    }
    const promptIds = toks.map((t) => t.id)
    const vec = decoder.row(featureId)

    const seqId = ++genSeq.current
    setGenerating(true)
    setErr(null)
    setBaselineIds([])
    setSteeredIds([])
    setStep(0)
    setPerTokenMs(null)
    setRanAlpha(alpha)
    setRanFeature(featureId)

    // Shared seed → baseline and steered draw the same uniforms; any divergence
    // is the steering, not the RNG. (At temperature 0 the RNG is unused.)
    const seed = (Math.random() * 0x7fffffff) | 0
    const rngBase = mulberry32(seed)
    const rngSteer = mulberry32(seed)

    const baseGen: number[] = []
    const steerGen: number[] = []
    const runner = await getRunner()
    let totalMs = 0
    let steps = 0

    try {
      for (let i = 0; i < maxNewTokens; i++) {
        if (seqId !== genSeq.current) return // superseded / stopped

        const t0 = performance.now()

        // --- baseline (no add) ---
        const baseSeqIds = [...promptIds, ...baseGen]
        if (baseSeqIds.length < MAX_SEQ) {
          const entering = await runner.residualsEnteringBlocks(baseSeqIds)
          const logits = await runner.continueFromBlock(
            entering[STEER_LAYER],
            STEER_LAYER,
            baseSeqIds.length,
          )
          if (i === 0 && !logitsAreFinite(logits))
            throw new Error('baseline logits are all NaN — bad EP numerics')
          // Nucleus + repetition penalty use the sampler defaults; `generated`
          // is this stream's own tokens so far (identical to steered at alpha=0).
          baseGen.push(sampleToken(logits, temperature, rngBase, { generated: baseGen }))
          if (seqId !== genSeq.current) return
          setBaselineIds([...baseGen])
        }

        // --- steered (add alpha * W_dec[f] in the RAW basis) ---
        const steerSeqIds = [...promptIds, ...steerGen]
        if (steerSeqIds.length < MAX_SEQ) {
          const entering = await runner.residualsEnteringBlocks(steerSeqIds)
          const steered = addSteeringVector(
            entering[STEER_LAYER],
            vec,
            alpha,
            steerSeqIds.length,
          )
          const logits = await runner.continueFromBlock(
            steered,
            STEER_LAYER,
            steerSeqIds.length,
          )
          if (i === 0 && !logitsAreFinite(logits))
            throw new Error('steered logits are all NaN — check basis/alpha')
          steerGen.push(sampleToken(logits, temperature, rngSteer, { generated: steerGen }))
          if (seqId !== genSeq.current) return
          setSteeredIds([...steerGen])
        }

        totalMs += performance.now() - t0
        steps++
        setStep(i + 1)
        setPerTokenMs(totalMs / steps)

        // Yield so tokens paint and the UI stays responsive (unthrottled even
        // if the user tabs away — see yieldToUI).
        await yieldToUI()
      }
    } catch (e) {
      if (seqId === genSeq.current)
        setErr(e instanceof Error ? e.message : String(e))
    } finally {
      if (seqId === genSeq.current) setGenerating(false)
    }
  }, [featureId, localPrompt, alpha, temperature, maxNewTokens, getRunner])

  const results = searchLabeledFeatures(labeled, query, 24)
  const pct =
    loadProg && loadProg.totalBytes > 0
      ? Math.min(100, (loadProg.loadedBytes / loadProg.totalBytes) * 100)
      : 0
  const genPct = maxNewTokens > 0 ? Math.min(100, (step / maxNewTokens) * 100) : 0
  const canGenerate = ready && featureId != null && !generating

  return (
    <div className="flex flex-col gap-5">
      <Explainer
        id="steering"
        lead={
          <>
            An SAE feature isn&rsquo;t just a readout — it&rsquo;s a{' '}
            <span className="text-accent">direction</span> you can push on. Here
            you pick a feature, choose how hard to push (the{' '}
            <span className="text-accent">strength</span>), and GPT-2 finishes
            your prompt <span className="text-accent">twice</span>: once as
            normal, and once with that feature&rsquo;s direction added into its
            layer-8 state. When it works you can watch the second completion bend
            toward the feature&rsquo;s concept — the same idea as
            &ldquo;Golden&nbsp;Gate&nbsp;Claude&rdquo;, shrunk to GPT-2.
          </>
        }
        points={[
          {
            label: 'Strength',
            text: 'How much of the feature to add. 0 is the untouched model; the effect usually becomes visible around +40 and grows from there; negative pushes away from the concept. Crank it high enough and the text turns stilted or single-minded — that ceiling is part of the demo.',
          },
          {
            label: 'Same starting point',
            text: 'Both sides use the same prompt and the same sampler seed, so the only difference between them is the steering.',
          },
          {
            label: 'Under the hood',
            text: 'The feature’s decoder vector is added to the residual stream just before block 8, then the forward pass continues. No retraining, no prompt tricks — a direct edit to the model’s internal state.',
          },
        ]}
      />

      {/* Load / status bar */}
      <div className="border-line bg-panel rounded-xl border p-3">
        {!ready ? (
          <div className="flex flex-col gap-3">
            <div className="text-muted font-mono text-[0.68rem] leading-relaxed">
              Steering loads GPT-2 (if it isn&rsquo;t already) plus a{' '}
              <span className="text-fg">37&nbsp;MB</span> decoder matrix (one row
              per feature), then generates in your browser.
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => void load()}
                disabled={busyLoad || !backend}
                className="bg-accent text-site font-display hover:bg-accent-dim disabled:bg-line disabled:text-muted rounded-lg px-4 py-1.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed"
              >
                {busyLoad ? 'Loading…' : 'Load steering'}
              </button>
              {!modelReady && !busyLoad && (
                <span className="text-muted font-mono text-[0.66rem]">
                  first load also downloads the model
                </span>
              )}
            </div>
            {busyLoad && loadProg && (
              <div className="w-full max-w-md">
                <div className="bg-site border-line h-2 w-full overflow-hidden rounded-full border">
                  <div
                    className="bg-accent h-full rounded-full transition-[width] duration-150"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <div className="text-muted mt-1 flex justify-between font-mono text-[0.66rem]">
                  <span>{loadProg.ready ? 'decoding rows…' : '↓ w_dec_fp16.bin'}</span>
                  <span>
                    {mb(loadProg.loadedBytes)} / {mb(loadProg.totalBytes)} MB
                  </span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-3 font-mono text-[0.68rem]">
            <span className="text-muted">
              runtime: <span className="text-accent">{ep ?? '—'}</span>
            </span>
            {perTokenMs != null && (
              <span className="border-line bg-site text-muted rounded-full border px-2.5 py-0.5">
                ~<span className="text-accent">{(perTokenMs / 1000).toFixed(1)} s</span>{' '}
                / token (baseline + steered)
              </span>
            )}
            {generating && (
              <span className="text-accent">
                generating… {step}/{maxNewTokens}
              </span>
            )}
          </div>
        )}
      </div>

      {err && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 font-mono text-xs text-red-300">
          {err}
        </div>
      )}

      {/* Feature picker */}
      <div className="border-line bg-panel/40 flex flex-col gap-3 rounded-xl border p-4">
        <div className="text-muted font-mono text-[0.7rem] tracking-wide">
          1 · pick a feature to steer with
        </div>

        {/* Presets */}
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => setFeatureId(p.id)}
              title={p.blurb}
              className={`rounded-lg border px-3 py-1.5 text-left font-mono text-[0.7rem] transition-colors ${
                featureId === p.id
                  ? 'border-accent bg-accent/10 text-fg'
                  : 'border-line text-muted hover:border-accent-dim hover:text-fg'
              }`}
            >
              <div className="text-fg">{p.name}</div>
              <div className="text-muted text-[0.62rem]">#{p.id} · {p.blurb}</div>
            </button>
          ))}
        </div>

        {/* Label search */}
        <div className="flex flex-col gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="…or search feature labels (e.g. “legal”, “sports”, or a feature id)"
            spellCheck={false}
            className="border-line bg-site text-fg focus:border-accent placeholder:text-muted/60 rounded-lg border px-3 py-2 font-mono text-xs outline-none"
          />
          {query.trim() && (
            <div className="border-line max-h-48 overflow-y-auto rounded-lg border">
              {results.length === 0 ? (
                <div className="text-muted p-3 font-mono text-[0.68rem]">
                  no labelled features match — most of the 24,576 are unlabelled
                </div>
              ) : (
                results.map((f) => (
                  <button
                    key={f.id}
                    onClick={() => {
                      setFeatureId(f.id)
                      setQuery('')
                    }}
                    className="hover:bg-panel-2 flex w-full items-center justify-between gap-3 px-3 py-1.5 text-left"
                  >
                    <span className="text-fg truncate font-mono text-[0.7rem]">
                      {f.label}
                    </span>
                    <span className="text-muted shrink-0 font-mono text-[0.62rem]">
                      #{f.id}
                      {f.hasDashboard ? ' ✦' : ''}
                    </span>
                  </button>
                ))
              )}
            </div>
          )}
        </div>

        {/* Selected */}
        <div className="border-line bg-site flex flex-wrap items-center gap-2 rounded-lg border p-3">
          <span className="text-muted font-mono text-[0.66rem]">steering with:</span>
          {featureId == null ? (
            <span className="text-muted font-mono text-[0.7rem]">
              nothing selected yet
            </span>
          ) : (
            <>
              <span className="text-fg font-mono text-[0.72rem]">
                {labelFor(featureId)}
              </span>
              <span className="border-accent-dim text-accent rounded-full border px-2 py-0.5 font-mono text-[0.6rem]">
                #{featureId}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Prompt + controls */}
      <div className="border-line bg-panel/40 flex flex-col gap-4 rounded-xl border p-4">
        <div className="text-muted font-mono text-[0.7rem] tracking-wide">
          2 · prompt &amp; strength
        </div>

        <textarea
          value={localPrompt}
          onChange={(e) => {
            editedRef.current = true
            setLocalPrompt(e.target.value)
          }}
          rows={2}
          spellCheck={false}
          placeholder="Enter a prompt for GPT-2 to continue…"
          className="border-line bg-site text-fg focus:border-accent placeholder:text-muted/60 min-h-[3.2rem] w-full resize-y rounded-lg border px-3 py-2 font-mono text-sm outline-none"
        />

        {/* Alpha slider */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between font-mono text-[0.68rem]">
            <span className="text-muted">
              strength (alpha) —{' '}
              <span className="text-fg">
                {alpha > 0 ? `+${alpha}` : alpha}
              </span>{' '}
              {alpha === 0 ? (
                <span className="text-muted">(no steering)</span>
              ) : alpha > 0 ? (
                <span className="text-accent">toward the concept</span>
              ) : (
                <span className="text-warm">away from it</span>
              )}
            </span>
          </div>
          <input
            type="range"
            min={-90}
            max={90}
            step={1}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="accent-accent w-full"
          />
          <div className="flex flex-wrap gap-1.5">
            {[-60, -30, 0, 30, 45, 60, 80].map((a) => (
              <button
                key={a}
                onClick={() => setAlpha(a)}
                className={`rounded border px-2 py-0.5 font-mono text-[0.62rem] transition-colors ${
                  alpha === a
                    ? 'border-accent text-accent'
                    : 'border-line text-muted hover:text-fg'
                }`}
              >
                {a > 0 ? `+${a}` : a}
              </button>
            ))}
          </div>
        </div>

        {/* Length + temperature */}
        <div className="flex flex-wrap gap-x-8 gap-y-3">
          <label className="flex flex-col gap-1 font-mono text-[0.66rem]">
            <span className="text-muted">
              new tokens: <span className="text-fg">{maxNewTokens}</span>
            </span>
            <input
              type="range"
              min={8}
              max={48}
              step={4}
              value={maxNewTokens}
              onChange={(e) => setMaxNewTokens(Number(e.target.value))}
              className="accent-accent w-40"
            />
          </label>
          <label className="flex flex-col gap-1 font-mono text-[0.66rem]">
            <span className="text-muted">
              temperature:{' '}
              <span className="text-fg">
                {temperature === 0 ? 'greedy' : temperature.toFixed(1)}
              </span>
            </span>
            <input
              type="range"
              min={0}
              max={1.2}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="accent-accent w-40"
            />
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {generating ? (
            <button
              onClick={stop}
              className="border-line text-fg hover:border-accent-dim font-display rounded-lg border px-5 py-2 text-sm font-semibold transition-colors"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={() => void generate()}
              disabled={!canGenerate}
              className="bg-accent text-site font-display hover:bg-accent-dim disabled:bg-line disabled:text-muted rounded-lg px-5 py-2 text-sm font-semibold transition-colors disabled:cursor-not-allowed"
            >
              {ready ? 'Generate both' : 'Load steering first'}
            </button>
          )}
          {featureId == null && ready && (
            <span className="text-muted font-mono text-[0.66rem]">
              pick a feature above to enable generation
            </span>
          )}
          {generating && (
            <div className="min-w-[8rem] flex-1">
              <div className="bg-site border-line h-1.5 w-full overflow-hidden rounded-full border">
                <div
                  className="bg-accent h-full transition-[width]"
                  style={{ width: `${genPct}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Side-by-side output */}
      {(baselineIds.length > 0 || steeredIds.length > 0 || generating) && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <CompletionColumn
            title="Baseline"
            subtitle="untouched model"
            tone="base"
            promptText={localPrompt}
            generatedText={decodeIdsForDisplay(baselineIds)}
          />
          <CompletionColumn
            title="Steered"
            subtitle={
              ranFeature != null
                ? `${labelFor(ranFeature)} · ${(ranAlpha ?? 0) > 0 ? '+' : ''}${ranAlpha ?? 0}`
                : 'steered'
            }
            tone="steer"
            promptText={localPrompt}
            generatedText={decodeIdsForDisplay(steeredIds)}
          />
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// One completion column
// ---------------------------------------------------------------------------

function CompletionColumn({
  title,
  subtitle,
  tone,
  promptText,
  generatedText,
}: {
  title: string
  subtitle: string
  tone: 'base' | 'steer'
  promptText: string
  generatedText: string
}) {
  const accentBorder = tone === 'steer' ? 'border-accent-dim/60' : 'border-line'
  return (
    <div className={`bg-site rounded-xl border ${accentBorder} flex flex-col`}>
      <div className="border-line flex items-baseline justify-between gap-2 border-b px-4 py-2">
        <span
          className={`font-display text-sm font-semibold ${
            tone === 'steer' ? 'text-accent' : 'text-fg'
          }`}
        >
          {title}
        </span>
        <span className="text-muted truncate font-mono text-[0.62rem]">
          {subtitle}
        </span>
      </div>
      <div className="p-4 text-sm leading-relaxed break-words">
        <span className="text-muted">{promptText}</span>
        <span className={tone === 'steer' ? 'text-accent' : 'text-fg'}>
          {generatedText}
        </span>
        {generatedText.length === 0 && (
          <span className="text-muted/60 font-mono text-xs"> …</span>
        )}
      </div>
    </div>
  )
}
