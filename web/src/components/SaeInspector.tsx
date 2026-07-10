import { useCallback, useEffect, useRef, useState } from 'react'
import type { BackendChoice, Runner } from '../lib/runner'
import {
  SaeEncoder,
  type FeatureHit,
  type SaeLoadProgress,
  type TokenTopK,
} from '../lib/sae'
import { tokenize, type Token } from '../lib/tokenizer'
import Explainer from './Explainer'
import FeaturePage from './FeaturePage'
import FeatureGallery from './FeatureGallery'

// How many features we compute per token, and how many chips show before the
// token is expanded. K_FULL is the expanded count.
const K_FULL = 8
const COLLAPSED_CHIPS = 3

function mb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1)
}

/**
 * SAE tab — as the user types, shows per token the top-k active SAE features
 * (with plain-language labels where they exist), computed live in the browser.
 *
 * Pipeline per recompute: the model runner produces layer-8 resid_pre
 * (`residualsEnteringBlocks(ids)[8]`); the SAE encoder centers it (the S2 basis
 * transform) and runs the fp16 encoder graph on the same EP the model uses;
 * top-k picks each token's strongest features and resolves their labels.
 *
 * Seam for later stories: `onFeatureClick` is where S8 will route to a full
 * feature page. Left unset it opens the lightweight inline panel below.
 */
export default function SaeInspector({
  prompt,
  backend,
  getRunner,
  modelReady,
  onFeatureClick,
  onSteer,
}: {
  prompt: string
  backend: BackendChoice | null
  getRunner: () => Promise<Runner>
  modelReady: boolean
  onFeatureClick?: (id: number) => void
  /** S9: route "Steer with this feature" from an open FeaturePage to the
   *  Steering Playground. Threaded straight into the overlay's FeaturePage. */
  onSteer?: (id: number) => void
}) {
  const encoderRef = useRef<SaeEncoder | null>(null)
  const computeSeq = useRef(0)

  const [ready, setReady] = useState(false)
  const [busyLoad, setBusyLoad] = useState(false)
  const [loadProg, setLoadProg] = useState<SaeLoadProgress | null>(null)
  const [computing, setComputing] = useState(false)
  const [hits, setHits] = useState<TokenTopK[] | null>(null)
  const [tokens, setTokens] = useState<Token[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [ms, setMs] = useState<number | null>(null)
  const [ep, setEp] = useState<string | null>(null)
  const [hasLabels, setHasLabels] = useState(false)
  // The feature whose full page is open over the inspector (null = none). The
  // inspector stays mounted behind it, so its tokens/hits survive a close.
  const [openFeatureId, setOpenFeatureId] = useState<number | null>(null)

  const load = useCallback(async () => {
    if (!backend) return
    setBusyLoad(true)
    setErr(null)
    try {
      await getRunner() // downloads/instantiates GPT-2 if not already loaded
      if (!encoderRef.current) {
        const enc = await SaeEncoder.create(backend, setLoadProg)
        encoderRef.current = enc
        setEp(enc.ep)
        setHasLabels(enc.hasLabels)
      }
      setReady(true)
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyLoad(false)
    }
  }, [backend, getRunner])

  const compute = useCallback(
    async (text: string) => {
      const enc = encoderRef.current
      if (!enc) return
      const toks = tokenize(text)
      setTokens(toks)
      if (toks.length === 0) {
        setHits(null)
        return
      }
      const seqId = ++computeSeq.current
      setComputing(true)
      setErr(null)
      try {
        const runner = await getRunner()
        const resids = await runner.residualsEnteringBlocks(toks.map((t) => t.id))
        const t0 = performance.now()
        const feats = await enc.encode(resids[8], toks.length)
        const top = enc.topK(feats, toks.length, K_FULL)
        if (seqId !== computeSeq.current) return // a newer keystroke won
        setMs(performance.now() - t0)
        setHits(top)
        if (import.meta.env.DEV) {
          // Proof the fp16 encoder is sane on this EP: count NaNs + active hits.
          let nan = 0
          let active = 0
          let maxv = 0
          for (const tk of top)
            for (const h of tk) {
              if (Number.isNaN(h.value)) nan++
              if (h.value > 0) active++
              if (h.value > maxv) maxv = h.value
            }
          // eslint-disable-next-line no-console
          console.log(
            `[SAE] ep=${enc.ep} tokens=${toks.length} topHitsActive=${active} maxAct=${maxv.toFixed(3)} NaNs=${nan} labelsLoaded=${enc.hasLabels}`,
          )
        }
      } catch (e) {
        if (seqId === computeSeq.current)
          setErr(e instanceof Error ? e.message : String(e))
      } finally {
        if (seqId === computeSeq.current) setComputing(false)
      }
    },
    [getRunner],
  )

  // If the model is already loaded (another tab did it), pull in the small SAE
  // encoder automatically so the tab feels live without a manual click.
  useEffect(() => {
    if (modelReady && !ready && !busyLoad && backend) void load()
  }, [modelReady, ready, busyLoad, backend, load])

  // Debounced recompute as the prompt changes.
  useEffect(() => {
    if (!ready) return
    const h = setTimeout(() => void compute(prompt), 300)
    return () => clearTimeout(h)
  }, [ready, prompt, compute])

  const handleFeatureClick = useCallback(
    (hit: FeatureHit) => {
      // External seam (e.g. S10 gallery) takes over when provided; otherwise
      // open the full feature page as an overlay right here.
      if (onFeatureClick) onFeatureClick(hit.id)
      else setOpenFeatureId(hit.id)
    },
    [onFeatureClick],
  )

  const pct =
    loadProg && loadProg.totalBytes > 0
      ? Math.min(100, (loadProg.loadedBytes / loadProg.totalBytes) * 100)
      : 0

  return (
    <div className="flex flex-col gap-5">
      <Explainer
        id="sae"
        lead={
          <>
            A <span className="text-accent">sparse autoencoder</span> (SAE) takes
            GPT-2's internal state at one layer and rewrites it as a short list of{' '}
            <span className="text-accent">features</span> — most switched off, a
            few switched on. Many of those features line up with something you can
            name: a kind of punctuation, a topic, a grammatical role. This view
            runs the SAE in your browser and shows, for each token, which features
            it switched on.
          </>
        }
        points={[
          {
            label: 'Layer 8',
            text: "The features are read from the residual stream just before block 8 of GPT-2 small — the layer these SAEs are usually shown at.",
          },
          {
            label: 'Chips',
            text: 'Each token lists its strongest features, longest bar first. A named chip carries a human-written label; “feature #12345” is one nobody has labelled yet — most aren’t.',
          },
          {
            label: 'Clicking',
            text: 'Click a token to see more of its features; click a feature to see its label and how often it fires across text.',
          },
        ]}
      />

      {/* Status / load bar */}
      <div className="border-line bg-panel rounded-xl border p-3">
        {!ready ? (
          <div className="flex flex-col gap-3">
            <div className="text-muted font-mono text-[0.68rem] leading-relaxed">
              The SAE tab loads GPT-2 (if it isn't already) plus a{' '}
              <span className="text-fg">37&nbsp;MB</span> encoder, then decomposes
              each token's layer-8 state into features as you type.
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => void load()}
                disabled={busyLoad || !backend}
                className="bg-accent text-site font-display hover:bg-accent-dim disabled:bg-line disabled:text-muted rounded-lg px-4 py-1.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed"
              >
                {busyLoad ? 'Loading…' : 'Load the SAE encoder'}
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
                  <span>
                    {loadProg.instantiating
                      ? 'instantiating encoder…'
                      : '↓ sae_enc_fp16.onnx'}
                  </span>
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
              encoder: <span className="text-accent">{ep ?? '—'}</span>
            </span>
            <span className="text-muted">
              recomputes as you type
              {computing && <span className="text-accent"> · computing…</span>}
            </span>
            {ms != null && (
              <span className="border-line bg-site text-muted rounded-full border px-2.5 py-0.5">
                encode <span className="text-accent">{ms.toFixed(0)} ms</span>
              </span>
            )}
            {!hasLabels && (
              <span className="text-warm">labels unavailable — showing ids</span>
            )}
          </div>
        )}
      </div>

      {err && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 font-mono text-xs text-red-300">
          {err}
        </div>
      )}

      {/* Per-token feature strip once a prompt has features; otherwise the
          curated landing gallery (S10). The gallery loads without the encoder,
          so a first-time visitor gets something to explore immediately, and it
          yields to the per-token view as soon as a prompt is decomposed. */}
      {hits && hits.length > 0 ? (
        <div className="flex flex-col gap-2">
          {hits.map((tokHits, i) => (
            <TokenFeatureRow
              key={i}
              token={tokens[i]?.display ?? '?'}
              tokenId={tokens[i]?.id}
              hits={tokHits}
              onFeatureClick={handleFeatureClick}
            />
          ))}
        </div>
      ) : computing ? (
        <div className="text-muted rounded-lg border border-dashed border-line p-6 text-center font-mono text-sm">
          computing features…
        </div>
      ) : (
        <FeatureGallery
          onOpen={(fid) => setOpenFeatureId(fid)}
          onSteer={onSteer}
          hint={
            !ready
              ? 'These open without loading the encoder. Load it above to break your own prompt into features.'
              : prompt.trim().length === 0
                ? 'Type a prompt above to break it into features.'
                : undefined
          }
        />
      )}

      {openFeatureId != null && (
        <FeaturePage
          id={openFeatureId}
          onClose={() => setOpenFeatureId(null)}
          onSteer={
            onSteer
              ? (fid) => {
                  setOpenFeatureId(null) // close the overlay as we leave the tab
                  onSteer(fid)
                }
              : undefined
          }
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// One token's row of feature chips
// ---------------------------------------------------------------------------

function TokenFeatureRow({
  token,
  tokenId,
  hits,
  onFeatureClick,
}: {
  token: string
  tokenId?: number
  hits: TokenTopK
  onFeatureClick: (hit: FeatureHit) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const max = hits[0]?.value ?? 1
  const shown = expanded ? hits : hits.slice(0, COLLAPSED_CHIPS)
  const hidden = hits.length - shown.length

  return (
    <div className="border-line bg-panel/40 rounded-lg border p-2.5">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start">
        <button
          onClick={() => setExpanded((e) => !e)}
          title={tokenId != null ? `token id ${tokenId}` : undefined}
          className="border-line bg-panel-2 text-fg hover:border-accent-dim flex shrink-0 items-center gap-1.5 self-start rounded border px-2 py-1 font-mono text-[0.72rem] sm:min-w-[5rem]"
        >
          <span className="truncate">{token}</span>
          {hits.length > COLLAPSED_CHIPS && (
            <span className="text-muted">{expanded ? '−' : '+'}</span>
          )}
        </button>

        {hits.length === 0 ? (
          <span className="text-muted self-center font-mono text-[0.66rem]">
            no active features
          </span>
        ) : (
          <div className="flex min-w-0 flex-1 flex-wrap gap-1.5">
            {shown.map((h) => (
              <FeatureChip
                key={h.id}
                hit={h}
                frac={max > 0 ? h.value / max : 0}
                onClick={() => onFeatureClick(h)}
              />
            ))}
            {!expanded && hidden > 0 && (
              <button
                onClick={() => setExpanded(true)}
                className="border-line text-muted hover:text-fg hover:border-accent-dim self-center rounded-md border px-2 py-1 font-mono text-[0.66rem] transition-colors"
              >
                +{hidden} more
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// A single feature chip: label + strength bar
// ---------------------------------------------------------------------------

function FeatureChip({
  hit,
  frac,
  onClick,
}: {
  hit: FeatureHit
  frac: number
  onClick: () => void
}) {
  const title =
    `${hit.label} · activation ${hit.value.toFixed(2)}` +
    (hit.freq > 0 ? ` · fires on ${(hit.freq * 100).toFixed(2)}% of tokens` : '')
  return (
    <button
      onClick={onClick}
      title={title}
      className={`group bg-site inline-flex max-w-[13rem] flex-col gap-1 rounded-md border px-2 py-1 text-left transition-colors ${
        hit.labeled
          ? 'border-accent-dim/60 hover:border-accent'
          : 'border-line hover:border-accent-dim'
      }`}
    >
      <span
        className={`truncate font-mono text-[0.68rem] ${
          hit.labeled ? 'text-fg' : 'text-muted'
        }`}
      >
        {hit.label}
      </span>
      <span className="bg-line block h-1 w-full overflow-hidden rounded-full">
        <span
          className="bg-accent block h-full rounded-full"
          style={{ width: `${Math.max(6, frac * 100)}%` }}
        />
      </span>
    </button>
  )
}
