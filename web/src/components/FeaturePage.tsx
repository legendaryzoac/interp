import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  decodeActs,
  histogramToBars,
  loadFeature,
  type FeatureExample,
  type FeatureView,
  type HistBar,
  type LabelConfidence,
  type LogitLensPair,
} from '../lib/featurePage'
import { toDisplay } from '../lib/tokenizer'

/**
 * FeaturePage — the full page for one SAE feature, opened as a dismissible
 * overlay from the S7 Token Inspector (the inspector stays mounted behind it,
 * so its tokens/state are intact on close).
 *
 * It shows what the feature detects (label + confidence), its strongest
 * real-text examples with per-token activation shading, its activation-frequency
 * histogram, and the tokens it pushes the model's next-token guess toward / away
 * from. Features without a curated dashboard (most of the 24576) degrade to a
 * minimal page — label + firing rate — and never trigger a chunk fetch.
 *
 * S9 seam: `onSteer` is threaded through for the (currently disabled) "Steer
 * with this feature" button.
 */
export default function FeaturePage({
  id,
  onClose,
  onSteer,
}: {
  id: number
  onClose: () => void
  /** S9 will wire feature steering here; the button is disabled until then. */
  onSteer?: (id: number) => void
}) {
  const [view, setView] = useState<FeatureView | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const reqId = useRef(0)

  // Load (or reload) whenever the feature id changes. A per-request counter
  // discards a stale resolve if the user clicks through to another feature.
  useEffect(() => {
    const seq = ++reqId.current
    setView(null)
    setErr(null)
    loadFeature(id)
      .then((v) => {
        if (seq === reqId.current) setView(v)
      })
      .catch((e) => {
        if (seq === reqId.current)
          setErr(e instanceof Error ? e.message : String(e))
      })
    return () => {
      // invalidate any in-flight resolve for the previous id
      reqId.current++
    }
  }, [id])

  // Escape-to-close + lock background scroll while the overlay is open.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
    }
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-50 overflow-y-auto overscroll-contain bg-black/70 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={`feature #${id}`}
      onClick={onClose}
    >
      {/*
        The outer element is the scroll container; this wrapper does the
        centering. `min-h-full` makes it at least as tall as the viewport, so a
        short panel is centered by the panel's `my-auto`, while a taller panel
        grows the wrapper and top-aligns — keeping the panel's top reachable and
        scrollable. Centering the panel directly inside the scroll container
        instead (the old layout) pushes an over-tall panel's top above the
        scroll range in browsers that don't apply "safe" overflow alignment
        (Firefox/Safari/older Chromium), tucking the title under the sticky
        header.
      */}
      <div className="flex min-h-full items-start justify-center p-3 sm:p-6">
        <div
          className="border-line bg-panel my-auto w-full max-w-2xl rounded-xl border shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Sticky header bar with the close control */}
          <div className="border-line bg-panel/95 sticky top-0 z-10 flex items-center justify-between gap-3 rounded-t-xl border-b px-4 py-2.5 backdrop-blur sm:px-5">
            <span className="text-muted font-mono text-[0.66rem] tracking-widest uppercase">
              SAE feature
            </span>
            <button
              onClick={onClose}
              className="text-muted hover:text-fg font-mono text-xs"
              aria-label="close feature page"
            >
              close ✕
            </button>
          </div>

          <div className="flex flex-col gap-6 p-4 sm:p-5" data-feature-page={id}>
            {!view && !err ? (
              <LoadingState id={id} />
            ) : err ? (
              <ErrorState id={id} message={err} />
            ) : view ? (
              <FeatureBody view={view} onSteer={onSteer} />
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------

function LoadingState({ id }: { id: number }) {
  return (
    <div className="flex items-center gap-3 py-8 text-muted" data-feature-loading>
      <span className="border-line border-t-accent inline-block h-5 w-5 animate-spin rounded-full border-2" />
      <span className="font-mono text-xs">loading feature #{id}…</span>
    </div>
  )
}

function ErrorState({ id, message }: { id: number; message: string }) {
  return (
    <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-4 font-mono text-xs text-red-300">
      Couldn&rsquo;t load feature #{id}: {message}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Body
// ---------------------------------------------------------------------------

function formatPct(freq: number): string {
  const p = freq * 100
  if (p <= 0) return '0%'
  if (p >= 1) return `${p.toFixed(1)}%`
  if (p >= 0.01) return `${p.toFixed(2)}%`
  return `${p.toPrecision(2)}%`
}

function ConfidencePill({ conf }: { conf: LabelConfidence | null }) {
  if (!conf) return null
  const tone =
    conf === 'high'
      ? 'border-accent-dim/60 text-accent'
      : conf === 'medium'
        ? 'border-warm/50 text-warm'
        : 'border-line text-muted'
  return (
    <span
      className={`rounded-full border px-2 py-0.5 font-mono text-[0.62rem] ${tone}`}
      title="how confident the auto-generated label is"
    >
      {conf} confidence
    </span>
  )
}

function FeatureBody({
  view,
  onSteer,
}: {
  view: FeatureView
  onSteer?: (id: number) => void
}) {
  const title = view.label ?? `feature #${view.id}`
  return (
    <>
      {/* Header */}
      <header className="flex flex-col gap-2">
        <h2 className="font-display text-fg text-lg leading-tight font-semibold sm:text-xl">
          {title}
        </h2>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 font-mono text-[0.68rem]">
          {view.label ? (
            <span className="text-accent">feature #{view.id}</span>
          ) : (
            <span className="text-muted">unlabeled</span>
          )}
          <ConfidencePill conf={view.labelConfidence} />
          <span className="text-muted">
            fires on <span className="text-fg">{formatPct(view.freq)}</span> of
            tokens
          </span>
        </div>
      </header>

      {view.kind === 'full' ? (
        <>
          <ExamplesSection view={view} />
          <HistogramSection bars={histogramToBars(view.histogram)} />
          <LogitLensSection
            promoted={view.promoted}
            suppressed={view.suppressed}
          />
        </>
      ) : (
        <p className="border-line bg-site text-muted rounded-lg border p-4 text-sm leading-relaxed">
          No detailed dashboard for this feature — only its firing rate is known.
          Most of the SAE&rsquo;s 24,576 features aren&rsquo;t individually
          documented; the ones with examples and a logit lens were selected during
          the harvest.
        </p>
      )}

      <SteerButton id={view.id} onSteer={onSteer} />
    </>
  )
}

// ---------------------------------------------------------------------------
// Max-activating examples (the centerpiece)
// ---------------------------------------------------------------------------

function SectionTitle({
  children,
  hint,
}: {
  children: ReactNode
  hint?: string
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <h3 className="font-display text-fg text-sm font-semibold">{children}</h3>
      {hint && (
        <p className="text-muted text-[0.72rem] leading-relaxed">{hint}</p>
      )}
    </div>
  )
}

function ExamplesSection({ view }: { view: Extract<FeatureView, { kind: 'full' }> }) {
  const examples = view.examples
  return (
    <section className="flex flex-col gap-3">
      <SectionTitle hint="Real text where this feature fires hardest. Each token is shaded by how strongly the feature activated on it; the outlined token is the peak.">
        Where it fires
      </SectionTitle>
      {examples.length === 0 ? (
        <p className="text-muted font-mono text-xs">no examples recorded</p>
      ) : (
        <div className="flex flex-col gap-2">
          {examples.map((ex, i) => (
            <ExampleRow key={i} ex={ex} />
          ))}
        </div>
      )}
    </section>
  )
}

function ExampleRow({ ex }: { ex: FeatureExample }) {
  const decoded = decodeActs(ex.acts, ex.maxAct)
  return (
    <div className="border-line bg-site flex flex-wrap items-center gap-x-[2px] gap-y-1 rounded-lg border p-2 leading-relaxed">
      {ex.tokens.map((t, i) => {
        const raw = ex.acts[i] ?? 0
        // 0 → transparent, peak (255) → full accent; gentle gamma so mid
        // activations stay visible. Matches the site's lens color ramp.
        const alpha = raw > 0 ? Math.pow(raw / 255, 0.7) : 0
        const strong = alpha > 0.5
        const isPeak = i === ex.actIndex
        return (
          <span
            key={i}
            title={`${toDisplay(t)} · activation ${decoded[i].toFixed(2)}`}
            className={`inline-block rounded-[3px] px-1 py-0.5 font-mono text-[0.72rem] ${
              isPeak ? 'ring-accent font-semibold ring-1' : ''
            } ${strong ? '' : 'text-fg'}`}
            style={{
              backgroundColor: alpha > 0 ? `rgba(0,229,204,${alpha})` : 'transparent',
              color: strong ? '#0d1117' : undefined,
            }}
          >
            {toDisplay(t)}
          </span>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Activation histogram (plain SVG, project idiom — no new chart lib)
// ---------------------------------------------------------------------------

function HistogramSection({ bars }: { bars: HistBar[] }) {
  if (bars.length === 0) return null
  const W = 320
  const H = 84
  const max = Math.max(1, ...bars.map((b) => b.count))
  const bw = W / bars.length
  const lo = bars[0].x0
  const hi = bars[bars.length - 1].x1
  return (
    <section className="flex flex-col gap-3">
      <SectionTitle hint="How often the feature fires at each activation strength, across the harvest sample. The activation axis is log-spaced; most firings are weak.">
        Activation distribution
      </SectionTitle>
      <div className="border-line bg-site rounded-lg border p-3">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block w-full"
          style={{ maxHeight: 110 }}
          role="img"
          aria-label="activation frequency histogram"
          data-histogram
          preserveAspectRatio="none"
        >
          {bars.map((b, i) => {
            const h = b.count > 0 ? Math.max(1.5, (b.count / max) * (H - 6)) : 0
            return (
              <rect
                key={i}
                x={i * bw + 0.5}
                y={H - h}
                width={Math.max(0.5, bw - 1)}
                height={h}
                fill="var(--color-accent)"
                opacity={0.85}
              >
                <title>
                  {b.x0.toFixed(2)}–{b.x1.toFixed(2)}: {b.count} tokens
                </title>
              </rect>
            )
          })}
        </svg>
        <div className="text-muted mt-1 flex justify-between font-mono text-[0.62rem]">
          <span>activation {lo.toFixed(2)}</span>
          <span>{hi.toFixed(1)}</span>
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Logit lens
// ---------------------------------------------------------------------------

function LogitLensSection({
  promoted,
  suppressed,
}: {
  promoted: LogitLensPair[]
  suppressed: LogitLensPair[]
}) {
  if (promoted.length === 0 && suppressed.length === 0) return null
  return (
    <section className="flex flex-col gap-3">
      <SectionTitle hint="When this feature is active it nudges the model's next-token guess — toward the tokens on the left, away from those on the right.">
        Effect on the next token
      </SectionTitle>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <LensColumn title="pushes toward" pairs={promoted} tone="promote" />
        <LensColumn title="pushes away" pairs={suppressed} tone="suppress" />
      </div>
    </section>
  )
}

function LensColumn({
  title,
  pairs,
  tone,
}: {
  title: string
  pairs: LogitLensPair[]
  tone: 'promote' | 'suppress'
}) {
  const max = Math.max(1e-6, ...pairs.map((p) => Math.abs(p.weight)))
  const rgb = tone === 'promote' ? '0,229,204' : '255,180,84'
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-muted font-mono text-[0.66rem]">{title}</div>
      {pairs.length === 0 ? (
        <span className="text-muted font-mono text-[0.66rem]">—</span>
      ) : (
        <ul className="flex flex-col gap-1">
          {pairs.map((p, i) => {
            const alpha = Math.pow(Math.abs(p.weight) / max, 0.7) * 0.8
            return (
              <li
                key={i}
                className="flex items-center justify-between gap-2 rounded px-1.5 py-1 font-mono text-[0.7rem]"
                style={{ backgroundColor: `rgba(${rgb},${alpha})` }}
              >
                <span className="text-fg truncate">{toDisplay(p.token)}</span>
                <span className="text-muted shrink-0 tabular-nums">
                  {p.weight >= 0 ? '+' : ''}
                  {p.weight.toFixed(2)}
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Steer button (S9 seam — disabled for now)
// ---------------------------------------------------------------------------

function SteerButton({
  id,
  onSteer,
}: {
  id: number
  onSteer?: (id: number) => void
}) {
  // S9: enabled once a steering handler is wired in (SaeInspector → App route
  // to the Steering Playground). Without one it stays disabled — the S7 inline
  // usage never provides `onSteer`.
  const enabled = !!onSteer
  return (
    <div className="border-line flex flex-wrap items-center gap-3 border-t pt-4">
      <button
        type="button"
        disabled={!enabled}
        onClick={() => onSteer?.(id)}
        title={
          enabled
            ? 'Open the steering playground with this feature'
            : 'Steering arrives in a later update'
        }
        className={
          enabled
            ? 'bg-accent text-site font-display hover:bg-accent-dim inline-flex items-center gap-2 rounded-lg px-4 py-1.5 text-xs font-semibold transition-colors'
            : 'border-line text-muted inline-flex cursor-not-allowed items-center gap-2 rounded-lg border px-4 py-1.5 font-mono text-xs'
        }
      >
        Steer with this feature
        {!enabled && (
          <span className="border-line text-muted rounded-full border px-1.5 py-0.5 text-[0.58rem]">
            coming soon
          </span>
        )}
      </button>
      <span className="text-muted font-mono text-[0.66rem]">
        {enabled
          ? 'add this feature to the residual stream and watch the completion bend.'
          : 'clamp this feature on or off and re-run the prompt — later.'}
      </span>
    </div>
  )
}
