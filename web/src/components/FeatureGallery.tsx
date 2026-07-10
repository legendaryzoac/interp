import { useEffect, useRef, useState } from 'react'
import {
  loadFeature,
  type FeatureExample,
  type FeatureView,
  type LabelConfidence,
} from '../lib/featurePage'
import { FEATURED_FEATURES, type FeaturedFeature } from '../data/featuredFeatures'

/**
 * FeatureGallery (S10) — the SAE tab's landing payoff. A grid of hand-curated
 * features (see data/featuredFeatures.ts) that a first-time visitor can browse
 * before running anything. Each card shows the feature's real label, its firing
 * rate, and a one-line taste of its strongest example (peak token emphasized),
 * and offers the two existing flows: open its full Feature Page (S8) or steer a
 * prompt with it (S9).
 *
 * Cards load lazily and independently via `loadFeature` (the memoized loader
 * shared with the Feature Page), so the whole gallery is a single index fetch
 * plus the two dashboard chunks these features live in — no refetch per card,
 * and no dependence on the 37 MB SAE encoder. A per-card skeleton shows while
 * its chunk streams; a card never blocks the tab.
 */
export default function FeatureGallery({
  onOpen,
  onSteer,
  hint,
}: {
  /** Open the S8 Feature Page overlay for this feature. */
  onOpen: (id: number) => void
  /** Route to the S9 Steering Playground with this feature preselected. */
  onSteer?: (id: number) => void
  /** Small contextual line under the intro (e.g. "load the encoder above…"). */
  hint?: string
}) {
  return (
    <div className="flex flex-col gap-3" data-feature-gallery>
      <div className="flex flex-col gap-1">
        <h3 className="font-display text-fg text-sm font-semibold">
          A few features to start with
        </h3>
        <p className="text-muted text-[0.8rem] leading-relaxed">
          Some of the more legible things this SAE learned — a mix of topics and
          plain grammar. Open one to see where it fires across real text, or steer
          a prompt with it.
        </p>
        {hint && (
          <p className="text-muted font-mono text-[0.66rem]">{hint}</p>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {FEATURED_FEATURES.map((f) => (
          <FeatureCard
            key={f.id}
            feature={f}
            onOpen={onOpen}
            onSteer={onSteer}
          />
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// One card
// ---------------------------------------------------------------------------

function FeatureCard({
  feature,
  onOpen,
  onSteer,
}: {
  feature: FeaturedFeature
  onOpen: (id: number) => void
  onSteer?: (id: number) => void
}) {
  const { id, blurb } = feature
  const [view, setView] = useState<FeatureView | null>(null)
  const [failed, setFailed] = useState(false)
  const reqId = useRef(0)

  useEffect(() => {
    const seq = ++reqId.current
    setView(null)
    setFailed(false)
    loadFeature(id)
      .then((v) => {
        if (seq === reqId.current) setView(v)
      })
      .catch(() => {
        if (seq === reqId.current) setFailed(true)
      })
    return () => {
      reqId.current++
    }
  }, [id])

  const title = blurb ?? view?.label ?? `feature #${id}`
  const example =
    view && view.kind === 'full' ? view.examples[0] : undefined

  return (
    <div className="border-line bg-panel/40 hover:border-accent-dim/60 flex flex-col overflow-hidden rounded-xl border transition-colors">
      {/* Body — clicking opens the full feature page */}
      <button
        type="button"
        onClick={() => onOpen(id)}
        title={`Open feature #${id}`}
        data-feature-card={id}
        className="group flex flex-1 flex-col gap-2 p-3 text-left"
      >
        <div className="flex items-start justify-between gap-2">
          <span className="font-display text-fg group-hover:text-accent text-[0.82rem] leading-snug font-semibold transition-colors">
            {title}
          </span>
          <span className="border-line text-muted shrink-0 rounded-full border px-1.5 py-0.5 font-mono text-[0.58rem]">
            #{id}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-[0.6rem]">
          {view ? (
            <>
              <ConfidenceDot conf={view.labelConfidence} />
              <span className="text-muted">
                fires on{' '}
                <span className="text-fg">{formatPct(view.freq)}</span> of tokens
              </span>
            </>
          ) : failed ? (
            <span className="text-muted">metadata unavailable</span>
          ) : (
            <span className="text-muted/60">loading…</span>
          )}
        </div>

        {/* Example taste */}
        <div className="mt-0.5 min-h-[2.2rem]">
          {example ? (
            <ExampleSnippet ex={example} />
          ) : view ? (
            <span className="text-muted/60 font-mono text-[0.66rem]">
              no example recorded
            </span>
          ) : failed ? null : (
            <SnippetSkeleton />
          )}
        </div>
      </button>

      {/* Footer actions */}
      <div className="border-line flex items-center gap-2 border-t px-3 py-2">
        <button
          type="button"
          onClick={() => onOpen(id)}
          className="text-muted hover:text-accent font-mono text-[0.66rem] transition-colors"
        >
          feature page →
        </button>
        <span className="text-line">·</span>
        <button
          type="button"
          disabled={!onSteer}
          onClick={() => onSteer?.(id)}
          title={
            onSteer
              ? 'Steer a prompt with this feature'
              : 'Steering unavailable'
          }
          className={
            onSteer
              ? 'text-accent hover:text-fg font-mono text-[0.66rem] transition-colors'
              : 'text-muted/50 cursor-not-allowed font-mono text-[0.66rem]'
          }
        >
          steer ↗
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Example snippet — a one-line taste with the peak token emphasized
// ---------------------------------------------------------------------------

const CTX_BEFORE = 7
const CTX_AFTER = 4

function ExampleSnippet({ ex }: { ex: FeatureExample }) {
  const peak = ex.actIndex
  const start = Math.max(0, peak - CTX_BEFORE)
  const end = Math.min(ex.tokens.length, peak + CTX_AFTER + 1)
  const before = ex.tokens.slice(start, peak).map(clean).join('')
  const peakTok = clean(ex.tokens[peak] ?? '')
  const after = ex.tokens.slice(peak + 1, end).map(clean).join('')
  return (
    <p className="text-muted font-mono text-[0.66rem] leading-relaxed break-words">
      {start > 0 && <span className="text-muted/50">…</span>}
      {trimLead(before)}
      <span className="text-accent font-semibold">{peakTok}</span>
      {after}
      {end < ex.tokens.length && <span className="text-muted/50">…</span>}
    </p>
  )
}

/** Collapse the BPE token's whitespace to something readable in a one-liner. */
function clean(tok: string): string {
  return tok
    .replace(/<\|endoftext\|>/g, ' ')
    .replace(/[\n\r\t]+/g, ' ')
}

/**
 * Drop leading whitespace (and any stray replacement char from the raw harvest
 * token) so the snippet starts flush on a real word.
 */
function trimLead(s: string): string {
  return s.replace(/^[\s�]+/, '')
}

function SnippetSkeleton() {
  return (
    <span className="flex flex-col gap-1" aria-hidden data-snippet-skeleton>
      <span className="bg-line/60 block h-2 w-[85%] animate-pulse rounded" />
      <span className="bg-line/60 block h-2 w-[60%] animate-pulse rounded" />
    </span>
  )
}

// ---------------------------------------------------------------------------
// Small bits
// ---------------------------------------------------------------------------

function ConfidenceDot({ conf }: { conf: LabelConfidence | null }) {
  if (!conf) return null
  const tone =
    conf === 'high'
      ? 'bg-accent'
      : conf === 'medium'
        ? 'bg-warm'
        : 'bg-line'
  return (
    <span
      className="text-muted inline-flex items-center gap-1"
      title={`auto-label confidence: ${conf}`}
    >
      <span className={`inline-block h-1.5 w-1.5 rounded-full ${tone}`} />
      {conf}
    </span>
  )
}

function formatPct(freq: number): string {
  const p = freq * 100
  if (p <= 0) return '0%'
  if (p >= 1) return `${p.toFixed(1)}%`
  if (p >= 0.01) return `${p.toFixed(2)}%`
  return `${p.toPrecision(2)}%`
}
