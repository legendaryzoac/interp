import { useState, type ReactNode } from 'react'

/**
 * A plain-language "what am I looking at?" panel that sits at the top of each
 * view, above the dense technical annotations. Deliberately uses the body font
 * (not mono) and normal prose so a non-ML visitor gets a readable, jargon-light
 * explanation, while the precise practitioner-facing copy stays underneath.
 *
 * Collapsible, with the open/closed choice remembered per-view in localStorage
 * so a returning visitor isn't re-shown a guide they've already dismissed.
 */
export interface ExplainerPoint {
  /** Short label for the thing being explained (e.g. "Rows", "Color"). */
  label: string
  /** Plain-language description of it. */
  text: ReactNode
}

const STORAGE_PREFIX = 'interp.explainer.'

function initialOpen(id: string): boolean {
  try {
    return localStorage.getItem(STORAGE_PREFIX + id) !== 'closed'
  } catch {
    return true
  }
}

function persist(id: string, open: boolean) {
  try {
    localStorage.setItem(STORAGE_PREFIX + id, open ? 'open' : 'closed')
  } catch {
    /* private mode / disabled storage — non-fatal, just don't remember */
  }
}

export default function Explainer({
  id,
  lead,
  points,
}: {
  /** Stable key used to remember the collapsed state across visits. */
  id: string
  /** The main plain-language paragraph(s). */
  lead: ReactNode
  /** Optional "how to read it" list of labelled points. */
  points?: ExplainerPoint[]
}) {
  const [open, setOpen] = useState(() => initialOpen(id))

  const toggle = () => {
    const next = !open
    setOpen(next)
    persist(id, next)
  }

  const bodyId = `explainer-body-${id}`

  return (
    <div className="border-line bg-panel/30 relative overflow-hidden rounded-xl border pl-4">
      <span
        aria-hidden="true"
        className="bg-accent/70 absolute top-0 left-0 h-full w-1"
      />
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        aria-controls={bodyId}
        className="group flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <span className="text-accent font-mono text-[0.7rem] font-medium tracking-widest uppercase">
          What am I looking at?
        </span>
        <span className="text-muted group-hover:text-fg flex items-center gap-1.5 font-mono text-[0.68rem] whitespace-nowrap transition-colors">
          {open ? 'hide' : 'show'}
          <span className={open ? 'transition-transform' : 'rotate-180 transition-transform'}>
            ⌃
          </span>
        </span>
      </button>

      {open && (
        <div id={bodyId} className="px-4 pt-0 pb-4">
          <p className="text-fg/90 max-w-3xl text-sm leading-relaxed">{lead}</p>
          {points && points.length > 0 && (
            <ul className="mt-3 flex flex-col gap-1.5">
              {points.map((p, i) => (
                <li
                  key={i}
                  className="text-muted max-w-3xl text-[0.82rem] leading-relaxed"
                >
                  <span className="text-accent font-medium">{p.label}</span>
                  <span className="text-muted"> — </span>
                  <span className="text-fg/80">{p.text}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
