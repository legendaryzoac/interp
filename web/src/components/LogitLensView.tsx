import { useState } from 'react'
import { lensColor, lensTextColor } from '../lib/color'
import type { ResultView } from '../lib/viewModel'
import Explainer from './Explainer'

/**
 * Logit-lens grid: rows = layers (0 at top → final at bottom), columns = token
 * positions. Each cell shows the top-1 predicted *next* token at that layer/
 * position, its background shaded by probability. Hovering reveals the top-5.
 * The final row is highlighted as the model's actual prediction.
 */
export default function LogitLensView({ view }: { view: ResultView }) {
  const { lens, tokens, seq } = view
  const nLayers = lens.length
  const [hover, setHover] = useState<{ layer: number; pos: number } | null>(null)

  const cellW = Math.max(52, Math.min(96, Math.floor(900 / Math.max(1, seq))))

  const hovered =
    hover != null ? lens[hover.layer]?.[hover.pos] : null

  return (
    <div className="flex flex-col gap-3">
      <Explainer
        id="lens"
        lead={
          <>
            GPT-2 doesn&rsquo;t pick the next word in one step; it refines its
            guess across 12 layers. The{' '}
            <span className="text-accent">logit lens</span> reads out that guess
            after each layer: given only what the model has worked out so far,
            what word would it pick? Reading top to bottom, the early layers are
            usually vague and the guess tends to settle by the last one.
          </>
        }
        points={[
          {
            label: 'Rows',
            text: "the model's layers, earliest at the top, the final answer at the bottom (outlined).",
          },
          {
            label: 'Columns',
            text: "positions in your text. Each cell is the model's best next-word guess at that layer and position.",
          },
          {
            label: 'Color',
            text: 'how confident that guess is: deeper shading means higher probability. Hover any cell for its top-5 candidates.',
          },
        ]}
      />

      <div className="text-muted font-mono text-xs">
        each cell = top-1 predicted next-token · shaded by probability · hover
        for top-5 · <span className="text-warm">bottom row</span> = model's
        actual prediction
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="inline-flex flex-col gap-[3px]">
          {/* column header: input tokens */}
          <div className="flex gap-[3px] pl-10">
            {tokens.slice(0, seq).map((t, p) => (
              <div
                key={p}
                className="text-muted truncate text-center font-mono text-[0.62rem]"
                style={{ width: cellW }}
                title={t}
              >
                {t}
              </div>
            ))}
          </div>

          {Array.from({ length: nLayers }, (_, l) => {
            const isFinal = l === nLayers - 1
            return (
              <div key={l} className="flex items-center gap-[3px]">
                <div
                  className={`w-9 pr-1 text-right font-mono text-[0.62rem] ${
                    isFinal ? 'text-warm' : 'text-muted'
                  }`}
                >
                  L{l}
                </div>
                {Array.from({ length: seq }, (_, p) => {
                  const cell = lens[l]?.[p]
                  const top = cell?.top5?.[0]
                  const prob = top?.prob ?? 0
                  const isHover = hover?.layer === l && hover?.pos === p
                  return (
                    <div
                      key={p}
                      onMouseEnter={() => setHover({ layer: l, pos: p })}
                      onMouseLeave={() => setHover(null)}
                      className={`flex h-6 cursor-default items-center justify-center overflow-hidden rounded-[3px] font-mono text-[0.64rem] transition-shadow ${
                        isHover ? 'ring-accent ring-2' : ''
                      } ${isFinal ? 'outline outline-1 outline-[var(--color-warm)]' : ''}`}
                      style={{
                        width: cellW,
                        background: lensColor(prob),
                        color: lensTextColor(prob),
                      }}
                      title={top ? `${top.token} · ${(prob * 100).toFixed(1)}%` : ''}
                    >
                      <span className="truncate px-1">{top?.token ?? '·'}</span>
                    </div>
                  )
                })}
              </div>
            )
          })}
        </div>
      </div>

      {/* Top-5 popover for the hovered cell */}
      <div className="border-line bg-site min-h-[5.5rem] rounded-lg border p-3">
        {hovered ? (
          <div>
            <div className="text-muted mb-1.5 font-mono text-[0.68rem]">
              layer {hover!.layer} · position {hover!.pos} (
              <span className="text-fg">{tokens[hover!.pos]}</span>) — top 5
            </div>
            <div className="flex flex-wrap gap-1.5">
              {hovered.top5.map((e, i) => (
                <div
                  key={i}
                  className="border-line bg-panel flex items-center gap-2 rounded border px-2 py-1"
                >
                  <span className="text-fg font-mono text-xs">{e.token}</span>
                  <span className="text-accent font-mono text-[0.66rem]">
                    {(e.prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <span className="text-muted font-mono text-xs">
            hover a cell to see its top-5 predictions
          </span>
        )}
      </div>
    </div>
  )
}
