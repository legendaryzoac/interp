import { useState } from 'react'
import { N_HEADS, N_LAYERS } from '../lib/runner'
import type { ResultView } from '../lib/viewModel'
import MiniHeatmap from './MiniHeatmap'
import ZoomedHeatmap from './ZoomedHeatmap'
import ArcView from './ArcView'

type Detail = 'heatmap' | 'arcs'

/**
 * The attention section: a 12×12 (layer × head) grid of mini heatmaps on the
 * left, a zoomable detail panel on the right that toggles between the full
 * heatmap (with token labels) and the token→token arc diagram.
 */
export default function AttentionView({ view }: { view: ResultView }) {
  const [sel, setSel] = useState<{ layer: number; head: number }>({
    layer: 0,
    head: 0,
  })
  const [detail, setDetail] = useState<Detail>('heatmap')

  const { patterns, seq, tokens } = view

  return (
    <div className="flex flex-col gap-6 lg:flex-row">
      {/* Small-multiples grid */}
      <div className="shrink-0">
        <div className="text-muted mb-2 font-mono text-xs">
          layer ↓ · head → (click any cell)
        </div>
        <div className="flex gap-1">
          {/* layer index gutter */}
          <div className="flex flex-col gap-1 pt-[1.15rem]">
            {Array.from({ length: N_LAYERS }, (_, l) => (
              <div
                key={l}
                className="text-muted flex h-[44px] items-center justify-end pr-1 font-mono text-[0.6rem]"
                style={{ width: 16 }}
              >
                {l}
              </div>
            ))}
          </div>
          <div>
            {/* head index header */}
            <div className="mb-1 flex gap-1">
              {Array.from({ length: N_HEADS }, (_, h) => (
                <div
                  key={h}
                  className="text-muted flex items-center justify-center font-mono text-[0.6rem]"
                  style={{ width: 44 }}
                >
                  {h}
                </div>
              ))}
            </div>
            <div className="flex flex-col gap-1">
              {Array.from({ length: N_LAYERS }, (_, l) => (
                <div key={l} className="flex gap-1">
                  {Array.from({ length: N_HEADS }, (_, h) => (
                    <MiniHeatmap
                      key={h}
                      pattern={patterns[l]}
                      head={h}
                      seq={seq}
                      selected={sel.layer === l && sel.head === h}
                      onClick={() => setSel({ layer: l, head: h })}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Detail panel */}
      <div className="border-line bg-panel min-w-0 flex-1 rounded-xl border p-4">
        <div className="mb-3 flex gap-2">
          <button
            onClick={() => setDetail('heatmap')}
            className={`font-display rounded px-3 py-1 text-xs transition-colors ${
              detail === 'heatmap'
                ? 'bg-accent text-site'
                : 'border-line text-muted hover:text-fg border'
            }`}
          >
            Heatmap
          </button>
          <button
            onClick={() => setDetail('arcs')}
            className={`font-display rounded px-3 py-1 text-xs transition-colors ${
              detail === 'arcs'
                ? 'bg-accent text-site'
                : 'border-line text-muted hover:text-fg border'
            }`}
          >
            Arcs
          </button>
        </div>

        {detail === 'heatmap' ? (
          <ZoomedHeatmap
            pattern={patterns[sel.layer]}
            layer={sel.layer}
            head={sel.head}
            seq={seq}
            tokens={tokens}
          />
        ) : (
          <ArcView
            pattern={patterns[sel.layer]}
            layer={sel.layer}
            head={sel.head}
            seq={seq}
            tokens={tokens}
          />
        )}
      </div>
    </div>
  )
}
