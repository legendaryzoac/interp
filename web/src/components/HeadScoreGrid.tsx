import { useEffect, useRef, useState } from 'react'
import { attentionColor } from '../lib/color'
import { N_HEADS, N_LAYERS } from '../lib/runner'

/**
 * A 12x12 (layer x head) heatmap of a single scalar score per head — reused by
 * the Compare tab (suffix-attention mass) and the Circuits tab (induction
 * scores). Values are a flat [N_LAYERS*N_HEADS] array, row-major (layer outer).
 * Colors reuse the shared attention ramp; the score is normalized to its own
 * max so the strongest heads stand out. Hovering a cell reports (layer, head,
 * value); cells above `highlightThreshold * max` get a ring.
 */
export default function HeadScoreGrid({
  scores,
  cellSize = 34,
  highlightThreshold = 0.6,
  valueLabel = 'score',
}: {
  scores: Float32Array | number[]
  cellSize?: number
  highlightThreshold?: number
  valueLabel?: string
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<{ l: number; h: number } | null>(null)

  let max = 0
  for (let i = 0; i < scores.length; i++) if (scores[i] > max) max = scores[i]
  const norm = max > 0 ? max : 1
  const dim = cellSize * N_HEADS

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, dim, cellSize * N_LAYERS)
    for (let l = 0; l < N_LAYERS; l++) {
      for (let h = 0; h < N_HEADS; h++) {
        const v = scores[l * N_HEADS + h] / norm
        ctx.fillStyle = attentionColor(v)
        ctx.fillRect(h * cellSize, l * cellSize, cellSize, cellSize)
        if (v >= highlightThreshold) {
          ctx.strokeStyle = 'var(--color-warm)'
          ctx.strokeStyle = '#ffb454'
          ctx.lineWidth = 2
          ctx.strokeRect(h * cellSize + 1, l * cellSize + 1, cellSize - 2, cellSize - 2)
        }
      }
    }
    ctx.strokeStyle = 'rgba(42,51,70,0.6)'
    ctx.lineWidth = 1
    for (let k = 0; k <= N_HEADS; k++) {
      ctx.beginPath()
      ctx.moveTo(k * cellSize, 0)
      ctx.lineTo(k * cellSize, cellSize * N_LAYERS)
      ctx.stroke()
    }
    for (let k = 0; k <= N_LAYERS; k++) {
      ctx.beginPath()
      ctx.moveTo(0, k * cellSize)
      ctx.lineTo(dim, k * cellSize)
      ctx.stroke()
    }
  }, [scores, norm, dim, cellSize, highlightThreshold])

  const hoverVal =
    hover != null ? scores[hover.l * N_HEADS + hover.h] : null

  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-x-auto pb-2">
        <div className="flex gap-1">
          {/* layer gutter */}
          <div className="flex flex-col pt-[1.1rem]">
            {Array.from({ length: N_LAYERS }, (_, l) => (
              <div
                key={l}
                className="text-muted flex items-center justify-end pr-1 font-mono text-[0.6rem]"
                style={{ height: cellSize, width: 16 }}
              >
                {l}
              </div>
            ))}
          </div>
          <div>
            {/* head header */}
            <div className="mb-1 flex">
              {Array.from({ length: N_HEADS }, (_, h) => (
                <div
                  key={h}
                  className="text-muted flex items-center justify-center font-mono text-[0.6rem]"
                  style={{ width: cellSize }}
                >
                  {h}
                </div>
              ))}
            </div>
            <canvas
              ref={ref}
              width={dim}
              height={cellSize * N_LAYERS}
              className="border-line rounded border"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect()
                const h = Math.floor((e.clientX - rect.left) / cellSize)
                const l = Math.floor((e.clientY - rect.top) / cellSize)
                if (l >= 0 && l < N_LAYERS && h >= 0 && h < N_HEADS)
                  setHover({ l, h })
              }}
              onMouseLeave={() => setHover(null)}
            />
          </div>
        </div>
      </div>
      <div className="border-line bg-site text-muted h-7 rounded border px-3 font-mono text-[0.7rem] leading-7">
        {hover && hoverVal != null ? (
          <span>
            L<span className="text-accent">{hover.l}</span> · H
            <span className="text-accent">{hover.h}</span> — {valueLabel}{' '}
            <span className="text-accent">{hoverVal.toFixed(4)}</span>
          </span>
        ) : (
          <span>hover a cell · brighter = higher · ring = top heads</span>
        )}
      </div>
    </div>
  )
}
