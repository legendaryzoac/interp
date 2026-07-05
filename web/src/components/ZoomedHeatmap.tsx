import { useEffect, useMemo, useRef, useState } from 'react'
import { attentionColor } from '../lib/color'
import { headMatrix } from '../lib/runner'

/**
 * Large heatmap for the selected (layer, head) with token labels on both axes.
 * Rows = query token (attending from), cols = key token (attended to).
 * Hovering a cell surfaces the exact weight and the token pair.
 */
export default function ZoomedHeatmap({
  pattern,
  layer,
  head,
  seq,
  tokens,
}: {
  pattern: Float32Array
  layer: number
  head: number
  seq: number
  tokens: string[]
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null)

  const matrix = useMemo(
    () => headMatrix(pattern, head, seq),
    [pattern, head, seq],
  )

  // Cell size scales down as seq grows, clamped for readability.
  const cell = Math.max(10, Math.min(30, Math.floor(460 / seq)))
  const dim = cell * seq

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, dim, dim)
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < seq; j++) {
        const w = matrix[i * seq + j]
        ctx.fillStyle = attentionColor(w)
        ctx.fillRect(j * cell, i * cell, cell, cell)
      }
    }
    // Grid lines
    ctx.strokeStyle = 'rgba(42,51,70,0.5)'
    ctx.lineWidth = 1
    for (let k = 0; k <= seq; k++) {
      ctx.beginPath()
      ctx.moveTo(k * cell, 0)
      ctx.lineTo(k * cell, dim)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(0, k * cell)
      ctx.lineTo(dim, k * cell)
      ctx.stroke()
    }
  }, [matrix, seq, cell, dim])

  const hoverWeight =
    hover != null ? matrix[hover.i * seq + hover.j] : null

  return (
    <div className="flex flex-col gap-3">
      <div className="text-muted font-mono text-xs">
        Layer <span className="text-accent">{layer}</span> · Head{' '}
        <span className="text-accent">{head}</span> — rows attend to columns
      </div>

      <div className="flex gap-2 overflow-x-auto pb-2">
        {/* Row labels (query tokens) */}
        <div className="flex flex-col justify-start pt-0" style={{ marginTop: 0 }}>
          {tokens.slice(0, seq).map((t, i) => (
            <div
              key={i}
              className={`flex items-center justify-end pr-1 font-mono text-[0.65rem] ${
                hover?.i === i ? 'text-accent' : 'text-muted'
              }`}
              style={{ height: cell, maxWidth: 90 }}
            >
              <span className="truncate">{t}</span>
            </div>
          ))}
        </div>

        <div className="flex flex-col gap-1">
          <canvas
            ref={canvasRef}
            width={dim}
            height={dim}
            className="border-line rounded border"
            style={{ imageRendering: 'pixelated' }}
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect()
              const j = Math.floor((e.clientX - rect.left) / cell)
              const i = Math.floor((e.clientY - rect.top) / cell)
              if (i >= 0 && i < seq && j >= 0 && j < seq) setHover({ i, j })
            }}
            onMouseLeave={() => setHover(null)}
          />
          {/* Column labels (key tokens), rotated */}
          <div className="relative" style={{ height: 70, width: dim }}>
            {tokens.slice(0, seq).map((t, j) => (
              <div
                key={j}
                className={`absolute origin-top-left font-mono text-[0.65rem] whitespace-nowrap ${
                  hover?.j === j ? 'text-accent' : 'text-muted'
                }`}
                style={{
                  left: j * cell + cell / 2,
                  top: 2,
                  transform: 'rotate(55deg)',
                }}
              >
                {t}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="border-line bg-site text-muted h-8 rounded border px-3 font-mono text-xs leading-8">
        {hover && hoverWeight != null ? (
          <span>
            <span className="text-fg">{tokens[hover.i]}</span> →{' '}
            <span className="text-fg">{tokens[hover.j]}</span> ={' '}
            <span className="text-accent">{hoverWeight.toFixed(4)}</span>
          </span>
        ) : (
          <span>hover a cell for the exact weight</span>
        )}
      </div>
    </div>
  )
}
