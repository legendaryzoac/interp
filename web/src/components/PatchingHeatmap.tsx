import { useEffect, useMemo, useRef, useState } from 'react'
import { attentionColor } from '../lib/color'
import { N_LAYERS } from '../lib/runner'

/**
 * 12 x seq heatmap for the IOI activation-patching sweep. Rows = layer (block
 * the residual was patched at), columns = token position patched. Color encodes
 * how much of the clean logit-diff was recovered, normalized so the corrupted
 * baseline maps to dark and the clean baseline maps to bright teal. Values that
 * over-recover (> clean) or under-recover (< corrupt) are clamped in color but
 * reported exactly on hover.
 */
export default function PatchingHeatmap({
  heatmap,
  tokens,
  logitDiffClean,
  logitDiffCorrupt,
  cellW = 34,
  cellH = 22,
}: {
  heatmap: number[][]
  tokens: string[]
  logitDiffClean: number
  logitDiffCorrupt: number
  cellW?: number
  cellH?: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<{ l: number; p: number } | null>(null)

  const seq = heatmap[0]?.length ?? 0
  const dimW = cellW * seq
  const dimH = cellH * N_LAYERS

  // Normalize recovery onto [0,1] between corrupt (0) and clean (1).
  const span = logitDiffClean - logitDiffCorrupt
  const norm = useMemo(
    () => (v: number) => {
      if (Math.abs(span) < 1e-9) return 0
      const t = (v - logitDiffCorrupt) / span
      return Math.max(0, Math.min(1, t))
    },
    [span, logitDiffCorrupt],
  )

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, dimW, dimH)
    for (let l = 0; l < N_LAYERS; l++) {
      const row = heatmap[l] ?? []
      for (let p = 0; p < seq; p++) {
        ctx.fillStyle = attentionColor(norm(row[p] ?? logitDiffCorrupt))
        ctx.fillRect(p * cellW, l * cellH, cellW, cellH)
      }
    }
    ctx.strokeStyle = 'rgba(42,51,70,0.6)'
    ctx.lineWidth = 1
    for (let k = 0; k <= seq; k++) {
      ctx.beginPath()
      ctx.moveTo(k * cellW, 0)
      ctx.lineTo(k * cellW, dimH)
      ctx.stroke()
    }
    for (let k = 0; k <= N_LAYERS; k++) {
      ctx.beginPath()
      ctx.moveTo(0, k * cellH)
      ctx.lineTo(dimW, k * cellH)
      ctx.stroke()
    }
  }, [heatmap, seq, norm, dimW, dimH, cellW, cellH, logitDiffCorrupt])

  const hoverVal =
    hover != null ? heatmap[hover.l]?.[hover.p] : null

  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-x-auto pb-2">
        <div className="flex gap-1">
          {/* layer gutter */}
          <div className="flex flex-col pt-[1.05rem]">
            {Array.from({ length: N_LAYERS }, (_, l) => (
              <div
                key={l}
                className="text-muted flex items-center justify-end pr-1 font-mono text-[0.6rem]"
                style={{ height: cellH, width: 22 }}
              >
                L{l}
              </div>
            ))}
          </div>
          <div>
            {/* token position header */}
            <div className="mb-1 flex" style={{ height: '1rem' }}>
              {tokens.slice(0, seq).map((t, p) => (
                <div
                  key={p}
                  className={`truncate text-center font-mono text-[0.56rem] ${
                    hover?.p === p ? 'text-accent' : 'text-muted'
                  }`}
                  style={{ width: cellW }}
                  title={t}
                >
                  {t}
                </div>
              ))}
            </div>
            <canvas
              ref={canvasRef}
              width={dimW}
              height={dimH}
              className="border-line rounded border"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect()
                const p = Math.floor((e.clientX - rect.left) / cellW)
                const l = Math.floor((e.clientY - rect.top) / cellH)
                if (l >= 0 && l < N_LAYERS && p >= 0 && p < seq)
                  setHover({ l, p })
              }}
              onMouseLeave={() => setHover(null)}
            />
          </div>
        </div>
      </div>
      <div className="border-line bg-site text-muted min-h-7 rounded border px-3 py-1 font-mono text-[0.7rem] leading-5">
        {hover && hoverVal != null ? (
          <span>
            patch L<span className="text-accent">{hover.l}</span> · pos{' '}
            <span className="text-accent">{hover.p}</span> (
            <span className="text-fg">{tokens[hover.p]}</span>) → logit-diff{' '}
            <span className="text-accent">{hoverVal.toFixed(3)}</span> ·{' '}
            {(norm(hoverVal) * 100).toFixed(0)}% recovered
          </span>
        ) : (
          <span>
            dark = corrupt baseline ({logitDiffCorrupt.toFixed(2)}) · bright =
            clean recovered ({logitDiffClean.toFixed(2)}) · hover a cell
          </span>
        )}
      </div>
    </div>
  )
}
