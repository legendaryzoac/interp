import { useEffect, useRef } from 'react'
import { attentionColor } from '../lib/color'
import { headMatrix } from '../lib/runner'

/**
 * A tiny canvas heatmap for one (layer, head) attention matrix, used in the
 * 12×12 small-multiples grid. Renders on a fixed pixel grid regardless of seq
 * so all cells align. Uses a typed-array view (no copy) into the pattern.
 */
export default function MiniHeatmap({
  pattern,
  head,
  seq,
  size = 44,
  selected,
  onClick,
}: {
  pattern: Float32Array
  head: number
  seq: number
  size?: number
  selected: boolean
  onClick: () => void
}) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const m = headMatrix(pattern, head, seq)
    // Draw seq×seq into size×size using nearest-neighbor blocks.
    const cell = size / seq
    ctx.clearRect(0, 0, size, size)
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < seq; j++) {
        const w = m[i * seq + j]
        if (w <= 0.002) continue
        ctx.fillStyle = attentionColor(w)
        ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell))
      }
    }
  }, [pattern, head, seq, size])

  return (
    <button
      onClick={onClick}
      className={`group relative rounded-sm outline-none transition-shadow ${
        selected ? 'ring-accent ring-2' : 'ring-line hover:ring-accent-dim ring-1'
      }`}
      style={{ width: size, height: size }}
      title={`head ${head}`}
    >
      <canvas
        ref={ref}
        width={size}
        height={size}
        className="block h-full w-full rounded-sm"
        style={{ imageRendering: 'pixelated' }}
      />
    </button>
  )
}
