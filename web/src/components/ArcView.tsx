import { useMemo } from 'react'
import { headMatrix } from '../lib/runner'

/**
 * Token→token attention arcs for the selected head. Tokens are laid out on a
 * horizontal line; an arc from query i to key j has opacity/width scaled by the
 * attention weight. Self-attention (i===j) is drawn as a short stub. Only arcs
 * above a small threshold are drawn to keep the SVG light.
 */
export default function ArcView({
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
  const matrix = useMemo(
    () => headMatrix(pattern, head, seq),
    [pattern, head, seq],
  )

  const pad = 24
  const gap = Math.max(26, Math.min(60, Math.floor(760 / Math.max(1, seq))))
  const width = pad * 2 + gap * (seq - 1)
  const maxArc = 120
  const height = maxArc + 60

  const arcs = useMemo(() => {
    const out: { i: number; j: number; w: number }[] = []
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j <= i; j++) {
        const w = matrix[i * seq + j]
        if (w > 0.04) out.push({ i, j, w })
      }
    }
    // draw stronger arcs last so they sit on top
    out.sort((a, b) => a.w - b.w)
    return out
  }, [matrix, seq])

  const x = (k: number) => pad + gap * k
  const baseY = maxArc + 10

  return (
    <div className="flex flex-col gap-2">
      <div className="text-muted font-mono text-xs">
        Layer <span className="text-accent">{layer}</span> · Head{' '}
        <span className="text-accent">{head}</span> — arc = token attends to
        earlier token
      </div>
      <div className="overflow-x-auto">
        <svg
          width={width}
          height={height}
          className="block"
          role="img"
          aria-label={`attention arcs for layer ${layer} head ${head}`}
        >
          {arcs.map(({ i, j, w }, idx) => {
            if (i === j) {
              // self-attention stub
              return (
                <line
                  key={idx}
                  x1={x(i)}
                  y1={baseY}
                  x2={x(i)}
                  y2={baseY - 10 - w * 12}
                  stroke="var(--color-warm)"
                  strokeWidth={0.5 + w * 3}
                  strokeOpacity={0.25 + w * 0.7}
                  strokeLinecap="round"
                />
              )
            }
            const xi = x(i)
            const xj = x(j)
            const mid = (xi + xj) / 2
            const span = Math.abs(xi - xj)
            const h = Math.min(maxArc, 14 + span * 0.5)
            return (
              <path
                key={idx}
                d={`M ${xj} ${baseY} Q ${mid} ${baseY - h} ${xi} ${baseY}`}
                fill="none"
                stroke="var(--color-accent)"
                strokeWidth={0.4 + w * 4}
                strokeOpacity={0.12 + w * 0.8}
              />
            )
          })}
          {/* token dots + labels */}
          {tokens.slice(0, seq).map((t, k) => (
            <g key={k}>
              <circle cx={x(k)} cy={baseY} r={2.5} fill="var(--color-muted)" />
              <text
                x={x(k)}
                y={baseY + 20}
                textAnchor="end"
                transform={`rotate(-45 ${x(k)} ${baseY + 20})`}
                className="font-mono"
                fontSize="10"
                fill="var(--color-muted)"
              >
                {t}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  )
}
