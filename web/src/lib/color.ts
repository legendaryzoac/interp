/**
 * color.ts — perceptual color scales for attention weights and lens
 * probabilities, tuned for the dark theme (accent teal / warm amber).
 */
import { interpolateRgb } from 'd3-interpolate'

// Attention: transparent-ish dark -> teal accent.
const attnLow = '#0d1117'
const attnHigh = '#00e5cc'
const attnInterp = interpolateRgb(attnLow, attnHigh)

/** weight in [0,1] -> css color. Non-linear ramp emphasizes strong weights. */
export function attentionColor(w: number): string {
  const t = Math.max(0, Math.min(1, w))
  // gamma < 1 brightens mid-weights so faint attention is still visible
  return attnInterp(Math.pow(t, 0.6))
}

// Logit-lens probability: dark panel -> warm amber for high confidence.
const lensLow = '#161b22'
const lensHigh = '#ffb454'
const lensInterp = interpolateRgb(lensLow, lensHigh)

export function lensColor(p: number): string {
  const t = Math.max(0, Math.min(1, p))
  return lensInterp(Math.pow(t, 0.7))
}

/** Pick readable text color against a shaded lens cell. */
export function lensTextColor(p: number): string {
  return p > 0.45 ? '#0d1117' : '#f0f4f8'
}
