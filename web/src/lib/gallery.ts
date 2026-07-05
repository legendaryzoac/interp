/**
 * gallery.ts — optional precomputed-examples loader.
 *
 * If `/gallery.json` exists it lets the app show worked examples (attention +
 * logit lens) *instantly*, before/without the ~164MB+ model download. The file
 * is produced by the model pipeline and integrated later; a missing file must
 * never error — the app just falls back to the live-model path.
 *
 * Schema (frozen by the milestone brief):
 *   { prompts: [ {
 *       id, text,
 *       tokens: string[],            // display labels
 *       token_ids: number[],
 *       lens_top5: [layer][pos][{ t, p }],   // top-5 predicted token + prob
 *       patterns_u8: {               // quantized attention, per layer
 *         shape: [n_layers, n_heads, seq, seq],
 *         scale: number,             // real = u8 * scale
 *         data: string               // base64 of the u8 buffer
 *       }
 *   } ] }
 */

export interface GalleryLensCell {
  t: string
  p: number
}

export interface GalleryPatternsU8 {
  shape: [number, number, number, number] // [layers, heads, seq, seq]
  scale: number
  data: string // base64 uint8
}

export interface GalleryPrompt {
  id: string
  text: string
  tokens: string[]
  token_ids: number[]
  lens_top5: GalleryLensCell[][][] // [layer][pos][top5...]
  patterns_u8: GalleryPatternsU8
}

export interface Gallery {
  prompts: GalleryPrompt[]
}

/** Decode base64 (browser) into a Uint8Array. */
export function base64ToU8(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

/**
 * Dequantize a gallery pattern block into per-layer Float32Array patterns in
 * the same [1, heads, seq, seq] layout the live runner produces, so the views
 * can consume either source identically.
 */
export function galleryPatternsToFloat(p: GalleryPatternsU8): Float32Array[] {
  const [layers, heads, s1, s2] = p.shape
  const u8 = base64ToU8(p.data)
  const perLayer = heads * s1 * s2
  const out: Float32Array[] = []
  for (let l = 0; l < layers; l++) {
    const f = new Float32Array(perLayer)
    const base = l * perLayer
    for (let i = 0; i < perLayer; i++) f[i] = u8[base + i] * p.scale
    out.push(f)
  }
  return out
}

/** Fetch /gallery.json. Returns null on any error or absence — never throws. */
export async function loadGallery(): Promise<Gallery | null> {
  try {
    const res = await fetch('/gallery.json', { cache: 'no-cache' })
    if (!res.ok) return null
    const json = (await res.json()) as Gallery
    if (!json || !Array.isArray(json.prompts)) return null
    return json
  } catch {
    return null
  }
}
