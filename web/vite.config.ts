import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { createReadStream, existsSync, statSync } from 'node:fs'
import path from 'node:path'

/**
 * Dev-only static server for the segmented ONNX artifacts.
 *
 * The model weights live outside the repo at D:/dev/interp-artifacts/onnx/
 * (652MB fp32 / 326MB fp16 / 164MB int8) to keep the near-full C: drive clean.
 * This middleware serves GET /models/<variant>/<file>.onnx straight from that
 * directory so the app can fetch them at `${VITE_MODEL_BASE_URL}/...` (default
 * `/models`) without copying anything into public/.
 *
 * Only whitelisted variant dirs and .onnx/.json files are served, and the
 * resolved path is confined to ARTIFACTS_ROOT to avoid path traversal.
 */
const ARTIFACTS_ROOT =
  process.env.INTERP_ARTIFACTS_ROOT ?? 'D:/dev/interp-artifacts/onnx'

const ALLOWED_VARIANTS = new Set(['fp32', 'fp16', 'int8'])

/**
 * Dev-only static server for the layer-8 SAE artifacts (encoder graph +
 * dashboards). These live outside the repo at D:/dev/sae-artifacts/L8/ for the
 * same reason the model graphs do. Serves GET /sae/<...path> straight from that
 * directory so the SAE tab can fetch `${VITE_SAE_BASE_URL}/...` (default `/sae`)
 * — the encoder (sae_enc_fp16.onnx), dashboards/manifest.json (version pointer)
 * and dashboards/index.json (feature labels).
 *
 * In production Zack wires VITE_SAE_BASE_URL in deploy.yml to the HF dataset
 * resolve URL, e.g.
 *   https://huggingface.co/datasets/<user>/interp-sae-gpt2-L8/resolve/main/L8
 * and this middleware is not used.
 *
 * Only .onnx/.json/.bin files are served, and the resolved path is confined to
 * SAE_ROOT to avoid path traversal.
 */
const SAE_ROOT = process.env.INTERP_SAE_ROOT ?? 'D:/dev/sae-artifacts/L8'

function serveModels(): Plugin {
  return {
    name: 'serve-onnx-models',
    configureServer(server) {
      server.middlewares.use('/models', (req, res, next) => {
        try {
          const url = new URL(req.url ?? '', 'http://localhost')
          // req.url is already stripped of the /models prefix by the mount.
          const rel = decodeURIComponent(url.pathname).replace(/^\/+/, '')
          const parts = rel.split('/')
          if (parts.length !== 2) return next()
          const [variant, file] = parts
          if (!ALLOWED_VARIANTS.has(variant)) return next()
          if (!/^[\w.-]+\.(onnx|json)$/.test(file)) return next()

          const root = path.resolve(ARTIFACTS_ROOT)
          const abs = path.resolve(root, variant, file)
          if (!abs.startsWith(root)) return next()
          if (!existsSync(abs)) {
            res.statusCode = 404
            res.end(`not found: ${variant}/${file}`)
            return
          }

          const { size } = statSync(abs)
          res.setHeader(
            'Content-Type',
            file.endsWith('.json') ? 'application/json' : 'application/octet-stream',
          )
          res.setHeader('Content-Length', String(size))
          // Allow the browser Cache API / HTTP cache to hold onto these.
          res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
          createReadStream(abs).pipe(res)
        } catch (err) {
          res.statusCode = 500
          res.end(String(err))
        }
      })
    },
  }
}

function serveSae(): Plugin {
  return {
    name: 'serve-sae-artifacts',
    configureServer(server) {
      server.middlewares.use('/sae', (req, res, next) => {
        try {
          const url = new URL(req.url ?? '', 'http://localhost')
          // req.url is already stripped of the /sae prefix by the mount, so
          // e.g. /sae/dashboards/index.json arrives as /dashboards/index.json.
          const rel = decodeURIComponent(url.pathname).replace(/^\/+/, '')
          if (!rel || rel.includes('..')) return next()
          if (!/\.(onnx|json|bin)$/.test(rel)) return next()

          const root = path.resolve(SAE_ROOT)
          const abs = path.resolve(root, rel)
          if (abs !== root && !abs.startsWith(root + path.sep)) return next()
          if (!existsSync(abs)) {
            res.statusCode = 404
            res.end(`not found: ${rel}`)
            return
          }

          const { size } = statSync(abs)
          res.setHeader(
            'Content-Type',
            abs.endsWith('.json') ? 'application/json' : 'application/octet-stream',
          )
          res.setHeader('Content-Length', String(size))
          res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
          createReadStream(abs).pipe(res)
        } catch (err) {
          res.statusCode = 500
          res.end(String(err))
        }
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), serveModels(), serveSae()],
  optimizeDeps: {
    // onnxruntime-web ships wasm + workers; let Vite pre-bundle the ESM entry.
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: {
      // Required for onnxruntime-web multi-threaded wasm (SharedArrayBuffer).
      // `credentialless` lets us keep cross-origin font/CDN loads working
      // without CORP headers while still enabling cross-origin isolation.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
})
