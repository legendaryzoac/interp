# web — GPT-2 mech-interp visualizer (M1.1)

React + Vite + TypeScript + Tailwind v4 + D3. GPT-2 small runs **client-side**
via `onnxruntime-web` (WebGPU with WASM fallback); the app exposes attention
patterns and the logit lens. Sibling of the adversarial playground — same dark
technical design tokens.

## Dev

```bash
npm install          # from repo root (npm workspace) or from web/
npm run dev          # vite, default port 5273
npm run build        # tsc -b && vite build
npm test             # vitest (tokenizer parity)
```

The 14 ONNX graphs live outside the repo at `D:/dev/interp-artifacts/onnx/`
(`fp32|fp16|int8`). A dev-only Vite middleware (`vite.config.ts` → `serveModels`)
serves them at `/models/<variant>/<file>` so nothing is copied onto the C: drive.
Override the artifacts root with `INTERP_ARTIFACTS_ROOT`, and the model base URL
(for prod/CDN) with `VITE_MODEL_BASE_URL` (default `/models`).

Force a backend for testing the fallback path on a WebGPU-capable machine:
`http://localhost:5273/?backend=wasm` (or `?backend=webgpu`).

## Architecture

| File | Role |
|---|---|
| `src/lib/runner.ts` | Backend pick, download+cache (Cache API), session reuse, `run()` chaining embed→blocks→unembed, top-k softmax helper. Keeps per-layer residuals for later milestones (patching/comparison). |
| `src/lib/tokenizer.ts` | GPT-2 BPE via `gpt-tokenizer/encoding/r50k_base` (NOT the default cl100k export). Visible whitespace markers. |
| `src/lib/gallery.ts` | Optional `/gallery.json` precomputed examples; missing file never errors. |
| `src/lib/viewModel.ts` | One `ResultView` shape produced from either a live run or a gallery entry. |
| `src/lib/color.ts` | Attention (teal) + logit-lens (amber) color scales. |
| `src/components/` | `AttentionView` (12×12 small-multiples + zoom + arcs), `LogitLensView`, `PromptBar`, `DownloadOverlay`, `TokenStrip`, `SiteNav`. |

## Contract note (artifact bug)

Per the milestone brief, the **fp32 embed graph had a rank bug** being fixed in
parallel. In addition, the **fp16 block graphs currently emit their `pattern`
output as `tensor(float16)`**, which violates the frozen "all float32 IO"
contract and makes ORT reject session creation on the WebGPU/fp16 path:

```
Type (tensor(float16)) of output arg (/attn/Cast_cast_to_pattern) …
does not match expected type (tensor(float)).
```

The **int8 variant is unaffected** and runs the full chain end-to-end. The app
picks int8 on the WASM path and fp16 on the WebGPU path, so once the fp16 export
is fixed both paths work with no code change here. See the agent report for
verification details.
