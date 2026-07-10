import { useCallback, useEffect, useRef, useState } from 'react'
import SiteNav from './components/SiteNav'
import PromptBar from './components/PromptBar'
import TokenStrip from './components/TokenStrip'
import DownloadOverlay from './components/DownloadOverlay'
import AttentionView from './components/AttentionView'
import LogitLensView from './components/LogitLensView'
import CompareView from './components/CompareView'
import CircuitsView from './components/CircuitsView'
import SiteFooter from './components/SiteFooter'
import {
  detectBackend,
  Runner,
  type BackendChoice,
  type DownloadProgress,
  type RunResult,
} from './lib/runner'
import { tokenize } from './lib/tokenizer'
import { viewFromRun, viewFromGallery, type ResultView } from './lib/viewModel'
import { loadGallery, type Gallery } from './lib/gallery'

type Tab = 'attention' | 'lens' | 'compare' | 'circuits'
type ModelState = 'idle' | 'loading' | 'ready' | 'error'

const DEFAULT_PROMPT =
  'When Mary and John went to the store, John gave a drink to'

export default function App() {
  const [backend, setBackend] = useState<BackendChoice | null>(null)
  const [modelState, setModelState] = useState<ModelState>('idle')
  const [progress, setProgress] = useState<DownloadProgress | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [running, setRunning] = useState(false)
  const [view, setView] = useState<ResultView | null>(null)
  const [tab, setTab] = useState<Tab>('attention')
  const [runMs, setRunMs] = useState<number | null>(null)

  const [gallery, setGallery] = useState<Gallery | null>(null)

  const runnerRef = useRef<Runner | null>(null)

  // Detect backend + probe the optional gallery on mount.
  useEffect(() => {
    detectBackend().then(setBackend)
    loadGallery().then(setGallery)
  }, [])

  const ensureRunner = useCallback(async (): Promise<Runner> => {
    if (runnerRef.current) return runnerRef.current
    if (!backend) throw new Error('backend not detected yet')
    setModelState('loading')
    setError(null)
    try {
      const r = await Runner.create(backend, setProgress)
      runnerRef.current = r
      setModelState('ready')
      return r
    } catch (e) {
      setModelState('error')
      setError(e instanceof Error ? e.message : String(e))
      throw e
    }
  }, [backend])

  const handleRun = useCallback(async () => {
    setRunning(true)
    setError(null)
    try {
      const runner = await ensureRunner()
      const toks = tokenize(prompt)
      if (toks.length === 0) throw new Error('prompt produced no tokens')
      const t0 = performance.now()
      const result = await runner.run(toks.map((t) => t.id))
      const t1 = performance.now()
      setRunMs(t1 - t0)
      setView(viewFromRun(result, toks.map((t) => t.display)))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }, [ensureRunner, prompt])

  // Live run helper the Compare tab drives (base + base+suffix forwards).
  // Ensures the model is loaded, then runs — sessions stay private to Runner.
  const runTokens = useCallback(
    async (ids: number[]): Promise<RunResult> => {
      const runner = await ensureRunner()
      return runner.run(ids)
    },
    [ensureRunner],
  )

  const showGalleryExample = useCallback(
    (idx: number) => {
      if (!gallery) return
      const g = gallery.prompts[idx]
      if (!g) return
      setPrompt(g.text)
      setView(viewFromGallery(g))
      setRunMs(null)
    },
    [gallery],
  )

  const busy = running || modelState === 'loading'
  const currentTokens = view?.tokens ?? tokenize(prompt).map((t) => t.display)

  return (
    <div className="min-h-screen pb-4">
      <SiteNav badge={backend?.label} />

      <main className="mx-auto max-w-6xl px-5 pt-24 sm:px-8">
        {/* Hero */}
        <header className="mb-8">
          <p className="text-accent mb-2 font-mono text-xs tracking-widest uppercase">
            // AI Safety Lab
          </p>
          <h1 className="font-display text-fg text-3xl font-bold tracking-tight sm:text-4xl">
            GPT-2, opened up
          </h1>
          <p className="text-muted mt-2 max-w-2xl text-sm leading-relaxed sm:text-base">
            GPT-2 is a small AI language model: give it some text and it predicts
            the word that comes next — normally a black box. This
            mechanistic-interpretability visualizer opens the box. The model runs
            entirely in your browser, and each tab below reveals a different
            piece of the machinery behind that guess: its{' '}
            <span className="text-accent">attention patterns</span>, its{' '}
            <span className="text-accent">logit lens</span>, and the internal{' '}
            <span className="text-accent">circuits</span> it uses. No ML
            background needed — every view has a plain-language guide.
          </p>
        </header>

        {/* Backend + gallery bar */}
        <div className="mb-4 flex flex-wrap items-center gap-3">
          {backend ? (
            <span className="border-line bg-panel text-muted rounded-full border px-3 py-1 font-mono text-[0.7rem]">
              backend: <span className="text-accent">{backend.label}</span>
              {backend.ep === 'wasm' && (
                <span className="text-muted"> (WebGPU unavailable)</span>
              )}
            </span>
          ) : (
            <span className="text-muted font-mono text-[0.7rem]">
              detecting backend…
            </span>
          )}
          {runMs != null && (
            <span className="border-line bg-panel text-muted rounded-full border px-3 py-1 font-mono text-[0.7rem]">
              inference: <span className="text-accent">{runMs.toFixed(0)} ms</span>
            </span>
          )}
        </div>

        <PromptBar
          value={prompt}
          onChange={setPrompt}
          onRun={handleRun}
          busy={busy}
          disabled={!backend}
        />

        {/* Gallery quick-picks (only if /gallery.json is present) */}
        {gallery && gallery.prompts.length > 0 && (
          <div className="border-line bg-panel/50 mt-3 rounded-xl border p-3">
            <div className="text-muted mb-2 font-mono text-[0.68rem]">
              precomputed examples — render instantly, no download
            </div>
            <div className="flex flex-wrap gap-1.5">
              {gallery.prompts.map((g, i) => (
                <button
                  key={g.id}
                  onClick={() => showGalleryExample(i)}
                  className="border-accent-dim text-accent hover:bg-accent hover:text-site rounded-full border px-2.5 py-1 font-mono text-[0.68rem] transition-colors"
                >
                  {g.id} ✦
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Token strip */}
        <div className="mt-4">
          <TokenStrip tokens={currentTokens} />
        </div>

        {/* Errors */}
        {error && (
          <div className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 p-3 font-mono text-xs text-red-300">
            {error}
          </div>
        )}

        {/* Loading overlay */}
        {modelState === 'loading' && progress && backend && (
          <div className="mt-6">
            <DownloadOverlay progress={progress} backendLabel={backend.label} />
          </div>
        )}

        {/* Results */}
        {(view || tab === 'compare' || tab === 'circuits') && (
          <section className="mt-8">
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <button
                onClick={() => setTab('attention')}
                className={`font-display rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                  tab === 'attention'
                    ? 'bg-accent text-site'
                    : 'border-line text-muted hover:text-fg border'
                }`}
              >
                Attention
              </button>
              <button
                onClick={() => setTab('lens')}
                className={`font-display rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                  tab === 'lens'
                    ? 'bg-accent text-site'
                    : 'border-line text-muted hover:text-fg border'
                }`}
              >
                Logit lens
              </button>
              <button
                onClick={() => setTab('compare')}
                className={`font-display rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                  tab === 'compare'
                    ? 'bg-accent text-site'
                    : 'border-line text-muted hover:text-fg border'
                }`}
              >
                Compare
              </button>
              <button
                onClick={() => setTab('circuits')}
                className={`font-display rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
                  tab === 'circuits'
                    ? 'bg-accent text-site'
                    : 'border-line text-muted hover:text-fg border'
                }`}
              >
                Circuits
              </button>
              {view?.source === 'precomputed' &&
                tab !== 'compare' &&
                tab !== 'circuits' && (
                <span className="border-accent-dim text-accent ml-1 rounded-full border px-2 py-0.5 font-mono text-[0.62rem]">
                  precomputed
                </span>
              )}
            </div>

            <div className="border-line bg-panel/40 rounded-xl border p-4 sm:p-6">
              {tab === 'compare' ? (
                <CompareView
                  basePrompt={prompt}
                  runTokens={runTokens}
                  modelReady={modelState === 'ready'}
                />
              ) : tab === 'circuits' ? (
                <CircuitsView
                  getRunner={ensureRunner}
                  modelReady={modelState === 'ready'}
                />
              ) : !view ? (
                <div className="text-muted py-6 text-center font-mono text-sm">
                  run a prompt to populate this view
                </div>
              ) : tab === 'attention' ? (
                <AttentionView view={view} />
              ) : (
                <LogitLensView view={view} />
              )}
            </div>
          </section>
        )}

        {!view &&
          tab !== 'compare' &&
          tab !== 'circuits' &&
          modelState !== 'loading' && (
            <div className="text-muted mt-10 text-center font-mono text-sm">
              enter a prompt and hit Run to load the model and see its internals —
              or explore{' '}
              <button
                onClick={() => setTab('compare')}
                className="text-accent underline decoration-dotted underline-offset-2 hover:no-underline"
              >
                Compare
              </button>{' '}
              /{' '}
              <button
                onClick={() => setTab('circuits')}
                className="text-accent underline decoration-dotted underline-offset-2 hover:no-underline"
              >
                Circuits
              </button>
            </div>
          )}

        <SiteFooter note="GPT-2 small runs entirely client-side via ONNX Runtime Web · 124M parameters" />
      </main>
    </div>
  )
}
