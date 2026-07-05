import { useState } from 'react'
import promptData from '../data/prompts.json'

interface ExamplePrompt {
  id: string
  text: string
}

const EXAMPLES: ExamplePrompt[] = (promptData as { prompts: ExamplePrompt[] })
  .prompts

/**
 * Prompt input + Run button + example-prompt picker. `busy` disables input
 * during a run; `disabled` (model not ready) greys everything.
 */
export default function PromptBar({
  value,
  onChange,
  onRun,
  busy,
  disabled,
}: {
  value: string
  onChange: (v: string) => void
  onRun: () => void
  busy: boolean
  disabled: boolean
}) {
  const [pickerOpen, setPickerOpen] = useState(false)

  return (
    <div className="border-line bg-panel rounded-xl border p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !busy && !disabled) {
              e.preventDefault()
              onRun()
            }
          }}
          rows={2}
          spellCheck={false}
          placeholder="Enter a prompt for GPT-2…"
          className="border-line bg-site text-fg focus:border-accent placeholder:text-muted/60 min-h-[3.2rem] flex-1 resize-y rounded-lg border px-3 py-2 font-mono text-sm outline-none"
        />
        <div className="flex gap-2 sm:flex-col">
          <button
            onClick={onRun}
            disabled={busy || disabled || value.trim().length === 0}
            className="bg-accent text-site font-display hover:bg-accent-dim disabled:bg-line disabled:text-muted flex-1 rounded-lg px-5 py-2 text-sm font-semibold transition-colors disabled:cursor-not-allowed sm:flex-none"
          >
            {busy ? 'Running…' : 'Run'}
          </button>
          <button
            onClick={() => setPickerOpen((o) => !o)}
            className="border-line text-muted hover:text-fg hover:border-accent-dim font-display rounded-lg border px-3 py-2 text-sm transition-colors"
          >
            Examples ▾
          </button>
        </div>
      </div>

      {pickerOpen && (
        <div className="border-line mt-3 grid max-h-56 grid-cols-1 gap-1 overflow-y-auto rounded-lg border p-2 sm:grid-cols-2">
          {EXAMPLES.map((p) => (
            <button
              key={p.id}
              onClick={() => {
                onChange(p.text)
                setPickerOpen(false)
              }}
              className="hover:bg-panel-2 group flex flex-col rounded px-2 py-1.5 text-left transition-colors"
            >
              <span className="text-accent font-mono text-[0.68rem] tracking-wide">
                {p.id}
              </span>
              <span className="text-muted group-hover:text-fg truncate font-mono text-xs">
                {p.text.replace(/\n/g, '⏎')}
              </span>
            </button>
          ))}
        </div>
      )}

      <div className="text-muted mt-2 font-mono text-[0.68rem]">
        ⌘/Ctrl + Enter to run
      </div>
    </div>
  )
}
