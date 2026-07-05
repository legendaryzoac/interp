/**
 * Renders a tokenized prompt as labeled chips with visible whitespace markers,
 * so a viewer can see exactly how GPT-2's BPE split the text (and that " the"
 * is a different token than "the").
 */
export default function TokenStrip({
  tokens,
  ids,
}: {
  tokens: string[]
  ids?: number[]
}) {
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="text-muted mr-1 font-mono text-[0.68rem]">
        {tokens.length} tokens:
      </span>
      {tokens.map((t, i) => (
        <span
          key={i}
          className="border-line bg-panel-2 text-fg inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-[0.72rem]"
          title={ids ? `id ${ids[i]}` : undefined}
        >
          {t}
        </span>
      ))}
    </div>
  )
}
