/**
 * Satellite-site nav matching zackwithers.com / the adversarial playground —
 * same tokens and layout, with an outlined back-link to the main site and a
 * live backend badge.
 */
export default function SiteNav({ badge }: { badge?: string }) {
  return (
    <nav className="bg-site/85 border-line fixed inset-x-0 top-0 z-50 flex h-[60px] items-center justify-between gap-x-3 border-b px-5 backdrop-blur-md sm:px-10">
      <div className="flex min-w-0 shrink items-baseline gap-2.5">
        <a
          href="https://zackwithers.com"
          className="text-accent font-mono text-sm tracking-wider whitespace-nowrap"
        >
          zw ~
        </a>
        <span className="text-muted hidden font-mono text-xs min-[480px]:inline">/</span>
        <span className="text-accent hidden truncate font-mono text-xs min-[480px]:inline">
          interp
        </span>
      </div>
      <div className="flex shrink-0 items-center gap-3 sm:gap-4 md:gap-6">
        {badge && (
          <span className="border-line bg-panel text-muted hidden rounded-full border px-3 py-1 font-mono text-[0.7rem] tracking-wide whitespace-nowrap md:inline">
            {badge}
          </span>
        )}
        <a
          href="https://playground.zackwithers.com"
          target="_blank"
          rel="noopener"
          className="font-display text-muted hover:text-fg hidden text-sm font-medium whitespace-nowrap transition-colors md:block"
        >
          Playground ↗
        </a>
        <a
          href="https://github.com/legendaryzoac/interp"
          target="_blank"
          rel="noopener"
          className="font-display text-muted hover:text-fg hidden text-sm font-medium whitespace-nowrap transition-colors sm:block"
        >
          GitHub ↗
        </a>
        <a
          href="https://zackwithers.com"
          className="border-accent text-accent font-display hover:bg-accent hover:text-site rounded border px-3.5 py-1.5 text-[0.82rem] font-medium whitespace-nowrap transition-colors"
        >
          ← zackwithers.com
        </a>
      </div>
    </nav>
  )
}
