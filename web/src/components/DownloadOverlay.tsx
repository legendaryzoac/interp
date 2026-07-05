import type { DownloadProgress } from '../lib/runner'

function mb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1)
}

/**
 * Full-panel overlay shown while the 14 ONNX graphs download + instantiate.
 * Reports per-file name, file count, and total MB with a progress bar.
 */
export default function DownloadOverlay({
  progress,
  backendLabel,
}: {
  progress: DownloadProgress
  backendLabel: string
}) {
  const pct =
    progress.totalBytes > 0
      ? Math.min(100, (progress.loadedBytes / progress.totalBytes) * 100)
      : (progress.loadedFiles / progress.totalFiles) * 100

  return (
    <div className="border-line bg-panel/70 flex flex-col items-center justify-center gap-5 rounded-xl border p-10">
      <div className="text-center">
        <div className="font-display text-fg text-lg font-semibold">
          Loading GPT-2 into your browser
        </div>
        <div className="text-muted mt-1 font-mono text-xs">
          14 ONNX graphs · {backendLabel}
        </div>
      </div>

      <div className="w-full max-w-md">
        <div className="bg-site border-line h-2.5 w-full overflow-hidden rounded-full border">
          <div
            className="bg-accent h-full rounded-full transition-[width] duration-150"
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="text-muted mt-2 flex justify-between font-mono text-[0.72rem]">
          <span>
            {progress.current
              ? `↓ ${progress.current}.onnx`
              : progress.done
                ? 'instantiating sessions…'
                : 'preparing…'}
          </span>
          <span>
            {progress.loadedFiles}/{progress.totalFiles} files
          </span>
        </div>
        <div className="text-muted mt-1 text-center font-mono text-[0.72rem]">
          {mb(progress.loadedBytes)} / {mb(progress.totalBytes)} MB
        </div>
      </div>

      <p className="text-muted max-w-sm text-center text-xs leading-relaxed">
        The model runs entirely on your machine — nothing is sent to a server.
        Files are cached, so subsequent runs are instant.
      </p>
    </div>
  )
}
