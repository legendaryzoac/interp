/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MODEL_BASE_URL?: string
  readonly VITE_SAE_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

// WebGPU feature-detection lives on navigator.gpu; typed loosely so we don't
// need the full @webgpu/types dependency for a presence check.
interface Navigator {
  readonly gpu?: unknown
}

// onnxruntime-web ships its wasm/mjs assets as package subpath exports; import
// them with Vite's ?url suffix to get a fingerprinted, same-origin URL string.
declare module 'onnxruntime-web/*?url' {
  const url: string
  export default url
}
