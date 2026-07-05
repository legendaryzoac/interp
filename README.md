# interp — mechanistic interpretability, live in your browser

GPT-2 small running client-side (ONNX Runtime Web, WebGPU with WASM fallback) with the
internals exposed: attention patterns, logit lens, adversarial-suffix comparison, and
activation patching. No inference server — the model runs on *your* machine, same as its
sibling project, the [adversarial playground](https://playground.zackwithers.com).

Live at **interp.zackwithers.com** *(once M1.1 ships)*.

## How it works

The model is not exported as one ONNX graph. `model-pipeline/` splits GPT-2 at every
residual-stream boundary:

```
embed.onnx → block_00.onnx → … → block_11.onnx → unembed.onnx
                  │ each block also outputs its per-head attention pattern
```

The web app chains the graphs and holds the residual stream between them, which makes
the interesting features cheap instead of hacky:

- **Logit lens** — run `unembed.onnx` on any layer's residual.
- **Activation patching** — swap residual tensors between two runs at any boundary.
- **Attention views** — patterns are first-class graph outputs, not hooks.

Exported weights are validated against
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) before shipping
— see `model-pipeline/reports/` for the current parity report.

## Repo layout

| Dir | What |
|---|---|
| `model-pipeline/` | Python: segmented ONNX export, fp16/int8 quantization, TransformerLens parity suite |
| `web/` | React/Vite frontend (M1.1+) |
| `infra/` | CDK: S3 + CloudFront + Route 53 (M1.1+) |

## Roadmap

- **M1.0** — model pipeline + parity report *(in progress)*
- **M1.1** — core visualizer: attention heatmaps, token arcs, logit lens
- **M1.2** — adversarial bridge: prompt vs prompt+suffix comparison, GCG suffixes
  generated offline against this exact model
- **M1.3** — circuits: induction head detection, activation patching (IOI)
