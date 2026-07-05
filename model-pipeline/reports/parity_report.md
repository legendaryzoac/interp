# ONNX export parity report

Reference: TransformerLens `from_pretrained_no_processing("gpt2")` over 16 fixed prompts (`prompts.json`). Logit-lens agreement is measured across all (layer, position) cells.

| Variant | Attn pattern max\|diff\| | Final logit max\|diff\| | Lens top-1 | Lens top-5 |
|---|---|---|---|---|
| fp32 | 8.46e-06 | 4.88e-04 | 100.00% | 100.00% |
| fp16 | 1.25e-02 | 4.53e-01 | 99.90% | 99.33% |
| int8 | 2.49e-02 | 3.51e+00 | 98.42% | 90.97% |

`int8` is the mixed low-bandwidth recipe shipped to the WASM path (weight-only int8 embed with per-dim MSE scales + fp16 blocks + outlier-split weight-only int8 unembed, ~252 MB; all compute fp32). Naive dynamic quantization is unusable here -- its per-tensor *activation* quantization collapses against GPT-2's residual-stream outliers (63% lens top-1). Recipe rationale and the full experiment table live in README.md.

Gates: PASS
