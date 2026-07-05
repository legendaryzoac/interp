"""Segmented ONNX export of GPT-2 small.

Splits the model at residual-stream boundaries into embed.onnx,
block_00..block_11.onnx and unembed.onnx. Each block graph also outputs its
post-softmax per-head attention pattern. The web app chains the graphs and
holds the residual stream in JS, which is what makes logit lens (run unembed
on any intermediate residual) and activation patching (swap residuals between
runs at any boundary) cheap.

Note: the token embedding matrix appears in both embed.onnx and unembed.onnx
(GPT-2 ties them) — separate graphs can't share weights, so ~150MB fp32 is
duplicated. Quantization decisions account for this (see quantize.py).

Usage:
    python export.py --out D:/dev/interp-artifacts/onnx/fp32
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

MODEL_ID = "gpt2"
OPSET = 17


class Embed(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.wte = model.transformer.wte
        self.wpe = model.transformer.wpe

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        return self.wte(input_ids) + self.wpe(positions)


class Block(torch.nn.Module):
    """One transformer block, re-emitting the post-softmax attention pattern.

    transformers 5.x changed GPT2Block.forward to return a *bare* hidden-states
    tensor and no longer surfaces attention weights per block (output_attentions
    is plumbed through the top-level model, not the block). The old wrapper's
    `self.block(resid, output_attentions=True)[0], [-1]` therefore indexed into
    the tensor's rows and silently produced rank-2 garbage for both outputs --
    that is the root cause of the "Invalid rank ... Got: 2 Expected: 3" failure.

    We instead replicate GPT2Block.forward by hand and call the attention module
    directly (which returns (attn_output, attn_weights)), so the pattern is
    captured cleanly. Verified bit-exact against HF's full-transformer
    resid_post and .attentions. A causal additive mask must be built here
    because, run in isolation, eager attention has no mask and would be
    bidirectional.

    Mask fill value is -1e4, NOT torch.finfo(dtype).min (-3.4e38): the fp16
    variant executes with native f16 arithmetic on the WebGPU EP, where a
    -3.4e38-derived constant saturates/overflows (f16 max is 65504) and
    `score + mask` becomes -inf, cascading to NaN through f16 softmax on some
    drivers. CPU/WASM EPs upcast internally, so validation here cannot see
    that failure mode. -1e4 is exactly representable in fp16, cannot overflow
    when summed with attention scores (~ +/-100), and matches the original
    GPT-2/HF `masked_bias = -1e4` convention. fp32 results are unchanged:
    exp((-1e4 + score) - rowmax) underflows to exactly 0.0, the same masked-
    cell zeros the -3.4e38 mask produced.
    """

    def __init__(self, block):
        super().__init__()
        self.block = block

    MASK_VALUE = -1e4  # fp16-representable causal mask fill (see class docstring)

    def forward(self, resid):
        b = self.block
        seq = resid.shape[1]
        causal = torch.triu(
            torch.full((seq, seq), self.MASK_VALUE, dtype=resid.dtype), diagonal=1
        )[None, None, :, :]  # [1, 1, seq, seq] additive mask

        attn_out, pattern = b.attn(b.ln_1(resid), attention_mask=causal)
        hidden = resid + attn_out
        hidden = hidden + b.mlp(b.ln_2(hidden))
        return hidden, pattern  # resid_out [1, seq, 768], pattern [1, heads, seq, seq]


class Unembed(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head

    def forward(self, resid):
        return self.lm_head(self.ln_f(resid))


def export_graph(module, example_inputs, path, input_names, output_names, dynamic_axes):
    torch.onnx.export(
        module,
        example_inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output dir for fp32 graphs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID} (eager attention -- required for pattern outputs)")
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    model.eval()
    cfg = model.config

    seq = 9  # arbitrary; seq axis is dynamic
    dummy_ids = torch.randint(0, cfg.vocab_size, (1, seq), dtype=torch.long)
    dummy_resid = torch.randn(1, seq, cfg.n_embd)

    with torch.no_grad():
        print("exporting embed.onnx")
        export_graph(
            Embed(model), (dummy_ids,), out / "embed.onnx",
            ["input_ids"], ["resid"],
            {"input_ids": {1: "seq"}, "resid": {1: "seq"}},
        )

        for i, block in enumerate(model.transformer.h):
            print(f"exporting block_{i:02d}.onnx")
            export_graph(
                Block(block), (dummy_resid,), out / f"block_{i:02d}.onnx",
                ["resid"], ["resid_out", "pattern"],
                {
                    "resid": {1: "seq"},
                    "resid_out": {1: "seq"},
                    "pattern": {2: "seq", 3: "seq"},
                },
            )

        print("exporting unembed.onnx")
        export_graph(
            Unembed(model), (dummy_resid,), out / "unembed.onnx",
            ["resid"], ["logits"],
            {"resid": {1: "seq"}, "logits": {1: "seq"}},
        )

    manifest = {
        "model_id": MODEL_ID,
        "opset": OPSET,
        "n_layers": cfg.n_layer,
        "n_heads": cfg.n_head,
        "d_model": cfg.n_embd,
        "n_ctx": cfg.n_positions,
        "vocab_size": cfg.vocab_size,
        "tokenizer": "gpt2 (byte-level BPE)",
        "files": {p.name: p.stat().st_size for p in sorted(out.glob("*.onnx"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(manifest["files"].values()) / 1e6
    print(f"done: {len(manifest['files'])} graphs, {total:.0f} MB total (fp32)")


if __name__ == "__main__":
    main()
