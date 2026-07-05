"""Precompute the showcase gallery, THROUGH THE fp32 ONNX CHAIN.

The gallery must match what the live browser produces, so every number here comes
from running the segmented fp32 graphs in onnxruntime -- NOT from TransformerLens.
(validate.py already proved the fp32 chain matches TL to <1e-3; here we just want
byte-for-byte the same code path the web app runs.)

For each selected prompt we emit:
  - tokens        : display strings (decoded per-token text, real spaces/newlines)
  - token_ids     : the GPT-2 BPE ids
  - lens_top5     : logit lens -- unembed(resid_post[layer]) softmaxed, top-5 per
                    (layer, position): [layer][pos] -> [{"t": token_str, "p": prob}]
  - patterns_u8   : all attention patterns [12 layers, 12 heads, T, T] quantized to
                    uint8 (value/scale, scale = max/255) and base64'd, for compact
                    transport. shape + scale let the client reconstruct floats.

Usage:
    python gallery.py --artifacts D:/dev/interp-artifacts/onnx
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import GPT2TokenizerFast

HERE = Path(__file__).parent
N_LAYERS = 12
GALLERY_IDS = ["pangram", "ioi", "induction", "factual"]


def load_sessions(variant_dir: Path):
    names = ["embed", "unembed"] + [f"block_{i:02d}" for i in range(N_LAYERS)]
    return {
        n: ort.InferenceSession(str(variant_dir / f"{n}.onnx"),
                                providers=["CPUExecutionProvider"])
        for n in names
    }


def run_chain(sessions, ids: np.ndarray):
    """Return per-layer resids (resid_post) and per-layer patterns."""
    resid = sessions["embed"].run(None, {"input_ids": ids})[0]
    resids, patterns = [], []
    for i in range(N_LAYERS):
        resid, pattern = sessions[f"block_{i:02d}"].run(None, {"resid": resid})
        resids.append(resid)
        patterns.append(pattern)
    return resids, patterns


def softmax_lastdim(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def lens_top5(sessions, resids, tokenizer):
    """[layer][pos] -> list of {"t": token_str, "p": prob} for the top-5 tokens."""
    out = []
    for r in resids:  # r: [1, seq, 768]
        logits = sessions["unembed"].run(None, {"resid": r})[0][0]  # [seq, vocab]
        probs = softmax_lastdim(logits)
        top5 = np.argsort(-probs, axis=-1)[:, :5]  # [seq, 5]
        layer_rows = []
        for pos in range(top5.shape[0]):
            cells = [
                {"t": tokenizer.decode([int(tid)]), "p": round(float(probs[pos, tid]), 6)}
                for tid in top5[pos]
            ]
            layer_rows.append(cells)
        out.append(layer_rows)
    return out


def pack_patterns_u8(patterns):
    """Stack [12, 12, T, T] float patterns, quantize to uint8, base64-encode.

    scale = global_max / 255 (per prompt); u8 = round(value / scale). Reconstruct
    client-side as float = u8 * scale."""
    arr = np.stack([p[0] for p in patterns], axis=0)  # [12, 12, T, T]
    max_val = float(arr.max())
    scale = max_val / 255.0 if max_val > 0 else 1.0
    u8 = np.clip(np.round(arr / scale), 0, 255).astype(np.uint8)
    return {
        "shape": list(u8.shape),
        "scale": scale,
        "data": base64.b64encode(u8.tobytes()).decode("ascii"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True,
                        help="parent dir containing fp32/ (gallery uses fp32 only)")
    args = parser.parse_args()
    fp32_dir = Path(args.artifacts) / "fp32"

    prompts = {p["id"]: p for p in
               json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))["prompts"]}
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    sessions = load_sessions(fp32_dir)

    out_prompts = []
    for pid in GALLERY_IDS:
        p = prompts[pid]
        ids = tokenizer(p["text"], return_tensors="np")["input_ids"].astype(np.int64)
        token_ids = ids[0].tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]  # display strings

        resids, patterns = run_chain(sessions, ids)
        out_prompts.append({
            "id": pid,
            "text": p["text"],
            "tokens": tokens,
            "token_ids": token_ids,
            "lens_top5": lens_top5(sessions, resids, tokenizer),
            "patterns_u8": pack_patterns_u8(patterns),
        })
        print(f"  {pid}: {len(token_ids)} tokens")

    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "gallery.json").write_text(
        json.dumps({"prompts": out_prompts}), encoding="utf-8")
    print(f"wrote {out_dir / 'gallery.json'} ({(out_dir / 'gallery.json').stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
