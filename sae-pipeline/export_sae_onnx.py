"""S3 — Export the `gpt2-small-res-jb` layer-L SAE encoder to ONNX + a raw W_dec blob.

Two artifacts feed the browser (S7 encoder, S9 steering):

1. ``sae_enc.onnx`` — the SAE *encoder* baked as a graph:

       feats = ReLU( (x - b_dec) @ W_enc + b_enc )                     # d_sae outputs

   ``x`` is the layer-L residual **already in the SAE basis**, i.e. the browser has
   applied ``basis_transform.to_sae_input`` (mean-center over d_model) *before* calling
   this graph — the centering is NOT baked in (see BASIS_CONTRACT.md → "S3"). Because
   ``apply_b_dec_to_input == True`` for this release, the encoder pre-subtracts ``b_dec``;
   because ``normalize_activations == "none"`` there is no input scale factor.

   Graph IO is **float32 on every variant** (fp32 and fp16), exactly like the
   model-pipeline segmented graphs, so the JS glue always passes/reads a Float32Array
   regardless of which variant loaded. The fp16 variant stores weights in fp16 but keeps
   fp32 IO via ``keep_io_types=True`` (+ the ``force_outputs_fp32`` safety net copied from
   model-pipeline/quantize.py).

2. ``w_dec_fp16.bin`` / ``b_dec_fp32.bin`` — the *decoder* is NOT a graph. Steering is a JS
   vector add ``resid' = resid + α·W_dec[f]``, so the browser needs indexable rows. We ship
   ``W_dec`` as a raw little-endian fp16 buffer, shape ``[d_sae, d_in]`` C-contiguous:
   feature ``f`` occupies bytes ``f*d_in*2 .. (f+1)*d_in*2``. ``b_dec`` ships as fp32 (768
   floats) for any later ``x_hat = f @ W_dec + b_dec`` reconstruction.

Run:
    export HF_HOME=D:/dev/hf-cache
    D:/dev/sae-venv/Scripts/python.exe export_sae_onnx.py \
        --layer 8 --out D:/dev/sae-artifacts/L8

Parity is a separate step (test_encoder_parity.py). This script only builds + self-checks
the fp32 graph against the torch reference on a random input and writes a manifest with a
content_hash over all artifact bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxconverter_common import float16

RELEASE = "gpt2-small-res-jb"
D_MODEL = 768


# ---------------------------------------------------------------------------
# SAE loading (mirrors verify_basis.py's unpack — return signature drifts)
# ---------------------------------------------------------------------------
def load_sae(layer: int, device: str = "cpu"):
    from sae_lens import SAE

    hook = f"blocks.{layer}.hook_resid_pre"
    loaded = SAE.from_pretrained(release=RELEASE, sae_id=hook, device=device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    return sae.to(device).eval(), hook


# ---------------------------------------------------------------------------
# fp32 encoder graph:  feats = ReLU((x - b_dec) @ W_enc + b_enc)
# ---------------------------------------------------------------------------
def build_encoder_fp32(W_enc: np.ndarray, b_enc: np.ndarray, b_dec: np.ndarray) -> onnx.ModelProto:
    d_in, d_sae = W_enc.shape
    assert d_in == D_MODEL, f"W_enc first dim {d_in} != d_model {D_MODEL}"
    assert b_enc.shape == (d_sae,) and b_dec.shape == (d_in,)

    W_enc = np.ascontiguousarray(W_enc, dtype=np.float32)
    b_enc = np.ascontiguousarray(b_enc, dtype=np.float32)
    b_dec = np.ascontiguousarray(b_dec, dtype=np.float32)

    inits = [
        numpy_helper.from_array(b_dec, "b_dec"),
        numpy_helper.from_array(W_enc, "W_enc"),
        numpy_helper.from_array(b_enc, "b_enc"),
    ]
    nodes = [
        helper.make_node("Sub", ["x", "b_dec"], ["x_sub"], name="sub_b_dec"),
        helper.make_node("MatMul", ["x_sub", "W_enc"], ["pre"], name="enc_matmul"),
        helper.make_node("Add", ["pre", "b_enc"], ["pre_b"], name="add_b_enc"),
        helper.make_node("Relu", ["pre_b"], ["feats"], name="relu"),
    ]
    # Contract: [1, seq, 768] -> [1, seq, 24576], seq dynamic (batch fixed at 1 as S7 uses).
    x_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "seq", d_in])
    feats_out = helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, "seq", d_sae])

    graph = helper.make_graph(nodes, "sae_encoder", [x_in], [feats_out], initializer=inits)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="interp-sae-pipeline-s3",
    )
    model.ir_version = 10  # onnxruntime 1.27 / onnx 1.22 compatible
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------
# fp16 variant — reuse model-pipeline/quantize.py's recipe verbatim in spirit:
# keep_io_types=True so graph IO stays fp32; force_outputs_fp32 as a safety net.
# No LayerNorm / softmax / causal-mask here, so nothing can overflow fp16's 65504
# (see report). We still block LayerNormalization defensively (no-op on this graph).
# ---------------------------------------------------------------------------
def force_outputs_fp32(model: onnx.ModelProto) -> int:
    """Re-declare any fp16 graph output as fp32 via a terminal Cast->FLOAT.

    Copied from model-pipeline/quantize.py. No-op when outputs already round-trip fp32.
    """
    g = model.graph
    fixed = 0
    for out in g.output:
        if out.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        inner = out.name + "_fp16_inner"
        for node in g.node:
            for i, o in enumerate(node.output):
                if o == out.name:
                    node.output[i] = inner
            for i, inp in enumerate(node.input):
                if inp == out.name:
                    node.input[i] = inner
        g.node.append(helper.make_node(
            "Cast", [inner], [out.name], name=out.name + "_force_fp32", to=TensorProto.FLOAT))
        out.type.tensor_type.elem_type = TensorProto.FLOAT
        fixed += 1
    return fixed


def convert_fp16(model_fp32: onnx.ModelProto) -> onnx.ModelProto:
    op_block = list(float16.DEFAULT_OP_BLOCK_LIST) + ["LayerNormalization"]
    model_fp16 = float16.convert_float_to_float16(
        onnx.ModelProto.FromString(model_fp32.SerializeToString()),
        keep_io_types=True, op_block_list=op_block)
    force_outputs_fp32(model_fp16)
    onnx.checker.check_model(model_fp16)
    return model_fp16


# ---------------------------------------------------------------------------
# content_hash over ALL artifact bytes (encoder graphs + blobs), same scheme as
# model-pipeline/stamp_manifests.py (per-file sha256, combined, first 16 hex).
# ---------------------------------------------------------------------------
def content_hash(files: list[Path]) -> str:
    combined = hashlib.sha256()
    for f in sorted(files, key=lambda p: p.name):
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        combined.update(f"{f.name}:{h.hexdigest()}\n".encode())
    return combined.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--out", default="D:/dev/sae-artifacts/L8")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading SAE {RELEASE} blocks.{args.layer}.hook_resid_pre ...")
    sae, hook = load_sae(args.layer)
    cfg = sae.cfg
    d_in = int(getattr(cfg, "d_in"))
    d_sae = int(getattr(cfg, "d_sae"))
    apply_b_dec = bool(getattr(cfg, "apply_b_dec_to_input"))
    normalize = getattr(cfg, "normalize_activations")
    assert apply_b_dec is True, "release changed: apply_b_dec_to_input must be True (see BASIS_CONTRACT)"
    assert str(normalize) == "none", f"release changed: normalize_activations={normalize!r} (expected 'none')"

    W_enc = sae.W_enc.detach().cpu().numpy().astype(np.float32)   # [768, 24576]
    b_enc = sae.b_enc.detach().cpu().numpy().astype(np.float32)   # [24576]
    b_dec = sae.b_dec.detach().cpu().numpy().astype(np.float32)   # [768]
    W_dec = sae.W_dec.detach().cpu().numpy().astype(np.float32)   # [24576, 768]
    print(f"  W_enc {W_enc.shape}  b_enc {b_enc.shape}  b_dec {b_dec.shape}  W_dec {W_dec.shape}")

    # ---- fp32 encoder graph ----
    print("building fp32 encoder graph ...")
    m32 = build_encoder_fp32(W_enc, b_enc, b_dec)
    f32_path = out / "sae_enc.onnx"
    onnx.save(m32, str(f32_path))
    print(f"  wrote {f32_path.name}  {f32_path.stat().st_size/1e6:.1f} MB")

    # self-check: onnx fp32 vs torch encode on a random realistic-scale input
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 12, d_in)).astype(np.float32) * 60.0)
    x = x - x.mean(-1, keepdims=True)                       # to_sae_input (centering)
    sess = ort.InferenceSession(str(f32_path), providers=["CPUExecutionProvider"])
    f_onnx = sess.run(None, {"x": x})[0]
    f_torch = sae.encode(torch.from_numpy(x)).numpy()
    d = float(np.abs(f_onnx - f_torch).max())
    print(f"  self-check onnx-fp32 vs torch.encode max|diff| = {d:.2e}")
    assert d < 1e-4, f"fp32 graph self-check failed: {d}"

    # ---- fp16 encoder graph ----
    print("converting fp16 encoder graph ...")
    # Largest constant magnitude check (fp16 overflow guard).
    max_const = max(float(np.abs(a).max()) for a in (W_enc, b_enc, b_dec))
    print(f"  max |weight constant| = {max_const:.3g}  (fp16 max finite = 65504)")
    m16 = convert_fp16(m32)
    f16_path = out / "sae_enc_fp16.onnx"
    onnx.save(m16, str(f16_path))
    print(f"  wrote {f16_path.name}  {f16_path.stat().st_size/1e6:.1f} MB")

    # ---- W_dec + b_dec raw blobs ----
    print("writing decoder blobs ...")
    W_dec_fp16 = np.ascontiguousarray(W_dec, dtype=np.float16)    # [24576, 768] row-major
    wdec_path = out / "w_dec_fp16.bin"
    W_dec_fp16.tofile(str(wdec_path))                             # little-endian on x86
    bdec_path = out / "b_dec_fp32.bin"
    np.ascontiguousarray(b_dec, dtype=np.float32).tofile(str(bdec_path))
    print(f"  wrote {wdec_path.name}  {wdec_path.stat().st_size/1e6:.1f} MB  (shape {W_dec_fp16.shape})")
    print(f"  wrote {bdec_path.name}  {bdec_path.stat().st_size} B  (shape {b_dec.shape})")

    # blob round-trip sanity: reload and assert row f == W_dec[f] within fp16 tol
    reloaded = np.fromfile(str(wdec_path), dtype=np.float16).reshape(d_sae, d_in)
    rt = float(np.abs(reloaded.astype(np.float32) - W_dec).max())
    for f in (0, 1234, d_sae // 2, d_sae - 1):
        assert np.array_equal(reloaded[f], W_dec[f].astype(np.float16)), f"row {f} mismatch"
    print(f"  W_dec blob round-trip max|fp16-fp32 diff| = {rt:.2e}  (rows spot-checked OK)")

    # ---- manifest + content_hash ----
    artifact_files = [f32_path, f16_path, wdec_path, bdec_path]
    chash = content_hash(artifact_files)
    manifest = {
        "story": "S3",
        "model": "gpt2 (raw HuggingFace, via segmented ONNX)",
        "sae_release": RELEASE,
        "sae_id": hook,
        "layer": args.layer,
        "d_in": d_in,
        "d_sae": d_sae,
        "apply_b_dec_to_input": apply_b_dec,
        "normalize_activations": str(normalize),
        "encode_formula": "feats = ReLU((x - b_dec) @ W_enc + b_enc)",
        "basis_transform": "center: subtract per-token mean over d_model (applied by caller BEFORE the graph; see BASIS_CONTRACT.md)",
        "input_scale": None,
        "encoder_graph": {
            "input": {"name": "x", "dtype": "float32", "shape": [1, "seq", d_in]},
            "output": {"name": "feats", "dtype": "float32", "shape": [1, "seq", d_sae]},
            "io_dtype": "float32 on every variant (fp16 stores weights fp16, keeps fp32 IO)",
        },
        "w_dec_blob": {
            "file": wdec_path.name,
            "dtype": "float16",
            "byte_order": "little-endian",
            "shape": [d_sae, d_in],
            "layout": "C-contiguous row-major; row f = W_dec[f] at bytes f*d_in*2 .. (f+1)*d_in*2",
            "row_stride_bytes": d_in * 2,
        },
        "b_dec_blob": {
            "file": bdec_path.name,
            "dtype": "float32",
            "byte_order": "little-endian",
            "shape": [d_in],
            "note": "decoder bias for x_hat = f @ W_dec + b_dec (unused by encoder graph)",
        },
        "files": {p.name: p.stat().st_size for p in artifact_files},
        "content_hash": chash,
        "tooling": {
            "sae_lens": _ver("sae_lens"),
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
        },
    }
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {mpath}")
    print(f"content_hash = {chash}")
    print("sizes:", {k: f"{v/1e6:.1f}MB" for k, v in manifest["files"].items()})


def _ver(mod: str) -> str:
    try:
        return __import__(mod).__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
