"""Produce fp16 and int8 variants of the fp32 segmented export.

- fp16: onnxconverter-common weight/compute conversion, keep_io_types=True so
  the JS glue passes Float32Array at every graph boundary regardless of variant.
- int8 ("low-bandwidth" variant, default on the WASM path): a MIXED recipe --
  weight-only int8 embed + outlier-split weight-only int8 unembed + fp16 blocks.
  All compute stays float; int8 is storage-only via DequantizeLinear (standard
  ONNX ops, safe for onnxruntime-web).

  Why not onnxruntime dynamic quantization: quantize_dynamic is NOT weight-only.
  It inserts DynamicQuantizeLinear, which quantizes *activations* with a single
  per-tensor scale at runtime. GPT-2's residual stream has extreme activation
  outliers (|x| up to ~370 while typical values are ~1), so that one scale
  destroys the signal: measured 63% lens top-1 / 10% top-5 / 0.97 pattern diff.
  Per-channel weight scales cannot fix it (measured: blocks stay at 0.86 pattern
  diff) because the damage is in the activations. Full experiment table in
  reports/parity_report.md history and README.md.

  Recipe details (each choice is measured, see README experiment table):
  - embed: int8 wte/wpe with per-DIM (axis=1) MSE-optimal scales. Per-row scales
    fail (0.145 pattern diff) because GPT-2 has outlier *dimensions*; per-dim
    isolates them (0.025).
  - blocks: fp16. Weight-only int8 blocks corrupt attention patterns (0.15-0.17
    max diff vs the 5e-2 gate) through accumulated residual drift, regardless of
    max-abs or MSE scales.
  - unembed: int8 per-dim MSE scales, plus the k=16 highest-range rows of the
    lm_head kept in fp32 as a parallel skinny MatMul (LLM.int8-style split;
    dim 138 and friends). Lifts lens top-1 ~+0.7pp over plain per-dim int8 for
    +3.2 MB.

Usage:
    python quantize.py --src D:/dev/interp-artifacts/onnx/fp32 \
                       --out-root D:/dev/interp-artifacts/onnx
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxconverter_common import float16

# The block graphs emit `pattern` straight off the attention's internal
# `attn_weights.type(value.dtype)` Cast node (from transformers'
# eager_attention_forward). onnxconverter-common's keep_io_types path mishandles
# an fp32-typed output that terminates on a Cast: it renames the Cast's output to
# the same string as the boundary-cast node it inserts, producing an invalid
# graph ("Type (float16) of output arg does not match expected type (float)").
# Blocking that node keeps it in fp32 and sidesteps the collision, but the
# converter then declares the graph output fp16. force_outputs_fp32 restores the
# contract (all variants keep fp32 graph IO) by appending a clean Cast->FLOAT.
PATTERN_CAST_NODE = "/attn/Cast"


def force_outputs_fp32(model):
    """Re-declare any fp16 graph output as fp32, inserting a terminal Cast->FLOAT.

    Variant-agnostic: only touches outputs the converter left as fp16, so it is a
    no-op on graphs whose outputs already round-tripped as fp32.
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
            "Cast", [inner], [out.name],
            name=out.name + "_force_fp32", to=TensorProto.FLOAT))
        out.type.tensor_type.elem_type = TensorProto.FLOAT
        fixed += 1
    return fixed


def convert_fp16(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    # LayerNormalization must stay fp32: GPT-2's residual stream reaches
    # |x| ~ 3e3 in late blocks (and ~4e2 into unembed's ln_f), so any native-f16
    # LN kernel that squares before normalizing overflows f16's 65504 max
    # (x^2 up to ~9.4e6) -> inf/NaN variance -> NaN patterns and degenerate
    # logits. CPU/WASM EPs upcast f16 ops to f32 internally and never hit this;
    # the WebGPU EP executes f16 natively and does. Blocking LN wraps it in
    # Cast(f32) -> LN -> Cast(f16): the residual VALUES fit f16 fine (3e3 <<
    # 65504) -- only the squared intermediates inside LN do not.
    op_block = list(float16.DEFAULT_OP_BLOCK_LIST) + ["LayerNormalization"]
    for f in sorted(src.glob("*.onnx")):
        model = onnx.load(str(f))
        # Block the pattern-terminal Cast (present only in block graphs); harmless
        # elsewhere since the name simply won't match.
        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True, op_block_list=op_block,
            node_block_list=[PATTERN_CAST_NODE])
        force_outputs_fp32(model_fp16)
        onnx.checker.check_model(model_fp16)
        onnx.save(model_fp16, str(dst_dir / f.name))
        print(f"  fp16 {f.name}: {(dst_dir / f.name).stat().st_size / 1e6:.1f} MB")


UNEMBED_OUTLIER_ROWS = 16  # fp32 rows kept out of the int8 lm_head (see docstring)


def _mse_scales(w: np.ndarray, axis: int) -> np.ndarray:
    """Symmetric per-channel int8 scales minimizing quantization MSE.

    Searches clip factors in [0.5, 1.0] of max-abs per channel; clipping a few
    outliers in exchange for a finer grid measurably beats plain max-abs here.
    """
    reduce_axes = tuple(i for i in range(w.ndim) if i != axis)
    base = np.maximum(np.abs(w).max(axis=reduce_axes) / 127.0, 1e-12)
    wm = np.moveaxis(w, axis, 0).reshape(w.shape[axis], -1)
    best, best_err = base.copy(), np.full(w.shape[axis], np.inf)
    for alpha in np.arange(0.50, 1.01, 0.05):
        s = np.maximum(base * alpha, 1e-12)
        q = np.clip(np.round(wm / s[:, None]), -127, 127)
        err = ((q * s[:, None] - wm) ** 2).mean(axis=1)
        better = err < best_err
        best[better] = s[better]
        best_err[better] = err[better]
    return best.astype(np.float32)


def _quantize_int8(w: np.ndarray, scale: np.ndarray, axis: int) -> np.ndarray:
    shape = [1] * w.ndim
    shape[axis] = -1
    return np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)


def _add_weight_qdq(graph, inits, name: str, axis: int) -> str:
    """Replace fp32 initializer `name` with int8 + DequantizeLinear; return the
    dequantized tensor name. Compute stays fp32 -- int8 is storage only."""
    w = numpy_helper.to_array(inits[name])
    scale = _mse_scales(w, axis)
    q = _quantize_int8(w, scale, axis)
    qn, sn, zn, dn = name + "_q8", name + "_q8_scale", name + "_q8_zp", name + "_deq"
    graph.initializer.remove(inits[name])
    graph.initializer.extend([
        numpy_helper.from_array(q, qn),
        numpy_helper.from_array(scale, sn),
        numpy_helper.from_array(np.zeros(scale.shape, np.int8), zn),
    ])
    graph.node.insert(0, helper.make_node(
        "DequantizeLinear", [qn, sn, zn], [dn], name=name + "_dq", axis=axis))
    return dn


def quantize_embed(src_file: Path, dst_file: Path):
    """Weight-only int8 embed: per-dim (axis=1) scales on the wte/wpe tables."""
    m = onnx.load(str(src_file))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    for node in g.node:
        if node.op_type != "Gather":
            continue
        data = node.input[0]
        if (data in inits and inits[data].data_type == TensorProto.FLOAT
                and len(inits[data].dims) == 2
                and int(np.prod(inits[data].dims)) >= 100_000):
            node.input[0] = _add_weight_qdq(g, inits, data, axis=1)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst_file))


def quantize_unembed(src_file: Path, dst_file: Path, k: int = UNEMBED_OUTLIER_ROWS):
    """Outlier-split weight-only int8 unembed.

    logits = MatMul(x, W_int8_deq) + MatMul(Gather(x, outlier_dims), W_outlier_fp32)
    where the k rows of W with the largest |value| (GPT-2's outlier dims, e.g.
    138) are zeroed in the int8 part and carried exactly by the fp32 branch.
    """
    m = onnx.load(str(src_file))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    mm = next(n for n in g.node
              if n.op_type == "MatMul" and n.input[1] in inits
              and len(inits[n.input[1]].dims) == 2)
    wname = mm.input[1]
    W = numpy_helper.to_array(inits[wname])  # [d_model, vocab]

    out_rows = np.sort(np.argsort(-np.abs(W).max(axis=1))[:k]).astype(np.int64)
    W_main = W.copy()
    W_main[out_rows, :] = 0.0
    scale = _mse_scales(W_main, axis=0)
    q = _quantize_int8(W_main, scale, axis=0)

    x_in, orig_out = mm.input[0], mm.output[0]
    g.initializer.remove(inits[wname])
    g.initializer.extend([
        numpy_helper.from_array(q, wname + "_q8"),
        numpy_helper.from_array(scale, wname + "_q8_scale"),
        numpy_helper.from_array(np.zeros(scale.shape, np.int8), wname + "_q8_zp"),
        numpy_helper.from_array(W[out_rows, :].copy(), wname + "_outrows"),
        numpy_helper.from_array(out_rows, wname + "_outidx"),
    ])
    mm.input[1] = wname + "_deq"
    mm.output[0] = orig_out + "_main"

    g.node.insert(0, helper.make_node(
        "DequantizeLinear", [wname + "_q8", wname + "_q8_scale", wname + "_q8_zp"],
        [wname + "_deq"], name=wname + "_dq", axis=0))
    mm_idx = list(g.node).index(mm)
    for i, n in enumerate([
        helper.make_node("Gather", [x_in, wname + "_outidx"], [x_in + "_outdims"],
                         name=wname + "_gather_x", axis=-1),
        helper.make_node("MatMul", [x_in + "_outdims", wname + "_outrows"],
                         [orig_out + "_outlier"], name=wname + "_mm_out"),
        helper.make_node("Add", [orig_out + "_main", orig_out + "_outlier"],
                         [orig_out], name=wname + "_add_out"),
    ], start=1):
        g.node.insert(mm_idx + i, n)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst_file))


def build_int8(src: Path, fp16_dir: Path, dst_dir: Path):
    """Assemble the mixed low-bandwidth variant (dir name stays `int8` -- the
    web runner selects it by path and needs zero changes)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.onnx")):
        dst = dst_dir / f.name
        if f.name == "embed.onnx":
            quantize_embed(f, dst)
        elif f.name == "unembed.onnx":
            quantize_unembed(f, dst)
        else:  # blocks: fp16 copies (int8 blocks corrupt attention patterns)
            shutil.copy2(fp16_dir / f.name, dst)
        print(f"  int8 {f.name}: {dst.stat().st_size / 1e6:.1f} MB")


def write_manifest(src: Path, dst_dir: Path):
    manifest = json.loads((src / "manifest.json").read_text())
    # Never inherit the source variant's content_hash: it describes fp32's bytes,
    # not this variant's, and the web runner keys its download cache on it --
    # a stale inherited hash would pin returning visitors to old graphs. The
    # correct per-variant hash is stamped by stamp_manifests.py (last step).
    manifest.pop("content_hash", None)
    manifest["files"] = {p.name: p.stat().st_size for p in sorted(dst_dir.glob("*.onnx"))}
    (dst_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return sum(manifest["files"].values()) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="fp32 graph dir (from export.py)")
    parser.add_argument("--out-root", required=True, help="parent dir for fp16/ and int8/")
    args = parser.parse_args()
    src = Path(args.src)
    out_root = Path(args.out_root)

    print("converting to fp16")
    convert_fp16(src, out_root / "fp16")
    total16 = write_manifest(src, out_root / "fp16")

    print("building int8 (weight-only int8 embed/unembed + fp16 blocks)")
    build_int8(src, out_root / "fp16", out_root / "int8")
    total8 = write_manifest(src, out_root / "int8")

    total32 = sum(p.stat().st_size for p in src.glob("*.onnx")) / 1e6
    print(f"totals: fp32 {total32:.0f} MB | fp16 {total16:.0f} MB | int8 {total8:.0f} MB")


if __name__ == "__main__":
    main()
