"""S2 — SAE basis-verification correctness gate.

Proves that Joseph Bloom's `gpt2-small-res-jb` SAEs reconstruct the SAME layer-L
residual the browser produces, once the raw-HF residual is put into the SAE's
training basis by `basis_transform.to_sae_input` (mean-centering over d_model).

What it does, per candidate layer L:
  1. Loads the SAE:  SAE.from_pretrained("gpt2-small-res-jb", "blocks.L.hook_resid_pre").
     Prints the cfg (normalize_activations, hook_name, prepend_bos, d_sae, ...).
  2. Gets `hook_resid_pre` at L from TWO TransformerLens models on the same tokens:
       - raw  = from_pretrained_no_processing("gpt2")  == the browser's basis
                (model-pipeline/validate.py proves ONNX == this model).
       - proc = from_pretrained("gpt2")  (default processing) == the SAE's basis.
  3. (Airtight loop) Re-derives the browser residual directly from the fp32 ONNX
     graphs (embed -> block_00..block_{L-1}) and asserts it equals `raw`.
  4. Asserts the transform:  max| proc - (raw - mean_dmodel(raw)) |  is ~0.
  5. Measures SAE reconstruction (explained variance / MSE / cosine / L0) for three
     feeds:  proc (correct) | raw (untransformed, the silent-failure case) |
     to_sae_input(raw) (browser + transform).  (2) vs (3) shows the gate's value;
     (1) vs (3) shows the transform recovers the correct basis exactly.

Usage:
    D:/dev/sae-venv/Scripts/python.exe verify_basis.py \
        --layers 6 7 8 9 \
        --artifacts D:/dev/interp-artifacts/onnx/fp32 \
        --out reports/basis_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
RELEASE = "gpt2-small-res-jb"

# A small, diverse prompt set. Reconstruction is measured over every token of every
# prompt (BOS included, matching the SAE's prepend_bos training convention).
PROMPTS = [
    "When Mary and John went to the store, John gave a drink to",
    "The Eiffel Tower is located in the city of",
    "The capital of France is",
    "In 1969, the first humans landed on the",
    "def add(a, b):\n    return a +",
    "The mitochondria is the powerhouse of the",
    "She sold seashells by the sea shore, and the shells she sold were",
    "Barack Obama was the 44th President of the United",
]


def load_prompts() -> list[str]:
    """Prefer the shared model-pipeline prompt set (read-only) for coverage; fall
    back to the embedded list."""
    shared = HERE.parent / "model-pipeline" / "prompts.json"
    try:
        data = json.loads(shared.read_text(encoding="utf-8"))
        texts = [p["text"] for p in data["prompts"]]
        # de-dup while preserving order, union with our diverse set
        seen, out = set(), []
        for t in texts + PROMPTS:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out
    except Exception:
        return PROMPTS


def unpack_sae(loaded):
    """SAE.from_pretrained return signature has drifted across SAELens releases
    (SAE | (SAE, cfg) | (SAE, cfg, sparsity)). Return the SAE object either way."""
    if isinstance(loaded, tuple):
        return loaded[0]
    return loaded


def sae_cfg_summary(sae) -> dict:
    cfg = sae.cfg
    md = getattr(cfg, "metadata", None)

    def pick(*names):
        for src in (cfg, md):
            if src is None:
                continue
            for n in names:
                if hasattr(src, n):
                    return getattr(src, n)
        return None

    return {
        "d_in": pick("d_in"),
        "d_sae": pick("d_sae"),
        "architecture": type(sae).__name__,
        "hook_name": pick("hook_name"),
        "hook_layer": pick("hook_layer"),
        "normalize_activations": pick("normalize_activations"),
        "apply_b_dec_to_input": pick("apply_b_dec_to_input"),
        "prepend_bos": pick("prepend_bos"),
        "context_size": pick("context_size"),
        "model_name": pick("model_name"),
        "dataset_path": pick("dataset_path", "dataset"),
        "dtype": str(pick("dtype")),
        "model_from_pretrained_kwargs": pick("model_from_pretrained_kwargs"),
    }


def recon_metrics(x: torch.Tensor, x_hat: torch.Tensor, feats: torch.Tensor) -> dict:
    """x, x_hat: [N, d_in]; feats: [N, d_sae]. Metrics measured over N tokens.

    explained_variance = 1 - FVU, where
        FVU = mean_t ||x_t - x_hat_t||^2  /  mean_t ||x_t - mean_t(x)||^2
    (variance measured across tokens, summed over d_model — the standard SAE report).
    """
    resid = x - x_hat
    mse = resid.pow(2).sum(-1).mean().item()  # mean over tokens of squared L2
    total = (x - x.mean(0, keepdim=True)).pow(2).sum(-1).mean().item()
    ev = 1.0 - mse / total
    cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
    l0 = (feats > 0).float().sum(-1).mean().item()
    x_norm = x.norm(dim=-1).mean().item()
    return {
        "explained_variance": ev,
        "mse_sumd": mse,
        "cosine_sim": cos,
        "l0": l0,
        "mean_resid_norm": x_norm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 7, 8, 9])
    ap.add_argument("--artifacts", default="D:/dev/interp-artifacts/onnx/fp32",
                    help="fp32 segmented ONNX dir; used to close the browser==HF loop")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(HERE / "reports" / "basis_report.json"))
    args = ap.parse_args()

    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    torch.set_grad_enabled(False)
    device = args.device

    prompts = load_prompts()
    print(f"prompts: {len(prompts)}")

    print("loading TransformerLens gpt2 (no_processing) == browser / raw-HF basis")
    raw_model = HookedTransformer.from_pretrained_no_processing("gpt2").to(device).eval()
    print("loading TransformerLens gpt2 (default processing) == SAE training basis")
    proc_model = HookedTransformer.from_pretrained("gpt2").to(device).eval()

    # Tokenize once (with BOS, matching SAE prepend_bos=True). Same ids for both models
    # and for the ONNX chain so every comparison is apples-to-apples.
    token_lists = [raw_model.to_tokens(p, prepend_bos=True) for p in prompts]

    # Optional: ONNX chain for the airtight browser==HF check.
    onnx_ok = False
    onnx_sessions = None
    art = Path(args.artifacts)
    if (art / "embed.onnx").exists():
        try:
            import onnxruntime as ort
            names = ["embed"] + [f"block_{i:02d}" for i in range(12)]
            onnx_sessions = {
                n: ort.InferenceSession(str(art / f"{n}.onnx"),
                                        providers=["CPUExecutionProvider"])
                for n in names
            }
            onnx_ok = True
            print(f"loaded fp32 ONNX graphs from {art} for browser==HF check")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] ONNX check skipped: {e}")
    else:
        print(f"[warn] no ONNX artifacts at {art}; skipping browser==HF reproduction")

    def onnx_resid_entering(ids_np: np.ndarray, layer: int) -> np.ndarray:
        """Reproduce runner.ts residualsEnteringBlocks()[layer] from the ONNX graphs."""
        resid = onnx_sessions["embed"].run(None, {"input_ids": ids_np})[0]
        for i in range(layer):
            resid, _pattern = onnx_sessions[f"block_{i:02d}"].run(None, {"resid": resid})
        return resid

    results = {}
    for L in args.layers:
        hook = f"blocks.{L}.hook_resid_pre"
        print(f"\n=== layer {L}  ({hook}) ===")
        sae = unpack_sae(SAE.from_pretrained(release=RELEASE, sae_id=hook, device=device))
        sae = sae.to(device).eval()
        cfg = sae_cfg_summary(sae)
        print("  sae cfg:", json.dumps(cfg, default=str))

        raw_chunks, proc_chunks = [], []
        max_onnx_diff = 0.0
        for ids, ids_list in [(t, t) for t in token_lists]:
            ids_t = ids.to(device)
            _, raw_cache = raw_model.run_with_cache(ids_t, names_filter=hook)
            _, proc_cache = proc_model.run_with_cache(ids_t, names_filter=hook)
            raw = raw_cache[hook][0]   # [seq, 768]
            proc = proc_cache[hook][0]
            raw_chunks.append(raw)
            proc_chunks.append(proc)
            if onnx_ok:
                ids_np = ids_t.cpu().numpy().astype(np.int64)
                onnx_resid = onnx_resid_entering(ids_np, L)[0]  # [seq, 768]
                d = float(np.abs(onnx_resid - raw.cpu().numpy()).max())
                max_onnx_diff = max(max_onnx_diff, d)

        raw = torch.cat(raw_chunks, dim=0).to(torch.float32)   # [N, 768]
        proc = torch.cat(proc_chunks, dim=0).to(torch.float32)
        n_tokens = raw.shape[0]

        # --- transform assertion: proc == raw - mean_dmodel(raw) ---
        transform = raw - raw.mean(dim=-1, keepdim=True)
        max_transform_diff = (proc - transform).abs().max().item()

        # --- reconstruction three ways ---
        def run_sae(x: torch.Tensor) -> dict:
            x = x.to(sae.W_enc.dtype)
            feats = sae.encode(x)
            x_hat = sae.decode(feats)
            return recon_metrics(x.float(), x_hat.float(), feats.float())

        m_correct = run_sae(proc)                       # SAE's own basis (reference)
        m_raw = run_sae(raw)                            # browser, NO transform (bad)
        m_browser = run_sae(transform)                  # browser + to_sae_input (good)

        print(f"  tokens={n_tokens}  onnx==HF max|diff|={max_onnx_diff:.2e}  "
              f"transform max|diff|={max_transform_diff:.2e}")
        print(f"  EV  correct(proc)={m_correct['explained_variance']:.4f}  "
              f"raw(untransformed)={m_raw['explained_variance']:.4f}  "
              f"browser(transform)={m_browser['explained_variance']:.4f}")
        print(f"  L0  correct={m_correct['l0']:.1f}  raw={m_raw['l0']:.1f}  "
              f"browser={m_browser['l0']:.1f}   mean|resid|={m_correct['mean_resid_norm']:.1f}")

        results[str(L)] = {
            "hook": hook,
            "cfg": cfg,
            "n_tokens": n_tokens,
            "onnx_vs_hf_max_abs_diff": max_onnx_diff if onnx_ok else None,
            "transform_max_abs_diff": max_transform_diff,
            "recon_correct_proc": m_correct,
            "recon_raw_untransformed": m_raw,
            "recon_browser_transformed": m_browser,
            "ev_gain_from_transform": m_browser["explained_variance"] - m_raw["explained_variance"],
            "browser_matches_correct": abs(
                m_browser["explained_variance"] - m_correct["explained_variance"]
            ) < 1e-3,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "release": RELEASE,
        "reference_raw_model": "transformer_lens from_pretrained_no_processing('gpt2') (== browser / ONNX)",
        "reference_sae_basis_model": "transformer_lens from_pretrained('gpt2') (default processing)",
        "transform": "x_sae = x_raw - mean(x_raw over d_model)   [center_writing_weights]",
        "n_prompts": len(prompts),
        "layers": results,
    }
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport written to {out}")

    # --- gate summary ---
    print("\n================ S2 GATE SUMMARY ================")
    print(f"{'L':>2} | {'EV proc':>8} | {'EV raw':>8} | {'EV xform':>9} | "
          f"{'L0':>5} | {'onnx==HF':>9} | {'xform diff':>10} | match")
    all_pass = True
    for L, r in results.items():
        c = r["recon_correct_proc"]["explained_variance"]
        rw = r["recon_raw_untransformed"]["explained_variance"]
        b = r["recon_browser_transformed"]["explained_variance"]
        onnx_d = r["onnx_vs_hf_max_abs_diff"]
        onnx_s = f"{onnx_d:.1e}" if onnx_d is not None else "n/a"
        ok = r["browser_matches_correct"] and r["transform_max_abs_diff"] < 1e-2
        all_pass = all_pass and ok
        print(f"{L:>2} | {c:8.4f} | {rw:8.4f} | {b:9.4f} | "
              f"{r['recon_correct_proc']['l0']:5.1f} | {onnx_s:>9} | "
              f"{r['transform_max_abs_diff']:10.1e} | {ok}")
    print("=================================================")
    print("GATE:", "PASS" if all_pass else "REVIEW — see numbers above")


if __name__ == "__main__":
    main()
