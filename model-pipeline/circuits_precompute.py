"""Precompute the Circuits-tab reference data for the web app.

WHY this exists: the web app's "Circuits" tab shows two canonical mechanistic-
interpretability results on GPT-2 small as ground truth to compare the live,
in-browser model against:

  1. Induction heads -- a per-(layer, head) score for how strongly each attention
     head implements the induction pattern (attend from a repeated token back to
     the token that followed its previous occurrence). This is the classic way to
     *locate* induction heads (Elhage et al., "A Mathematical Framework"; Olsson
     et al., "In-context Learning and Induction Heads").

  2. Activation patching on the IOI task (Wang et al., "Interpretability in the
     Wild") -- a (layer x position) heatmap of how much patching a clean
     residual stream into a corrupted run recovers the correct answer. The
     recovery band localizes the S-inhibition / name-mover machinery.

Unlike gallery.py, this is REFERENCE data, not ONNX-parity data, so we use
TransformerLens HookedTransformer.from_pretrained("gpt2") with its processed
(folded-LayerNorm, centered) weights -- the standard, clean way to run these
experiments. Numbers here are the "textbook" answer the live model should echo.

Output: out/circuits.json, schema (exact):
    { "induction": { "scores": [12][12] },
      "patching": { "tokens": [...], "clean_target": str, "corrupt_target": str,
                    "heatmap": [layers][positions],
                    "logit_diff_clean": float, "logit_diff_corrupt": float } }
Floats rounded to 4dp to keep the file small.

Usage:
    D:/dev/interp-venv/Scripts/python.exe circuits_precompute.py
"""

import argparse
import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

HERE = Path(__file__).parent
MODEL_ID = "gpt2"

# Induction probe: a block of random tokens repeated once. The 2nd copy is where
# induction heads fire -- at each position in the 2nd copy, an induction head
# attends back to the token that followed the same token in the 1st copy.
INDUCTION_SEQ_LEN = 25   # tokens per copy (total prompt = 2*this + BOS)
INDUCTION_SEED = 1234    # fixed for reproducibility

# Known GPT-2 small induction heads from the literature, to sanity-check against.
KNOWN_INDUCTION_HEADS = [(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)]

# IOI minimal pair. Clean answer is " Mary" (the indirect object / non-repeated
# name). The corrupt swaps the SUBJECT name (the giver "John" -> "Mary"), which
# is a single-token change at one position and flips the answer to " John".
# (Verified empirically: swapping only the first-clause order does NOT flip the
# answer for GPT-2 small; swapping the giver name does.)
IOI_CLEAN = "When Mary and John went to the store, John gave a drink to"
IOI_CORRUPT = "When Mary and John went to the store, Mary gave a drink to"
IOI_CLEAN_TARGET = " Mary"
IOI_CORRUPT_TARGET = " John"


def induction_scores(model) -> torch.Tensor:
    """Return a [n_layers, n_heads] tensor of induction scores.

    For a sequence = random block R of length L, repeated (so total length 2L,
    plus a BOS token TransformerLens prepends), an induction head at a position
    p in the SECOND copy attends to position (p - L + 1): i.e. the token right
    after the first occurrence of the current token. We average that specific
    attention weight over all second-copy positions and over the batch.
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    L = INDUCTION_SEQ_LEN

    torch.manual_seed(INDUCTION_SEED)
    # Sample from the middle of vocab to avoid special tokens; shape [1, L].
    rand = torch.randint(1000, model.cfg.d_vocab - 1000, (1, L))
    # Prepend BOS, then [R, R].
    bos = torch.tensor([[model.tokenizer.bos_token_id]])
    tokens = torch.cat([bos, rand, rand], dim=1)  # [1, 1 + 2L]

    _, cache = model.run_with_cache(
        tokens, return_type=None,
        names_filter=lambda n: n.endswith("hook_pattern"),
    )

    scores = torch.zeros(n_layers, n_heads)
    # Second copy occupies token positions [1 + L, 1 + 2L). With BOS at index 0,
    # the induction source for query position q is (q - L): the position right
    # after the first occurrence of that token. (First-copy token at q-L is the
    # SAME token as at q; the token that FOLLOWED it in copy 1 sits at q-L+1... but
    # the *destination we attend to* to predict the next token is q-L+1's key.)
    # Standard TransformerLens induction score uses offset -(L-1) => attend from q
    # to q-(L-1). We compute exactly that.
    for layer in range(n_layers):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # [heads, q, k]
        # query positions in the 2nd copy
        q_idx = torch.arange(1 + L, 1 + 2 * L)
        k_idx = q_idx - (L - 1)  # induction source position
        # gather pattern[head, q, k] for each (q,k) pair
        vals = pattern[:, q_idx, k_idx]  # [heads, L]
        scores[layer] = vals.mean(dim=1)
    return scores


def ioi_patching(model):
    """Residual-stream (resid_pre) activation patching over (layer x position).

    Metric: logit_diff = logit[" Mary"] - logit[" John"] at the final position.
    We patch each (layer, position) of resid_pre from the CLEAN run into the
    CORRUPT run and record the recovered logit_diff, then normalize:
        recovery = (patched_ld - corrupt_ld) / (clean_ld - corrupt_ld)
    so 0 = no recovery (corrupt behavior), 1 = full recovery (clean behavior).
    """
    mary = model.to_single_token(IOI_CLEAN_TARGET)
    john = model.to_single_token(IOI_CORRUPT_TARGET)

    clean_tokens = model.to_tokens(IOI_CLEAN)
    corrupt_tokens = model.to_tokens(IOI_CORRUPT)
    assert clean_tokens.shape == corrupt_tokens.shape, "pair must be token-aligned"
    n_pos = clean_tokens.shape[1]
    n_layers = model.cfg.n_layers

    def logit_diff(logits):
        last = logits[0, -1]
        return (last[mary] - last[john]).item()

    # Baselines.
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupt_logits = model(corrupt_tokens)
    ld_clean = logit_diff(clean_logits)
    ld_corrupt = logit_diff(corrupt_logits)
    denom = ld_clean - ld_corrupt

    heatmap = [[0.0] * n_pos for _ in range(n_layers)]
    for layer in range(n_layers):
        clean_resid = clean_cache[f"blocks.{layer}.hook_resid_pre"]  # [1, pos, d]
        for pos in range(n_pos):
            def hook(resid, hook, _pos=pos, _clean=clean_resid):
                resid[:, _pos, :] = _clean[:, _pos, :]
                return resid

            patched = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)],
            )
            ld = logit_diff(patched)
            recovery = (ld - ld_corrupt) / denom if denom != 0 else 0.0
            heatmap[layer][pos] = round(recovery, 4)

    tokens = model.to_str_tokens(IOI_CLEAN)
    return {
        "tokens": tokens,
        "clean_target": IOI_CLEAN_TARGET,
        "corrupt_target": IOI_CORRUPT_TARGET,
        "heatmap": heatmap,
        "logit_diff_clean": round(ld_clean, 4),
        "logit_diff_corrupt": round(ld_corrupt, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE / "out" / "circuits.json"))
    args = parser.parse_args()

    print(f"loading {MODEL_ID} via TransformerLens (processed weights)")
    model = HookedTransformer.from_pretrained(MODEL_ID, device="cpu")
    model.eval()

    print("computing induction scores (repeated-random-token sequence)")
    scores = induction_scores(model)
    scores_list = [[round(float(v), 4) for v in row] for row in scores]

    # Sanity report: top-5 heads by score, vs the known literature heads.
    flat = [(float(scores[l, h]), l, h)
            for l in range(scores.shape[0]) for h in range(scores.shape[1])]
    flat.sort(reverse=True)
    print("  top-5 induction heads (score, layer, head):")
    for s, l, h in flat[:5]:
        known = " [known]" if (l, h) in KNOWN_INDUCTION_HEADS else ""
        print(f"    L{l}H{h}: {s:.4f}{known}")
    print(f"  known-head scores: " + ", ".join(
        f"L{l}H{h}={float(scores[l,h]):.3f}" for l, h in KNOWN_INDUCTION_HEADS))

    print("running IOI activation patching (resid_pre, layer x position)")
    patching = ioi_patching(model)
    print(f"  logit_diff clean={patching['logit_diff_clean']} "
          f"corrupt={patching['logit_diff_corrupt']}")
    # report peak recovery cell
    peak_val, peak_l, peak_p = max(
        (patching["heatmap"][l][p], l, p)
        for l in range(len(patching["heatmap"]))
        for p in range(len(patching["heatmap"][0]))
    )
    print(f"  peak recovery {peak_val:.3f} at L{peak_l} pos{peak_p} "
          f"({patching['tokens'][peak_p]!r})")

    out = {"induction": {"scores": scores_list}, "patching": patching}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
