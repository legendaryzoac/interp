"""Optimize real GCG adversarial suffixes against GPT-2, on local hardware.

WHY this exists: the interpretability web app has an "Adversarial" tab that shows
suffixes which steer GPT-2's next-token behavior. The whole point of the demo is
that these are *not* copied from a paper -- they were GCG-optimized against the
EXACT gpt2 (124M) checkpoint that the ONNX export is built from, on my own box
(on CPU -- torch here is CPU-only; see note below). Provenance therefore
genuinely reads "gcg".

GCG (Greedy Coordinate Gradient, Zou et al. 2023) picks, at each step, token
substitutions in the suffix that most reduce the cross-entropy of a chosen target
continuation. nanoGCG is the reference minimal implementation.

GPT-2 is a *base* LM, not instruction-tuned, so there is no "refusal" to jailbreak.
The meaningful adversarial objective for a base model is next-token steering:
force a benign prompt to continue with a token/phrase the model would otherwise
consider very unlikely. Each objective below does exactly that.

Environment notes:
  - Run from the ISOLATED gcg venv (D:/dev/gcg-venv), NOT the parity venv
    (D:/dev/interp-venv). nanoGCG pins an older transformers; keeping it out of
    the parity venv protects the ONNX-parity version lock.
  - torch is CPU here. GCG on a 124M model is small: ~2.4 s/step on this box, so
    ~120 steps/suffix is a few minutes. GPU (a cu-torch build) would be faster but
    is not needed for a handful of portfolio suffixes, and a torch reinstall would
    risk the pipeline. CPU is the deliberate, low-risk choice.
  - nanoGCG renders the prompt via tokenizer.apply_chat_template. GPT-2 ships no
    chat template, so we install a verbatim passthrough template (concatenate
    message contents, no roles, no special tokens) -- the base-LM prompt and the
    appended optim suffix reach the model exactly as written.

Output: out/gcg_suffixes.json, schema matching web/public/adversarial.json:
    { "suffixes": [ { "id", "text", "provenance", "note", "target" } ] }
Provenance is "gcg" for every entry (all genuinely optimized here). Real run
metadata (steps, final loss, wall-time) is printed and also stored in a sidecar
report file so the numbers in the write-up are auditable.

Usage:
    # from the gcg venv:
    D:/dev/gcg-venv/Scripts/python.exe gcg_suffixes.py
    # tuning knobs:
    D:/dev/gcg-venv/Scripts/python.exe gcg_suffixes.py --steps 120 --search-width 256
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nanogcg
from nanogcg import GCGConfig

HERE = Path(__file__).parent
MODEL_ID = "gpt2"

# Verbatim passthrough: a base LM must see prompt+suffix with no chat scaffolding.
PASSTHROUGH_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"

# The objectives. Each is a benign prompt whose natural continuation is something
# ordinary; the target is a deliberately *unlikely* continuation we force GCG to
# elicit. Keeping targets to one or two tokens keeps CPU runs short and the
# optimization well-posed for a 124M model. Leading spaces matter: GPT-2 BPE
# encodes " word" (with the space) as the natural mid-sentence token.
OBJECTIVES = [
    {
        "id": "gcg-weather-catastrophic",
        "prompt": "The weather today is",
        "target": " catastrophic",
        "note": (
            "GCG-optimized on CPU (my local GTX 1070 box; torch was CPU-only) "
            "against this exact gpt2 checkpoint. Forces the benign prompt 'The "
            "weather today is' to continue with the alarmist ' catastrophic' "
            "instead of a mild weather word."
        ),
    },
    {
        "id": "gcg-capital-paris-berlin",
        "prompt": "The capital of France is",
        "target": " Berlin",
        "note": (
            "GCG-optimized on CPU (my local GTX 1070 box; torch was CPU-only) "
            "against this exact gpt2 checkpoint. Overrides a factual completion: "
            "pushes 'The capital of France is' toward the wrong answer ' Berlin' "
            "over ' Paris'."
        ),
    },
    {
        "id": "gcg-review-terrible",
        "prompt": "I loved this movie, it was absolutely",
        "target": " terrible",
        "note": (
            "GCG-optimized on CPU (my local GTX 1070 box; torch was CPU-only) "
            "against this exact gpt2 checkpoint. Flips sentiment: a clearly "
            "positive prompt is steered to continue with the negative ' terrible'."
        ),
    },
    {
        "id": "gcg-two-plus-two",
        "prompt": "Two plus two equals",
        "target": " five",
        "note": (
            "GCG-optimized on CPU (my local GTX 1070 box; torch was CPU-only) "
            "against this exact gpt2 checkpoint. An Orwellian arithmetic override: "
            "forces 'Two plus two equals' to continue ' five'."
        ),
    },
]


def load_model_and_tokenizer():
    """Load gpt2 in fp32 with the verbatim passthrough chat template installed."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.chat_template = PASSTHROUGH_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    return model, tok


def continuation_after(model, tok, prompt, suffix, max_new_tokens=6):
    """Greedy-decode the continuation of prompt+suffix, for a qualitative check.

    This is NOT the optimization signal (GCG optimizes target cross-entropy); it
    just lets the report show what the model actually says after the attack.
    """
    text = prompt + suffix
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120,
                        help="GCG num_steps per suffix (short by design)")
    parser.add_argument("--search-width", type=int, default=256,
                        help="candidate substitutions evaluated per step")
    parser.add_argument("--topk", type=int, default=128,
                        help="top-k gradient coordinates considered per position")
    parser.add_argument("--seed", type=int, default=0,
                        help="fixed seed for reproducible suffixes")
    parser.add_argument("--out", default=str(HERE / "out" / "gcg_suffixes.json"))
    args = parser.parse_args()

    print(f"loading {MODEL_ID} (fp32, CPU) + passthrough chat template")
    model, tok = load_model_and_tokenizer()

    cfg = GCGConfig(
        num_steps=args.steps,
        search_width=args.search_width,
        topk=args.topk,
        seed=args.seed,
        # Base-model suffixes are only interesting if a human could type them, so
        # keep the search on printable ASCII.
        allow_non_ascii=False,
        verbosity="WARNING",
    )

    suffixes, report_rows = [], []
    for obj in OBJECTIVES:
        print(f"\n=== {obj['id']} ===")
        print(f"    prompt : {obj['prompt']!r}")
        print(f"    target : {obj['target']!r}")
        t0 = time.time()
        res = nanogcg.run(model, tok, obj["prompt"], obj["target"], cfg)
        dt = time.time() - t0

        cont = continuation_after(model, tok, obj["prompt"], res.best_string)
        print(f"    steps={len(res.losses)} final_loss={res.best_loss:.4f} "
              f"walltime={dt:.1f}s")
        print(f"    suffix : {res.best_string!r}")
        print(f"    greedy continuation now: {cont!r}")

        suffixes.append({
            "id": obj["id"],
            "text": res.best_string,
            "provenance": "gcg",
            "note": obj["note"],
            "target": obj["target"],
        })
        report_rows.append({
            "id": obj["id"],
            "prompt": obj["prompt"],
            "target": obj["target"],
            "steps": len(res.losses),
            "final_loss": round(float(res.best_loss), 4),
            "walltime_s": round(dt, 1),
            "suffix": res.best_string,
            "greedy_continuation": cont,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"suffixes": suffixes}, indent=2),
                        encoding="utf-8")
    print(f"\nwrote {out_path} ({len(suffixes)} suffixes)")

    # Auditable sidecar: exact config + per-suffix real metadata.
    report = {
        "model_id": MODEL_ID,
        "device": "cpu",
        "torch": torch.__version__,
        "host": platform.node(),
        "config": {
            "num_steps": args.steps,
            "search_width": args.search_width,
            "topk": args.topk,
            "seed": args.seed,
            "allow_non_ascii": False,
        },
        "runs": report_rows,
    }
    reports_dir = HERE / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "gcg_suffixes_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
