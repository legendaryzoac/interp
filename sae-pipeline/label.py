#!/usr/bin/env python
"""S5 — Haiku auto-labeling for the SAE Feature Explorer.

For each curated SAE feature, show Claude Haiku the feature's strongest
activating examples (peak token marked) plus the logit-lens promoted tokens,
and ask for a concise (<=8-word) concept label + a confidence (high/med/low).

Route: Amazon Bedrock `converse` (model = US cross-region inference profile
`us.anthropic.claude-haiku-4-5-20251001-v1:0`) via ambient AWS credentials.
Fallback (only if Bedrock is unreachable): direct Anthropic API.

Resumable (disk cache keyed by feature id), `--limit N`, `--force`.

The script has two phases:
  1. label   — call Haiku for every curated feature, cache each result.
  2. write   — merge cached labels into features_XXXX.json + index.json.
By default it runs both; use --no-write to only (re)label, or --write-only to
only merge an existing cache.

Nothing here mutates model-pipeline or the content_hash stamper. It only edits
the dashboard JSONs under D:/dev/sae-artifacts/L8/dashboards/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
DASH = Path("D:/dev/sae-artifacts/L8/dashboards")
CHUNK_FILES = ["features_0000.json", "features_0001.json"]
INDEX_FILE = "index.json"
CACHE_PATH = Path("D:/dev/sae-artifacts/L8/labels_cache.json")  # keep on D:

MODEL_BEDROCK = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
MODEL_ANTHROPIC = "claude-haiku-4-5-20251001"
REGION = "us-east-1"

# Haiku 4.5 published pricing (USD per 1M tokens). Reported, not billed here.
PRICE_IN_PER_MTOK = 1.00
PRICE_OUT_PER_MTOK = 5.00

N_EXAMPLES = 10          # top-N activating examples shown per feature
SECONDARY_FRAC = 0.50    # mark non-peak tokens whose act >= this fraction of max
MAX_WORKERS = 4          # be gentle: a few in flight
MAX_TOKENS_OUT = 120
TEMPERATURE = 0.0
VALID_CONF = {"high", "medium", "low"}

SYSTEM_PROMPT = (
    "You are an interpretability assistant that labels features of a sparse "
    "autoencoder (SAE) trained on the residual stream of GPT-2 small. Each "
    "feature fires on particular tokens in text. You are shown the feature's "
    "strongest activating text snippets. In every snippet, the single token "
    "where the feature fires MOST strongly is wrapped in «double angle "
    "brackets», and other strongly-activating tokens are wrapped in "
    "‹single angle brackets›. The surrounding words are context that "
    "tells you WHY the token fires.\n\n"
    "Your job: identify the ONE concept, pattern, or category the feature "
    "detects. Focus on what the «marked» tokens have in common. It may "
    "be a topic (e.g. legal proceedings), a syntactic/lexical pattern (e.g. "
    "closing brackets in citations), a script/language (e.g. Arabic "
    "characters), or a morphological pattern (e.g. hyphenated compound "
    "modifiers).\n\n"
    "Respond with STRICT JSON and nothing else — no markdown, no code "
    "fences, no commentary:\n"
    "{\"label\": \"<concise phrase, AT MOST 8 words, naming what the feature "
    "detects>\", \"confidence\": \"high|medium|low\"}\n\n"
    "confidence rules: \"high\" = the marked tokens share a crisp, consistent "
    "common thread; \"medium\" = a plausible theme with some noise; \"low\" = "
    "the examples look polysemantic or share no clear common thread. Do not "
    "invent a theme that is not supported — use \"low\" instead."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _visible(s: str) -> str:
    """Make whitespace tokens visible & keep each snippet on one line."""
    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def render_example(ex: dict) -> str:
    tokens = ex["tokens"]
    acts = ex["acts"]
    peak = ex.get("act_index", 0)
    ex_max = ex.get("max_act", 0.0) or 0.0
    peak_val = (acts[peak] / 255.0) * ex_max if 0 <= peak < len(acts) else 0.0
    thresh = 255 * SECONDARY_FRAC  # compare on the raw uint8 scale
    out = []
    for i, tok in enumerate(tokens):
        if i == peak:
            out.append(f"«{tok}»")
        elif i != peak and acts[i] >= thresh and acts[i] > 0:
            out.append(f"‹{tok}›")
        else:
            out.append(tok)
    return _visible("".join(out)), peak_val


def build_user_prompt(feat: dict) -> str:
    exs = feat.get("top_examples", [])[:N_EXAMPLES]
    lines = [
        f"SAE feature #{feat['id']}. Its top {len(exs)} activating snippets "
        f"(peak activation value shown in brackets):",
        "",
    ]
    for i, ex in enumerate(exs, 1):
        text, peak_val = render_example(ex)
        lines.append(f"{i}. [{peak_val:.1f}] {text}")
    promoted = [t for t, _ in feat.get("logit_lens", {}).get("promoted", [])][:10]
    if promoted:
        lines.append("")
        lines.append(
            "Tokens this feature most pushes the model to predict NEXT "
            "(logit lens, secondary signal): " + ", ".join(repr(t) for t in promoted)
        )
    lines.append("")
    lines.append("Return the JSON label now.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_label(text: str):
    """Return (label:str|None, confidence:str|None). None,None if unparseable."""
    if not text:
        return None, None
    s = text.strip()
    s = _FENCE.sub("", s).strip()
    # extract the first {...} block
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1 and b > a:
        s = s[a : b + 1]
    try:
        obj = json.loads(s)
    except Exception:
        return None, None
    if not isinstance(obj, dict):
        return None, None
    label = obj.get("label")
    conf = obj.get("confidence")
    if isinstance(conf, str):
        conf = conf.strip().lower()
    if conf not in VALID_CONF:
        return None, None
    if label is not None and not isinstance(label, str):
        label = str(label)
    if isinstance(label, str):
        label = label.strip()
        if not label:
            label = None
    return label, conf


# ---------------------------------------------------------------------------
# Bedrock call with backoff
# ---------------------------------------------------------------------------
class LabelClient:
    def __init__(self, use_anthropic=False):
        self.use_anthropic = use_anthropic
        if use_anthropic:
            import anthropic  # noqa
            self._ac = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model = MODEL_ANTHROPIC
        else:
            import boto3
            self._br = boto3.client("bedrock-runtime", region_name=REGION)
            self.model = MODEL_BEDROCK

    def invoke(self, system: str, user: str):
        """Return (text, in_tok, out_tok). Retries on throttling."""
        delay = 2.0
        last_err = None
        for attempt in range(6):
            try:
                if self.use_anthropic:
                    r = self._ac.messages.create(
                        model=self.model,
                        max_tokens=MAX_TOKENS_OUT,
                        temperature=TEMPERATURE,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    text = "".join(b.text for b in r.content if b.type == "text")
                    return text, r.usage.input_tokens, r.usage.output_tokens
                else:
                    r = self._br.converse(
                        modelId=self.model,
                        system=[{"text": system}],
                        messages=[{"role": "user", "content": [{"text": user}]}],
                        inferenceConfig={
                            "maxTokens": MAX_TOKENS_OUT,
                            "temperature": TEMPERATURE,
                        },
                    )
                    text = r["output"]["message"]["content"][0]["text"]
                    u = r["usage"]
                    return text, u["inputTokens"], u["outputTokens"]
            except Exception as e:  # noqa
                name = type(e).__name__
                msg = str(e)
                throttle = (
                    "Throttl" in name
                    or "TooManyRequests" in msg
                    or "ThrottlingException" in msg
                    or "ServiceUnavailable" in name
                    or "InternalServerException" in name
                )
                last_err = e
                if throttle and attempt < 5:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise
        raise last_err


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------
def load_curated():
    """Return list of (feature_dict, chunk_index) in file order."""
    feats = []
    for ci, fn in enumerate(CHUNK_FILES):
        arr = json.loads((DASH / fn).read_text(encoding="utf-8"))
        for f in arr:
            feats.append((f, ci))
    return feats


# ---------------------------------------------------------------------------
# Label phase
# ---------------------------------------------------------------------------
def label_phase(args, client: LabelClient):
    feats = load_curated()
    if args.limit:
        feats = feats[: args.limit]
    print(f"[label] {len(feats)} curated features to consider "
          f"(model={client.model})", flush=True)

    cache = {}
    if CACHE_PATH.exists() and not args.force:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        print(f"[label] loaded cache with {len(cache)} entries "
              f"from {CACHE_PATH}", flush=True)

    lock = threading.Lock()
    totals = {"in": 0, "out": 0, "calls": 0, "null": 0, "done": 0}

    def flush_cache():
        tmp = CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=0),
                       encoding="utf-8")
        tmp.replace(CACHE_PATH)

    todo = [(f, ci) for (f, ci) in feats if (str(f["id"]) not in cache or args.force)]
    print(f"[label] {len(feats) - len(todo)} already cached, "
          f"{len(todo)} to label", flush=True)

    def work(feat):
        system = SYSTEM_PROMPT
        user = build_user_prompt(feat)
        attempts = 0
        label = conf = None
        in_tok = out_tok = 0
        raw_last = ""
        for attempt in range(2):  # one retry on unparseable
            attempts += 1
            text, it, ot = client.invoke(system, user)
            in_tok += it
            out_tok += ot
            raw_last = text
            label, conf = parse_label(text)
            if conf is not None:
                break
        if conf is None:
            conf = "low"
            label = None
        return {
            "id": feat["id"],
            "label": label,
            "confidence": conf,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "attempts": attempts,
            "parsed": label is not None or conf in VALID_CONF,
            "unparseable": (label is None and attempts >= 2),
            "raw_last": raw_last[:300],
        }

    fmap = {f["id"]: f for (f, _) in feats}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, f): f["id"] for (f, _) in todo}
        for fut in as_completed(futs):
            fid = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa
                print(f"[label] feature {fid} FAILED hard: "
                      f"{type(e).__name__}: {e}", flush=True)
                res = {
                    "id": fid, "label": None, "confidence": "low",
                    "input_tokens": 0, "output_tokens": 0, "attempts": 0,
                    "parsed": False, "unparseable": True, "raw_last": f"ERROR:{e}",
                }
            with lock:
                cache[str(fid)] = res
                totals["in"] += res["input_tokens"]
                totals["out"] += res["output_tokens"]
                totals["calls"] += res["attempts"]
                totals["done"] += 1
                if res["label"] is None:
                    totals["null"] += 1
                    print(f"[label] NULL/low for feature {fid} "
                          f"(raw={res['raw_last']!r})", flush=True)
                if totals["done"] % 25 == 0 or totals["done"] == len(todo):
                    flush_cache()
                    print(f"[label] {totals['done']}/{len(todo)} done "
                          f"(in={totals['in']} out={totals['out']} tok)",
                          flush=True)
    flush_cache()
    cost = (totals["in"] / 1e6) * PRICE_IN_PER_MTOK + \
           (totals["out"] / 1e6) * PRICE_OUT_PER_MTOK
    print(f"[label] complete. this run: {totals['done']} labeled, "
          f"{totals['null']} null, {totals['calls']} API calls, "
          f"in={totals['in']} out={totals['out']} tok, "
          f"est ${cost:.4f}", flush=True)
    return cache, totals, cost


# ---------------------------------------------------------------------------
# Print sample (for --limit sanity check)
# ---------------------------------------------------------------------------
def print_sample(feats, cache, n):
    print("\n" + "=" * 78)
    print(f"SAMPLE OF {n} LABELS (with top-3 example snippets)")
    print("=" * 78)
    for (f, _) in feats[:n]:
        c = cache.get(str(f["id"]), {})
        print(f"\nfeature {f['id']}  ({f.get('selection_reason','?')})  "
              f"freq={f.get('freq'):.5f}")
        print(f"  LABEL: {c.get('label')!r}   confidence={c.get('confidence')!r}")
        for ex in f.get("top_examples", [])[:3]:
            text, pv = render_example(ex)
            print(f"    [{pv:.1f}] {text}")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Write-back phase
# ---------------------------------------------------------------------------
def write_phase(cache):
    """Merge cached labels into features_XXXX.json + index.json (compact)."""
    # index.json
    idx_path = DASH / INDEX_FILE
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    n_idx = 0
    for fid_str, entry in idx["features"].items():
        if entry.get("hasDashboard"):
            c = cache.get(fid_str)
            if c is None:
                continue
            entry["label"] = c["label"]
            entry["label_confidence"] = c["confidence"]
            n_idx += 1
    tmp = idx_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, ensure_ascii=True, separators=(",", ":")),
                   encoding="utf-8")
    tmp.replace(idx_path)
    print(f"[write] index.json: added label to {n_idx} hasDashboard entries",
          flush=True)

    # chunk files
    n_chunk = 0
    for fn in CHUNK_FILES:
        p = DASH / fn
        arr = json.loads(p.read_text(encoding="utf-8"))
        for f in arr:
            c = cache.get(str(f["id"]))
            if c is None:
                continue
            f["label"] = c["label"]
            f["label_confidence"] = c["confidence"]
            n_chunk += 1
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(arr, ensure_ascii=True, separators=(",", ":")),
                       encoding="utf-8")
        tmp.replace(p)
        print(f"[write] {fn}: {len(arr)} features updated", flush=True)
    print(f"[write] total chunk features labeled: {n_chunk}", flush=True)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify():
    idx = json.loads((DASH / INDEX_FILE).read_text(encoding="utf-8"))
    dash_entries = {k: v for k, v in idx["features"].items() if v.get("hasDashboard")}
    n_total = len(dash_entries)
    missing_conf = [k for k, v in dash_entries.items()
                    if v.get("label_confidence") not in VALID_CONF]
    null_label = [k for k, v in dash_entries.items() if v.get("label") is None]
    # cross-check chunk files vs index for 20 sampled ids
    chunk_label = {}
    for fn in CHUNK_FILES:
        for f in json.loads((DASH / fn).read_text(encoding="utf-8")):
            chunk_label[str(f["id"])] = (f.get("label"), f.get("label_confidence"))
    import random
    sample = random.Random(0).sample(list(dash_entries.keys()),
                                      min(20, len(dash_entries)))
    mism = []
    for k in sample:
        ie = dash_entries[k]
        cl = chunk_label.get(k, (None, "MISSING"))
        if (ie.get("label"), ie.get("label_confidence")) != cl:
            mism.append(k)
    conf_dist = {"high": 0, "medium": 0, "low": 0}
    for v in dash_entries.values():
        c = v.get("label_confidence")
        if c in conf_dist:
            conf_dist[c] += 1
    return {
        "n_dashboard": n_total,
        "n_missing_confidence": len(missing_conf),
        "missing_confidence_ids": missing_conf[:20],
        "n_null_label": len(null_label),
        "null_label_ids": null_label,
        "n_chunk_features": len(chunk_label),
        "sample_size": len(sample),
        "sample_mismatches": mism,
        "confidence_distribution": conf_dist,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="label only the first N curated features")
    ap.add_argument("--force", action="store_true",
                    help="ignore cache; re-label everything")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--no-write", action="store_true",
                    help="only (re)label; skip write-back")
    ap.add_argument("--write-only", action="store_true",
                    help="skip labeling; merge existing cache into JSONs")
    ap.add_argument("--sample", type=int, default=0,
                    help="print this many labels+snippets after labeling")
    ap.add_argument("--anthropic", action="store_true",
                    help="force the direct Anthropic API fallback")
    ap.add_argument("--verify", action="store_true",
                    help="run verification only and print JSON")
    args = ap.parse_args()

    if args.verify:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return

    if args.write_only:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        write_phase(cache)
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return

    client = LabelClient(use_anthropic=args.anthropic)
    cache, totals, cost = label_phase(args, client)

    feats = load_curated()
    if args.limit:
        feats = feats[: args.limit]
    if args.sample:
        print_sample(feats, cache, args.sample)

    if not args.no_write and not args.limit:
        write_phase(cache)
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
    elif args.limit:
        print("[main] --limit set: skipping write-back (sanity run).", flush=True)


if __name__ == "__main__":
    main()
