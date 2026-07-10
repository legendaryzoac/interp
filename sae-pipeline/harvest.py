"""S4 - Bounded activation harvest -> curated feature set.

Streams OpenWebText, runs GPT-2-small, extracts layer-L ``resid_pre`` the
``verify_basis`` way (raw-HF residual, proven == the browser's basis), applies
``basis_transform.to_sae_input`` (mean-center over d_model), encodes with the
``gpt2-small-res-jb`` SAE, and accumulates *memory-bounded* per-feature statistics
over the whole stream:

  * activation frequency        freq_count[f] / total_nonbos_tokens
  * max activation              max_act[f]
  * histogram of nonzero acts   shared log-spaced bins (documented below)
  * top-K activating snippets   per-feature min-heap (K=16), context window +-16
                                stored as token IDS + per-token acts (decoded to
                                strings only at emit time, for the curated set)

Then it selects a bounded curated set (~300-500 features) = top-frequency alive
features UNION a diversity pass across frequency bands, and emits per-feature
dashboard JSON (chunked) + an index.json over ALL 24576 features.

Stages (run all by default; each is resumable):
  tokenize : stream + BPE-tokenize + pack into [n_seq, seq_len] uint16 on D:.
  harvest  : forward+encode every sequence, update stats, checkpoint periodically.
  emit     : curated selection + per-feature JSON + index.json + logit lens + report.

Design notes that matter
------------------------
* **BOS / position 0 is excluded** from every statistic (freq, max, histogram,
  snippets). It is the attention-sink token: its SAE activations are huge and
  degenerate (the |feats|~500 seen in S3 parity is the sink), so counting it
  would poison feature frequencies and top-snippet heaps. BOS is still PREPENDED
  to every 128-token context (the SAE was trained with prepend_bos=True), it is
  just never itself an "activating token" and never enters the denominators.
* **Histogram binning is single-pass** using bins that are the SAME for every
  feature: log-spaced edges over [HIST_LO, HIST_HI] with an under/overflow clamp,
  plus the exact per-feature max stored separately. This avoids a second forward
  pass (a true [0, per-feature-max] histogram would need to re-encode every token
  after the max is known). The shared log axis is standard for SAE dashboards and
  is written into every feature's JSON so the browser needs no side channel.
* **Top-K heaps store token IDs, not strings.** The task text suggests decoding to
  strings during the stream; storing uint16 IDs + fp16 per-token acts instead is
  strictly smaller (~50 MB for all 24576 heaps) and we decode to strings only for
  the few hundred curated features at emit time. Same K-best memory bound.

Run:
    export HF_HOME=D:/dev/hf-cache
    D:/dev/sae-venv/Scripts/python.exe harvest.py --tokens 2e6

See sae-pipeline/README.md for the full command set.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

RELEASE = "gpt2-small-res-jb"
D_MODEL = 768
D_SAE = 24576
BOS_ID = 50256  # GPT-2 <|endoftext|> == bos == eos

# Shared log-spaced histogram edges (20 bins). Nonzero acts are clamped into
# [HIST_LO, HIST_HI] before binning; the exact per-feature max is stored too.
HIST_LO = 0.05
HIST_HI = 512.0
HIST_NBINS = 20
HIST_EDGES = np.logspace(np.log10(HIST_LO), np.log10(HIST_HI), HIST_NBINS + 1).astype(np.float64)


# ---------------------------------------------------------------------------
# Stage 1: tokenize -> packed [n_seq, seq_len] uint16 on D:
# ---------------------------------------------------------------------------
def stage_tokenize(args, tok) -> Path:
    """Stream the dataset, BPE-tokenize, pack into fixed-length contexts.

    Each emitted context is [BOS] + 127 document tokens (seq_len=128). Documents
    are concatenated into a rolling buffer and chunked (SAELens-style packing);
    BOS is prepended per context to match the SAE's prepend_bos=True training.
    """
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    tok_path = work / "tokens_u16.npy"
    meta_path = work / "tokens_meta.json"

    target_nonbos = int(float(args.tokens))
    per_seq = args.seq_len - 1  # non-BOS tokens per context
    n_seq_target = -(-target_nonbos // per_seq)  # ceil

    if tok_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("n_seq", 0) >= n_seq_target and meta.get("seq_len") == args.seq_len \
                and meta.get("dataset") == args.dataset:
            print(f"[tokenize] reuse {tok_path.name} ({meta['n_seq']} seq, "
                  f"{meta['n_seq']*per_seq} non-BOS tok) >= target {n_seq_target}")
            return tok_path
        print(f"[tokenize] existing token file insufficient/mismatched -> re-tokenizing")

    from datasets import load_dataset
    print(f"[tokenize] streaming {args.dataset} for >= {target_nonbos:,} non-BOS tokens "
          f"({n_seq_target:,} contexts of {args.seq_len})")
    ds = load_dataset(args.dataset, split="train", streaming=True)

    seqs = np.empty((n_seq_target, args.seq_len), dtype=np.uint16)
    buf: list[int] = []
    n_seq = 0
    n_docs = 0
    t0 = time.time()
    for rec in ds:
        text = rec.get("text") or ""
        if not text.strip():
            continue
        ids = tok.encode(text)
        if not ids:
            continue
        n_docs += 1
        buf.extend(ids)
        while len(buf) >= per_seq and n_seq < n_seq_target:
            ctx = buf[:per_seq]
            del buf[:per_seq]
            seqs[n_seq, 0] = BOS_ID
            seqs[n_seq, 1:] = np.asarray(ctx, dtype=np.uint16)
            n_seq += 1
            if n_seq % 2000 == 0:
                rate = n_seq * per_seq / max(time.time() - t0, 1e-6)
                print(f"[tokenize]   {n_seq}/{n_seq_target} contexts  "
                      f"({n_seq*per_seq:,} tok)  {rate:,.0f} tok/s  docs={n_docs}")
        if n_seq >= n_seq_target:
            break

    if n_seq < n_seq_target:
        print(f"[tokenize] dataset exhausted at {n_seq} contexts (< {n_seq_target}); using what we have")
        seqs = seqs[:n_seq]

    np.save(tok_path, seqs)
    meta = {
        "dataset": args.dataset,
        "seq_len": args.seq_len,
        "n_seq": int(n_seq),
        "nonbos_per_seq": per_seq,
        "nonbos_tokens": int(n_seq * per_seq),
        "n_docs": int(n_docs),
        "bos_id": BOS_ID,
        "tokenize_seconds": round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[tokenize] wrote {tok_path} shape {seqs.shape}  "
          f"({meta['nonbos_tokens']:,} non-BOS tok from {n_docs} docs) in {meta['tokenize_seconds']}s")
    return tok_path


# ---------------------------------------------------------------------------
# Streaming per-feature state (all memory-bounded)
# ---------------------------------------------------------------------------
class HarvestState:
    def __init__(self, topk: int, ctx: int):
        self.topk = topk
        self.ctx = ctx
        self.freq_count = np.zeros(D_SAE, dtype=np.int64)
        self.max_act = np.zeros(D_SAE, dtype=np.float32)
        self.hist = np.zeros((D_SAE, HIST_NBINS), dtype=np.int64)
        # per-feature min-heaps of (act, uid, win_ids[uint16], win_acts[fp16], off)
        self.heaps: list[list] = [[] for _ in range(D_SAE)]
        # thresh[f] = current K-th best act if heap full else 0.0 -> candidate iff act > thresh
        self.thresh = np.zeros(D_SAE, dtype=np.float32)
        self.uid = 0
        self.total_nonbos = 0
        self.next_seq = 0

    def update_chunk(self, feats: np.ndarray, ids: np.ndarray):
        """feats: [b, L, D_SAE] float32 (BOS at pos 0 included).
        ids:   [b, L]  uint16 token ids for this chunk."""
        b, L, _ = feats.shape
        f = feats[:, 1:, :]                       # drop BOS position from stats
        active = f > 0.0
        # frequency + max
        self.freq_count += active.reshape(-1, D_SAE).sum(axis=0).astype(np.int64)
        np.maximum(self.max_act, f.max(axis=(0, 1)), out=self.max_act)
        # histogram of nonzero acts (shared log bins), vectorized via bincount
        rr, pp, cc = np.nonzero(active)
        if cc.size:
            vals = f[rr, pp, cc]
            clamped = np.clip(vals, HIST_LO, HIST_HI)
            binidx = np.clip(np.searchsorted(HIST_EDGES, clamped, side="right") - 1,
                             0, HIST_NBINS - 1)
            flat = cc.astype(np.int64) * HIST_NBINS + binidx
            self.hist += np.bincount(flat, minlength=D_SAE * HIST_NBINS).reshape(D_SAE, HIST_NBINS)
        self.total_nonbos += b * (L - 1)

        # top-K snippet heaps: candidate iff act > per-feature threshold
        cand = f > self.thresh[None, None, :]
        cr, cp, cc2 = np.nonzero(cand)
        if cr.size:
            cvals = f[cr, cp, cc2]
            pos_full = cp + 1                     # position within the full seq (BOS at 0)
            touched = set()
            K = self.topk
            heaps = self.heaps
            CTX = self.ctx
            for i in range(cr.size):
                bi = cr[i]; pf = pos_full[i]; fe = int(cc2[i]); av = float(cvals[i])
                h = heaps[fe]
                if len(h) >= K and av <= h[0][0]:
                    continue
                lo = pf - CTX if pf - CTX > 0 else 0
                hi = pf + CTX + 1 if pf + CTX + 1 < L else L
                win_ids = ids[bi, lo:hi].astype(np.uint16).copy()
                win_acts = feats[bi, lo:hi, fe].astype(np.float16).copy()
                off = pf - lo
                entry = (av, self.uid, win_ids, win_acts, off)
                self.uid += 1
                if len(h) < K:
                    heapq.heappush(h, entry)
                else:
                    heapq.heappushpop(h, entry)
                touched.add(fe)
            for fe in touched:
                h = heaps[fe]
                self.thresh[fe] = h[0][0] if len(h) >= K else 0.0


# ---------------------------------------------------------------------------
# Stage 2: harvest (forward + encode + accumulate), resumable
# ---------------------------------------------------------------------------
def stage_harvest(args, tok_path: Path) -> "HarvestState":
    import torch
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    torch.set_grad_enabled(False)
    device = args.device
    resid_hook = f"blocks.{args.layer - 1}.hook_resid_post"  # == blocks.{layer}.hook_resid_pre

    seqs = np.load(tok_path)                       # [n_seq, seq_len] uint16
    n_seq, L = seqs.shape
    ckpt_path = Path(args.work) / "harvest_ckpt.pkl"

    st: HarvestState | None = None
    if args.resume and ckpt_path.exists():
        with ckpt_path.open("rb") as fh:
            saved = pickle.load(fh)
        if saved.get("n_seq") == n_seq and saved.get("layer") == args.layer \
                and saved.get("topk") == args.topk and saved.get("ctx") == args.ctx:
            st = saved["state"]
            print(f"[harvest] resumed from checkpoint at seq {st.next_seq}/{n_seq}")
        else:
            print("[harvest] checkpoint params mismatch -> starting fresh")
    if st is None:
        st = HarvestState(args.topk, args.ctx)

    if st.next_seq >= n_seq:
        print("[harvest] already complete")
        return st

    print(f"[harvest] loading raw-HF GPT-2 + SAE ({resid_hook}) on {device}")
    model = HookedTransformer.from_pretrained_no_processing("gpt2").to(device).eval()
    loaded = SAE.from_pretrained(release=RELEASE, sae_id=f"blocks.{args.layer}.hook_resid_pre", device=device)
    sae = (loaded[0] if isinstance(loaded, tuple) else loaded).to(device).eval()

    cs = args.chunk_seqs
    t0 = time.time()
    chunks_done = 0
    start_seq = st.next_seq
    for s in range(st.next_seq, n_seq, cs):
        e = min(s + cs, n_seq)
        ids_np = seqs[s:e].astype(np.int64)
        ids_t = torch.from_numpy(ids_np).to(device)
        _, cache = model.run_with_cache(ids_t, stop_at_layer=args.layer, names_filter=resid_hook)
        resid = cache[resid_hook]                              # [b, L, 768]
        x = resid - resid.mean(dim=-1, keepdim=True)           # to_sae_input (center)
        feats_t = sae.encode(x.reshape(-1, D_MODEL)).reshape(e - s, L, D_SAE)
        feats = feats_t.detach().to(torch.float32).cpu().numpy()
        del feats_t, resid, x, cache
        st.update_chunk(feats, seqs[s:e])
        st.next_seq = e
        del feats
        chunks_done += 1
        if chunks_done % args.log_every == 0:
            done = e - start_seq
            rate = done * L / max(time.time() - t0, 1e-6)
            eta = (n_seq - e) * L / max(rate, 1e-6)
            print(f"[harvest]   seq {e}/{n_seq}  {e*L:,} tok  {rate:,.0f} tok/s  "
                  f"ETA {eta/60:.1f} min  alive={int((st.freq_count>0).sum())}", flush=True)
        if chunks_done % args.checkpoint_every == 0:
            _save_ckpt(ckpt_path, st, n_seq, args)

    _save_ckpt(ckpt_path, st, n_seq, args)
    wall = time.time() - t0
    st.harvest_seconds = wall  # type: ignore[attr-defined]
    st.tok_per_sec = (n_seq - start_seq) * L / max(wall, 1e-6)  # type: ignore[attr-defined]
    print(f"[harvest] done: {n_seq} seq, {st.total_nonbos:,} non-BOS tok in {wall/60:.1f} min "
          f"({st.tok_per_sec:,.0f} tok/s incl BOS)")
    return st


def _save_ckpt(path: Path, st: HarvestState, n_seq: int, args):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"state": st, "n_seq": n_seq, "layer": args.layer,
                     "topk": args.topk, "ctx": args.ctx}, fh, protocol=4)
    tmp.replace(path)
    print(f"[harvest]   checkpoint @ seq {st.next_seq}/{n_seq} -> {path.name}", flush=True)


# ---------------------------------------------------------------------------
# Stage 3: curated selection + emit dashboards + index + logit lens + report
# ---------------------------------------------------------------------------
def _snippet_token_diversity(heap: list) -> int:
    """Distinct activating-token ids among a feature's heap entries (proxy for how
    varied / conceptual vs single-token the feature is)."""
    toks = set()
    for (_av, _uid, win_ids, _wa, off) in heap:
        toks.add(int(win_ids[off]))
    return len(toks)


def select_curated(st: HarvestState, total_nonbos: int, target: int,
                   min_acts: int, max_freq: float, min_max_act: float) -> dict:
    freq = st.freq_count / max(total_nonbos, 1)
    alive = st.freq_count > 0
    n_alive = int(alive.sum())
    n_dead = D_SAE - n_alive

    pool_mask = (st.freq_count >= min_acts) & (freq <= max_freq) & (st.max_act >= min_max_act)
    pool = np.nonzero(pool_mask)[0]
    reason: dict[int, str] = {}

    # Set A: top-frequency features in the pool.
    n_top = min(target // 2, pool.size)
    top_by_freq = pool[np.argsort(-freq[pool])][:n_top]
    for f in top_by_freq:
        reason[int(f)] = "high_frequency"

    # Set B: diversity across log-frequency bands; within each band prefer features
    # whose top snippets fire on the most DISTINCT tokens (avoids shipping 400
    # near-duplicate high-frequency function-word features).
    remaining = np.array([f for f in pool if int(f) not in reason], dtype=np.int64)
    if remaining.size:
        lf = np.log10(freq[remaining])
        n_bands = 8
        edges = np.linspace(lf.min(), lf.max() + 1e-9, n_bands + 1)
        band = np.clip(np.digitize(lf, edges) - 1, 0, n_bands - 1)
        want_total = target - len(reason)
        per_band = max(1, -(-want_total // n_bands))
        for bi in range(n_bands):
            members = remaining[band == bi]
            if members.size == 0:
                continue
            div = np.array([_snippet_token_diversity(st.heaps[int(f)]) for f in members])
            order = members[np.argsort(-div)]
            for f in order[:per_band]:
                if len(reason) >= target:
                    break
                if int(f) not in reason:
                    reason[int(f)] = f"diversity_band{bi}"
            if len(reason) >= target:
                break

    curated = sorted(reason.keys())
    return {
        "curated": curated,
        "reason": reason,
        "freq": freq,
        "n_alive": n_alive,
        "n_dead": n_dead,
        "pool_size": int(pool.size),
        "thresholds": {
            "min_acts": min_acts, "min_freq": min_acts / max(total_nonbos, 1),
            "max_freq": max_freq, "min_max_act": min_max_act,
            "top_by_freq_count": int(n_top), "target": target,
        },
    }


def compute_logit_lens(curated: list[int], W_dec: np.ndarray, tok, top: int = 10) -> dict:
    """logit_lens[f] = top/bottom `top` tokens of W_dec[f] @ W_U, where W_U is from
    from_pretrained('gpt2') (fold_ln + center_unembed applied) -- the standard SAE
    logit lens (applies the ln_f fold approximation; documented in the report)."""
    import torch
    from transformer_lens import HookedTransformer
    print("[emit] loading default-processed gpt2 for W_U (logit lens) ...")
    proc = HookedTransformer.from_pretrained("gpt2").eval()
    W_U = proc.W_U.detach().cpu().numpy().astype(np.float32)      # [768, 50257]
    del proc
    out = {}
    W_curated = W_dec[np.asarray(curated)]                        # [n, 768]
    logits = W_curated @ W_U                                      # [n, 50257]
    for i, f in enumerate(curated):
        row = logits[i]
        top_idx = np.argsort(-row)[:top]
        bot_idx = np.argsort(row)[:top]
        out[int(f)] = {
            "promoted": [[tok.decode([int(j)]), round(float(row[j]), 4)] for j in top_idx],
            "suppressed": [[tok.decode([int(j)]), round(float(row[j]), 4)] for j in bot_idx],
        }
    return out


def _example_from_entry(entry, tok, max_tokens: int):
    av, _uid, win_ids, win_acts, off = entry
    ids = win_ids.astype(np.int64)
    acts = win_acts.astype(np.float32)
    # trim to <= max_tokens centered on the activating token
    if len(ids) > max_tokens:
        half_before = max_tokens // 2 - 1
        lo = max(0, off - half_before)
        hi = min(len(ids), lo + max_tokens)
        lo = max(0, hi - max_tokens)
        ids = ids[lo:hi]; acts = acts[lo:hi]; off = off - lo
    toks = [tok.decode([int(j)]) for j in ids]
    scale = float(max(acts.max(), 1e-6))
    q = np.clip(np.round(acts / scale * 255.0), 0, 255).astype(np.uint8)
    return {
        "tokens": toks,
        "acts": [int(v) for v in q],           # 8-bit quantized (act = acts/255 * max_act)
        "max_act": round(scale, 4),
        "act_index": int(off),                 # which token is the activating one
    }


def stage_emit(args, st: HarvestState):
    from sae_lens import SAE
    from transformers import GPT2TokenizerFast

    out_root = Path(args.out)
    dash = out_root / "dashboards"
    dash.mkdir(parents=True, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    total_nonbos = st.total_nonbos
    sel = select_curated(st, total_nonbos, args.curated_target,
                         args.min_acts, args.max_freq, args.min_max_act)
    curated = sel["curated"]
    freq = sel["freq"]
    print(f"[emit] curated {len(curated)} features "
          f"(alive={sel['n_alive']}, dead={sel['n_dead']}, pool={sel['pool_size']})")

    # decoder weights for logit lens
    loaded = SAE.from_pretrained(release=RELEASE, sae_id=f"blocks.{args.layer}.hook_resid_pre", device="cpu")
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    W_dec = sae.W_dec.detach().cpu().numpy().astype(np.float32)   # [24576, 768]
    logit_lens = compute_logit_lens(curated, W_dec, tok) if args.logit_lens else {}

    # ---- per-feature JSON, chunked ----
    edges_list = [round(float(x), 5) for x in HIST_EDGES]
    per_chunk = args.features_per_file
    chunk_of: dict[int, int] = {}
    n_chunks = -(-len(curated) // per_chunk)
    total_bytes = 0
    for ci in range(n_chunks):
        part = curated[ci * per_chunk:(ci + 1) * per_chunk]
        feats_json = []
        for f in part:
            chunk_of[f] = ci
            heap = sorted(st.heaps[f], key=lambda x: -x[0])[:args.examples_per_feature]
            examples = [_example_from_entry(en, tok, args.tokens_per_example) for en in heap]
            obj = {
                "id": int(f),
                "freq": float(f"{freq[f]:.6g}"),
                "max_act": round(float(st.max_act[f]), 4),
                "n_active": int(st.freq_count[f]),
                "selection_reason": sel["reason"][f],
                "histogram": {"bins": edges_list, "counts": [int(c) for c in st.hist[f]]},
                "top_examples": examples,
            }
            if f in logit_lens:
                obj["logit_lens"] = logit_lens[f]
            feats_json.append(obj)
        p = dash / f"features_{ci:04d}.json"
        p.write_text(json.dumps(feats_json, separators=(",", ":")))
        total_bytes += p.stat().st_size
        print(f"[emit]   wrote {p.name}  {len(part)} features  {p.stat().st_size/1024:.0f} KB")

    # ---- index.json over ALL 24576 features ----
    index = {}
    for f in range(D_SAE):
        index[str(f)] = {
            "freq": float(f"{freq[f]:.5g}"),
            "chunk": chunk_of.get(f),                     # int or null
            "hasDashboard": f in chunk_of,
        }
    idx_path = dash / "index.json"
    idx_path.write_text(json.dumps({
        "layer": args.layer,
        "sae_release": RELEASE,
        "d_sae": D_SAE,
        "total_nonbos_tokens": total_nonbos,
        "histogram_bins": edges_list,
        "features": index,
    }, separators=(",", ":")))
    total_bytes += idx_path.stat().st_size
    print(f"[emit]   wrote {idx_path.name}  {idx_path.stat().st_size/1024:.0f} KB (all {D_SAE} features)")

    # ---- curated_features.json (ids + reasons) ----
    cur_path = out_root / "dashboards" / "curated_features.json"
    cur_path.write_text(json.dumps({
        "layer": args.layer,
        "total_nonbos_tokens": total_nonbos,
        "count": len(curated),
        "thresholds": sel["thresholds"],
        "features": [{"id": int(f), "freq": float(f"{freq[f]:.6g}"),
                      "n_active": int(st.freq_count[f]), "reason": sel["reason"][f]}
                     for f in curated],
    }, indent=2))
    print(f"[emit]   wrote {cur_path.name}")

    # ---- report ----
    report = build_report(args, st, sel, total_bytes, n_chunks, tok)
    rep_path = Path(__file__).parent / "reports" / "harvest_report.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[emit]   wrote {rep_path}")
    return report, sel


def build_report(args, st, sel, total_bytes, n_chunks, tok):
    freq = sel["freq"]
    curated = sel["curated"]
    # spot-check: top-5 snippets for 10 random curated features
    rng = np.random.default_rng(args.seed)
    spot_ids = sorted(rng.choice(curated, size=min(10, len(curated)), replace=False).tolist())
    spot = []
    for f in spot_ids:
        heap = sorted(st.heaps[f], key=lambda x: -x[0])[:5]
        exs = []
        for (av, _uid, win_ids, win_acts, off) in heap:
            ids_list = [int(j) for j in win_ids.astype(np.int64)]
            act_tok = tok.decode([ids_list[off]]) if 0 <= off < len(ids_list) else "?"
            text = tok.decode(ids_list)  # full-window decode -> clean UTF-8
            exs.append({"act": round(float(av), 3), "act_token": act_tok,
                        "snippet": text.replace("\n", "\\n")})
        spot.append({"id": int(f), "freq": float(f"{freq[f]:.4g}"),
                     "n_active": int(st.freq_count[f]),
                     "reason": sel["reason"][int(f)], "top5": exs})
    return {
        "story": "S4",
        "dataset": args.dataset,
        "layer": args.layer,
        "device": args.device,
        "harvest_seconds": round(getattr(st, "harvest_seconds", 0.0), 1),
        "harvest_minutes": round(getattr(st, "harvest_seconds", 0.0) / 60.0, 2),
        "tok_per_sec_incl_bos": round(getattr(st, "tok_per_sec", 0.0), 0),
        "total_nonbos_tokens": st.total_nonbos,
        "n_alive": sel["n_alive"],
        "n_dead": sel["n_dead"],
        "pool_size": sel["pool_size"],
        "curated_count": len(curated),
        "curated_reason_counts": _reason_counts(sel["reason"]),
        "thresholds": sel["thresholds"],
        "histogram_bins": HIST_NBINS,
        "histogram_range": [HIST_LO, HIST_HI],
        "topk_heap": args.topk,
        "context_window": args.ctx,
        "n_dashboard_chunks": n_chunks,
        "total_dashboard_bytes": total_bytes,
        "total_dashboard_mb": round(total_bytes / 1e6, 2),
        "spot_check": spot,
    }


def _reason_counts(reason: dict) -> dict:
    out: dict[str, int] = {}
    for v in reason.values():
        key = "high_frequency" if v == "high_frequency" else "diversity"
        out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="S4 bounded activation harvest")
    ap.add_argument("--tokens", default="2e6", help="target non-BOS tokens (e.g. 2e6)")
    ap.add_argument("--dataset", default="Skylion007/openwebtext")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--chunk-seqs", type=int, default=32, help="sequences per forward/encode batch")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="D:/dev/sae-artifacts/L8")
    ap.add_argument("--work", default="D:/dev/sae-artifacts/L8/harvest")
    ap.add_argument("--topk", type=int, default=16, help="snippets kept per feature (heap size)")
    ap.add_argument("--ctx", type=int, default=16, help="context window +- tokens")
    ap.add_argument("--curated-target", type=int, default=400)
    ap.add_argument("--min-acts", type=int, default=24, help="min activations to be curatable")
    ap.add_argument("--max-freq", type=float, default=0.20, help="exclude near-always-on features")
    ap.add_argument("--min-max-act", type=float, default=1.0)
    ap.add_argument("--features-per-file", type=int, default=256)
    ap.add_argument("--examples-per-feature", type=int, default=12)
    ap.add_argument("--tokens-per-example", type=int, default=24)
    ap.add_argument("--logit-lens", action="store_true", default=True)
    ap.add_argument("--no-logit-lens", dest="logit_lens", action="store_false")
    ap.add_argument("--checkpoint-every", type=int, default=40, help="chunks between checkpoints")
    ap.add_argument("--log-every", type=int, default=10, help="chunks between progress logs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--stage", choices=["tokenize", "harvest", "emit", "all"], default="all")
    args = ap.parse_args()

    # Byte-level BPE tokens can be partial UTF-8 (U+FFFD) and snippets contain any
    # unicode; make stdout robust to the Windows cp1252 console. Done first, before
    # sae_lens/wandb wrap stdout.
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    if args.stage in ("tokenize", "all"):
        tok_path = stage_tokenize(args, tok)
    else:
        tok_path = Path(args.work) / "tokens_u16.npy"
    if args.stage == "tokenize":
        return

    if args.stage in ("harvest", "all"):
        st = stage_harvest(args, tok_path)  # also writes final harvest_ckpt.pkl
    if args.stage == "harvest":
        return
    if args.stage == "emit":
        with (Path(args.work) / "harvest_ckpt.pkl").open("rb") as fh:
            st = pickle.load(fh)["state"]

    report, sel = stage_emit(args, st)
    print("\n================= S4 HARVEST SUMMARY =================")
    print(f"dataset={report['dataset']}  layer={report['layer']}  device={report['device']}")
    print(f"non-BOS tokens={report['total_nonbos_tokens']:,}  "
          f"wall={report['harvest_minutes']} min  ({report['tok_per_sec_incl_bos']:,.0f} tok/s)")
    print(f"alive={report['n_alive']}  dead={report['n_dead']}  pool={report['pool_size']}  "
          f"curated={report['curated_count']} {report['curated_reason_counts']}")
    print(f"dashboards: {report['n_dashboard_chunks']} chunks + index  "
          f"total {report['total_dashboard_mb']} MB")
    print("=====================================================")
    print("\nSPOT-CHECK (top-5 snippets for 10 random curated features):")
    for s in report["spot_check"]:
        print(f"\n  feature {s['id']}  freq={s['freq']}  n_active={s['n_active']}  ({s['reason']})")
        for ex in s["top5"]:
            print(f"    [{ex['act']:.2f}] '{ex['act_token']}' | {ex['snippet'][:120]}")


if __name__ == "__main__":
    main()
