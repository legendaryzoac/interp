# S4 — Bounded activation harvest report

Machine-readable numbers: [`harvest_report.json`](./harvest_report.json). This is the
human-readable companion (the "do the features look real?" check + the frozen schema for
the S6/S7 handoff).

## Summary

| | |
|---|---|
| Dataset | **`Skylion007/openwebtext`** (streaming; matches the SAE's `cfg.dataset_path`) |
| Tokens | **2,000,123 non-BOS** tokens (15,749 contexts × 127; BOS prepended, excluded from stats) |
| Model / layer | GPT-2-small raw-HF, **layer 8 `resid_pre`** (== `blocks.7.hook_resid_post`, `stop_at_layer=8`) |
| SAE | `gpt2-small-res-jb` `blocks.8.hook_resid_pre` (d_sae = 24,576) |
| Basis | `to_sae_input` (mean-center over d_model) — the S2 contract, applied before encode |
| **Device / wall-clock** | **CPU, 46.5 min** (2,788 s; 723 tok/s incl BOS, 4 torch threads) |
| Alive / dead | **24,571 alive / 5 dead** (a feature is "dead" if it never fired on a non-BOS token) |
| Curatable pool | 24,569 features (passed the thresholds below) |
| **Curated (shipped) set** | **384 features** = 200 high-frequency ∪ 184 diversity |
| Dashboards total size | **3.14 MB** (`features_0000.json` 1.09 MB + `features_0001.json` 0.55 MB + `index.json` 1.50 MB); `curated_features.json` +46 KB |

### Why CPU, not the GTX 1070

The pinned S2/S3 venv (`D:/dev/sae-venv`) ships **CPU torch 2.13.0** and the whole verified
`sae-lens 6.45.3` / `transformer-lens` stack depends on it. Swapping in a CUDA build risked
(a) churning that pinned stack, (b) a ~2.5 GB download the **Pascal (sm_61)** 1070 may not
have kernels for in current torch wheels, and (c) filling C: (99 % full) via pip temp.
CPU at 46.5 min is "well under an hour" and reuses the verified venv exactly as instructed.
GPU projection: the GPT-2 forward + SAE encode would run ~20–40× faster on the 1070, but the
CPU-side streaming bookkeeping (per-feature candidate heaps + histogram `bincount`) would then
dominate, so realistic end-to-end on GPU ≈ **8–15 min**, not ~1 min. Not required for S5/S6.

## Method (memory-bounded, single pass)

- **BOS / position 0 excluded from every statistic** (frequency, max, histogram, snippets).
  It is the attention-sink token (S3 parity's |feats|≈500 is the sink); counting it would
  poison frequencies and top-snippet heaps. BOS is still prepended to every 128-token context
  (SAE trained `prepend_bos=True`); it just never counts as an activating token.
- **Frequency** = `n_active[f] / 2,000,123`. **Max** tracked exactly online.
- **Histogram**: single pass, **shared log-spaced bins** (20 bins, edges `0.05 … 512`, values
  clamped into range). Chosen over a true per-feature `[0, max]` histogram because the latter
  needs a second forward pass after the max is known; the shared log axis is standard for SAE
  dashboards and is written into every feature's JSON. Exact per-feature `max_act` stored too.
- **Top-K snippets**: per-feature min-heap, **K = 16**, context window **±16 tokens**. Stored as
  token IDs + per-token acts during the stream (decoded to strings only for the 384 curated
  features at emit) — a rolling per-feature activation threshold keeps the candidate pushes
  sparse so this stays fast and ~50 MB. Emitted as ≤12 examples × ≤24 tokens.
- Resumable: `tokenize` → packed uint16 token file; `harvest` checkpoints a pickle every 40
  chunks; `emit` reads the final checkpoint.

## Curated selection (the exact thresholds)

A feature is **curatable** iff `n_active ≥ 24` **and** `freq ≤ 0.20` **and** `max_act ≥ 1.0`:

| threshold | value | purpose |
|---|---|---|
| `min_acts` | 24 | enough activations for K=16 good snippets (min_freq = 1.2e-5) |
| `max_freq` | 0.20 | drop near-always-on / degenerate function-word features |
| `min_max_act` | 1.0 | drop barely-firing noise features |

The shipped 384 = union of:
1. **`high_frequency`** (200): top-200 by frequency within the pool.
2. **`diversity_bandN`** (184): pool split into 8 log-frequency bands; within each band, features
   ranked by **distinct-activating-token count** (favours varied/conceptual features over 400
   near-duplicate high-frequency ones), then filled toward the target of 400. (Landed at 384 —
   some low-frequency bands had fewer members than the per-band quota. Within the 300–500 target.)

Hand-curation of the final gallery is deferred to S8; this is the auto-selected superset.

## logit_lens — INCLUDED

**Yes** — all **384/384** curated per-feature objects carry a `logit_lens` field.
`logit_lens[f] = W_dec[f] @ W_U`, where `W_dec` is the SAE decoder row and **`W_U` is from the
default-processed `from_pretrained("gpt2")`** (so `fold_ln` **and** `center_unembed` are applied).
This is the standard SAE logit lens and it **applies the `ln_f` fold approximation** (the final
LayerNorm's affine scale is folded into `W_U`; the per-token LN normalization gain is not
re-applied — the usual, documented approximation). Top-10 `promoted` / bottom-10 `suppressed`
tokens, decoded to strings, with scores. S6 can ship these as-is; no recompute needed.

> Validation of the lens: curated **feature 18** fires on citation numbers inside square
> brackets (activating token `48` in `…heights.[48]\n\nAs originally…`), and its `logit_lens.promoted`
> is `]`, `],[`, `])`, `][` — closing brackets. Activating context and logit lens agree.

## Spot-check — 10 random curated features, top-5 activating snippets

Format: `[activation] «activating token» … snippet` (BOS shown as `<|endoftext|>`).

**feature 130** — freq 0.0090, n_active 18,042 (diversity_band7) — *narrative action / reaching verbs*
- `[9.54] « take»` …before I reached out to **take** her hands. Her head moved, resting so it lay sideways…
- `[9.52] « to»` …watched from atop a nearby roof, trying desperately **to** come up with a new play…
- `[9.45] « come»` …trying desperately to **come** up with a new play…but instead, thinking about what Mantis…
- `[9.04] « her»` …flexed her legs to soften **her** landing, and stood. She was less than fifty feet away…
- `[9.00] « barely»` …he could just **barely** touch the phone. The screen was cracked…

**feature 585** — freq 0.0356, n_active 71,108 (high_frequency) — *LaTeX / code / subword fragments (polysemantic)*
- `[5.57] « \»` …`{\displaystyle \varphi \to \psi }` … (LaTeX math markup)
- `[5.51] «var»` …`{\displaystyle \var`phi \to \psi } …
- `[5.12] «car»` …Bryan Coquard (Fra) Team Europ**car** 0:11:26…
- `[5.05] «car»` …Damien Gaudin (Fra) Team Europ**car**…
- `[4.78] « court»` …spread out as if they were on a basketball **court**…

**feature 1070** — freq 0.0285, n_active 57,025 (high_frequency) — *mixed / mid-clause (polysemantic)*
- `[5.46] « purposes»` …for all intents and **purposes**,
- `[5.39] « from»` …win probabilities derived **from** them…
- `[5.30] « strikes»` …looking for survivors following reported air **strikes** in Aleppo…
- `[4.99] « nothing»` …court dates at which **nothing** of consequence happens…
- `[4.92] « it»` …thrown out of the house or is **it** basically that you felt…

**feature 3892** — freq 0.0228, n_active 45,577 (high_frequency) — *evaluative adjectives / nouns*
- `[12.48] « commonplace»` …had become a cultural **commonplace**, there was a new backlash…
- `[11.51] « soon»` …progress on an immigration bill looking unlikely anytime **soon**…
- `[11.48] « small»` …the scale of his work was usually rather **small**…
- `[11.40] « impressive»` …commitment to environmental sustainability is **impressive**…
- `[10.94] « investigation»` …now under police **investigation** it would be inappropriate…

**feature 5710** — freq 0.00022, n_active 449 (diversity_band2) — **Arabic-script text (clean, monosemantic)**
- `[38.31] «ر»` …Dari عَرَبيْ - Arabic فارسی - Persian پښتو - Pashto…
- `[35.73] «ل»` …‘Red Crescent Idlib’ in Arabic: “الهلال الأحمرادلب”…
- `[35.58] «ن»` …Arabic: معطوب لونّاس‎…
- `[35.10] «د»` …the hotel (Arabic: فندق كارلتون)…
- `[34.74] «ر»` …the hotel (Arabic: فندق كارلتون)…

**feature 6568** — freq 0.0239, n_active 47,707 (high_frequency) — *mid-sentence verbs/nouns (polysemantic)*
- `[4.35] « worked»` …the printing factory where I **worked**
- `[4.18] « pops»` …Try it and out **pops** a card with a random famous quote
- `[4.11] « unusual»` …hidden in a book: an **unusual** offer…
- `[4.05] « provides»` …keep its negotiating demands confidential, and **provides** important information…
- `[4.03] « adviser»` …thanks to her wealth **adviser** and Oprah Winfrey?

**feature 11242** — freq 0.0234, n_active 46,700 (high_frequency) — *academic discourse markers ("However", "see")*
- `[6.63] « see»` …(For discussion of this possibility relevant to openness, **see**
- `[6.07] « However»` …the specific psychological mechanisms underlying those interactions. **However**, on a…
- `[5.53] «�»` …New Jersey and Pennsylvania: Reply.” American… (journal-citation context)
- `[5.33] « When»` …“that feeling of total flatness.” **When** they got…
- `[5.23] « However»` …(conservatives = −.07, liberals = .00). **However**,

**feature 14498** — freq 0.0035, n_active 6,938 (diversity_band5) — **hyphenated-compound modifiers**
- `[19.96] « snow»` …one of the sunny, **snow**-free days in early spring…
- `[19.64] « sub»` …still making challenging, **sub**-low beats…
- `[18.50] « eyewitness»` …noted for her sympathetic, **eyewitness** Letters Written in France…
- `[18.35] « winds»` …a lifeless, **winds**wept mound of rock…
- `[17.97] « fiber»` …flexible, **fiber**-optic solar cell…

**feature 19084** — freq 0.0014, n_active 2,747 (diversity_band4) — **numbers in scientific-notation / array dumps (clean)**
- `[33.24] «43»` …`3.514543`65e+00, -2.31743276e+00, 9.41614308e-01…
- `[33.08] «40»` …4.75871679e+06 -7.68**40**7951e+06 3.05267070e+05…
- `[32.96] «77»` …[ 1898.61079121] [ 126.23**77**6885] [ -223.4…
- `[32.83] «78»` …-2.485821**76**e-01, 4.34780093e-02… (numeric mantissa)
- `[32.69] «28»` …9.11611817e+06 7.372**82**635e+06 -9.03411488e+06…

**feature 19444** — freq 0.0424, n_active 84,879 (high_frequency) — *mixed prose + code (polysemantic)*
- `[6.54] « bag»` …Pour 4 cuts of water into the **bag**.
- `[6.45] « min»` …to `std::min`. namespace test { int **min**(int x, int y);…
- `[5.85] « atmospheric»` …“Intense and **atmospheric**…INDOCTRINATION”
- `[5.58] « poles»` …Zeros and **poles**. k : float System gain…
- `[5.57] «Contin»` …I will start by defining a few key terms… **Contin**uation

**Read of the spot-check:** the low-/mid-frequency diversity picks are the crisp, monosemantic
ones — **Arabic script (5710)**, **numeric/scientific-notation (19084)**, **hyphenated
compounds (14498)**, **academic discourse markers (11242)**, **citation brackets (18)**. The
top-frequency picks (585, 1070, 6568, 19444) are more polysemantic, as expected for very common
features — which is exactly why the diversity pass exists. The features are real and label-able.

## Recommendation on token count

2M tokens is **sufficient for S5/S6 to proceed** on the current 384: every curated feature has
≥24 activations (high-frequency ones 10⁴–10⁵), all top-16 heaps are full, and only 5 features
are fully dead. For a later "real" run, ~**5–10M tokens** would (a) lift more rare-but-interesting
features above the `min_acts=24` floor so the diversity pass has richer low-frequency bands, and
(b) sharpen the histograms of the lowest-frequency curated features. Not required now — flagged
rather than silently bumped. Re-run: `harvest.py --tokens 8e6 --curated-target 500`.

## Frozen schema for S6 (publish) / S7 (browser Inspector)

These are the **exact shapes as emitted** (files are compact single-line JSON). S5 adds `label`
and `label_confidence`; nothing else changes.

### `dashboards/index.json` — one entry for ALL 24,576 features
```json
{
  "layer": 8,
  "sae_release": "gpt2-small-res-jb",
  "d_sae": 24576,
  "total_nonbos_tokens": 2000123,
  "histogram_bins": [0.05, 0.0721, ... , 512.0],   // 21 shared log edges (20 bins)
  "features": {
    "18":  {"freq": 0.0011939, "chunk": 0,    "hasDashboard": true},
    "0":   {"freq": 0.0013654, "chunk": null, "hasDashboard": false}
  }
}
```
- Keys are **stringified feature ids** `"0".."24575"`. `chunk` = the `features_XXXX.json`
  index (int) if dashboarded, else `null`. `hasDashboard` true for the 384 curated features.
- Un-dashboarded features still get live activations in S7; the Inspector reads `freq` here and,
  when `hasDashboard`, lazy-loads `features_{chunk:04d}.json`. **S5 will add `"label"` (and
  `"label_confidence"`) to each curated feature's entry here.**

### `dashboards/features_XXXX.json` — array of curated dashboards (256/file → `features_0000.json`, `features_0001.json`)
```json
{
  "id": 18,
  "freq": 0.00119393,
  "max_act": 68.4129,
  "n_active": 2388,
  "selection_reason": "diversity_band4",           // or "high_frequency"
  "histogram": {
    "bins":   [0.05, ... , 512.0],                 // 21 edges (shared; == index.histogram_bins)
    "counts": [17, 1, 13, ... , 0]                 // 20 ints, nonzero-act counts per bin
  },
  "top_examples": [                                 // ≤12, sorted by activation desc
    {
      "tokens":    ["," , " the", " BAR", ...],     // ≤24 decoded string tokens (BPE; may be partial-UTF8)
      "acts":      [0, 0, 0, ..., 255, ..., 0],     // uint8 per token; true act = acts/255 * max_act
      "max_act":   68.4375,                         // this example's peak activation (quant scale)
      "act_index": 11                               // index of the activating token within tokens[]
    }
  ],
  "logit_lens": {
    "promoted":   [["]", 1.0989], ["],[", 1.0289], ...],   // top-10 [token, score]
    "suppressed": [["agra", -0.6639], [" hog", -0.5964], ...] // bottom-10 [token, score]
  }
  // label / label_confidence: ABSENT here — added by S5 (in index.json entries, per the brief)
}
```

### `dashboards/curated_features.json` — selection manifest (ids + reasons + thresholds)
Build artifact for S5/S8; not required to publish. `{layer, total_nonbos_tokens, count, thresholds, features:[{id, freq, n_active, reason}]}`.

## Notes for S5/S6
- S5 labels **only** the 384 curated features (prompt = each feature's `top_examples` with the
  `act_index` token marked). Cost stays single-digit dollars. Write `label` + `label_confidence`
  into the **index.json** entries (per the brief's `{feature_id → {label, freq, chunk, hasDashboard}}`).
- S6 publishes `index.json` + `features_XXXX.json` to the HF **dataset** repo and stamps a
  `content_hash` (reuse `export_sae_onnx.py::content_hash`). Total payload is ~3.14 MB.
- All artifacts are on **D:** (`D:/dev/sae-artifacts/L8/dashboards/`); nothing large in the repo.
