# S4 — Bounded activation harvest → curated feature set

Execute **S4 only**. Do not start S5 (labeling) or S6 (publish). No git commands.

## Read first
1. `interp/SAE_EXPLORER_BUILD_BRIEF.md` → "Epic B stories" → S4, and the per-feature JSON schema.
2. `interp/sae-pipeline/BASIS_CONTRACT.md` — authoritative residual-basis transform (`to_sae_input`).
3. `interp/sae-pipeline/basis_transform.py`, `verify_basis.py` — reuse the residual-extraction + SAE-load code; do NOT re-derive it.

## Environment (reuse S2/S3 setup — do not rebuild)
- Python: `D:/dev/sae-venv/Scripts/python.exe`; always `HF_HOME=D:/dev/hf-cache`.
- **C: is ~98% full.** All artifacts + the OpenWebText/dataset cache go on **D:** (e.g. `D:/dev/sae-artifacts/L8/dashboards/`, dataset cache under `HF_HOME` on D:). Only code + small reports in the repo.
- **GPU:** this desktop has a GTX 1070 (8GB). Use CUDA torch if available for the harvest — check `torch.cuda.is_available()`; if the S2/S3 venv has CPU-only torch, install a CUDA build into `D:/dev/sae-venv` (caches on D:) OR run CPU and report the slower wall-clock. State which you did. GPT-2-small forward over a few M tokens on the 1070 should be well under an hour; on CPU it's tolerable but slower.
- Own only `interp/sae-pipeline/`. Do NOT touch `web/`, `infra/`, `model-pipeline/`.

## Scope discipline (the whole point of S4)
The original plan wanted dashboards for all 24,576 features over 20M+ tokens — we are NOT doing that. Ship dashboards for a **bounded curated set (~300–500 features)**; the browser computes live activations for any feature later, so un-dashboarded features still work in the Inspector. This keeps harvest time, S5 labeling cost, and hosting size ~50× smaller.

## Task

### 1. `harvest.py` — stream + encode
- Stream **~2–5M tokens** of OpenWebText via HF `datasets` (`stas/openwebtext-10k` or `Skylion007/openwebtext` streaming — [VERIFY] a currently-loadable id; pick one and note it). Tokenize with GPT-2 BPE (match the model's tokenizer). Pack into sequences (e.g. length 128), skip empties.
- Batch through GPT-2-small, extract layer-8 `resid_pre` the `verify_basis` way, apply `to_sae_input`, encode with the SAE → feature activations `[tokens, 24576]`.
- Make token count a CLI flag (`--tokens 2e6`), default ~2M for a first run; support resuming/checkpointing so a crash doesn't lose hours.

### 2. Per-feature statistics (streaming, memory-bounded)
Never hold all activations in RAM. As you stream, maintain per feature (24576 of them):
- **activation frequency** = fraction of tokens where the feature is active (act > 0).
- **max activation** and a **histogram** of nonzero activations (fixed bins, e.g. 20 bins over [0, running-max] — a two-pass or log-bin approach is fine; document it).
- **top-K activating snippets** (K≈16) via a per-feature min-heap keyed on the token's activation: keep the token, its activation, and a context window (≈±16 tokens) as token STRINGS (decode now) plus the activating position and its value. Bound memory: only the K best per feature.

### 3. Curated feature selection (~300–500)
Select the shipped set as the union of:
- the **top-N by activation frequency** among "alive" features (exclude dead features, freq≈0, and near-always-on/degenerate ones — document the thresholds), and
- a **diversity/interestingness** pass so it's not all high-freq function-word features (e.g. sample across frequency bands, or keep features whose top snippets are token-diverse). Keep it simple and documented; hand-curation happens later in S8.
Write the chosen ids to `curated_features.json` with the selection reason per feature.

### 4. Emit artifacts (to D:/dev/sae-artifacts/L8/dashboards/)
- **Per-feature JSON** (schema from the build brief) for ONLY the curated set — `label`/`label_confidence` left null/absent (S5 fills them). Include `id`, `freq`, `histogram`, `top_examples` (≤12 examples × ≤24 tokens, activations 8-bit-quantized per the brief), and the logit-lens fields IF cheap to compute here (`W_dec[f] @ W_U`, top/bottom 10 tokens — [DECIDE] apply ln_f approx or skip, document); otherwise leave logit_lens for later and note it.
- Chunk them (e.g. 256 features/file: `features_0000.json` …).
- **`index.json`** covering ALL 24,576 features: `{id: {freq, chunk|null, hasDashboard: bool}}` (label added in S5). This is what the browser Inspector reads to know which features have dashboards; un-dashboarded features still show live activations in S7.
- Keep total dashboard size modest (hundreds of features × ≤~15KB). Report the total.

### 5. Verify + report
- Spot-check: print the top-5 activating snippets for ~10 random curated features so a human can eyeball coherence (e.g. a feature that fires on years, or on legal terms). Include these in your report — this is the "do the features look real?" check.
- Report: dataset id used, token count, GPU-or-CPU + wall-clock, #alive/#dead features, curated count + selection thresholds, total artifact size, the 10 spot-checked features with their snippets, and anything S5/S6 need (schema of index.json + per-feature JSON as emitted).
- Update `sae-pipeline/README.md` with the S4 command(s).

## Notes
- Do NOT call Claude/Bedrock for labels — that is S5.
- Do NOT upload to HF Hub — that is S6. Local D: artifacts only.
- If ~2M tokens gives too few high-quality snippets for some curated features, note it and recommend a token count for the "real" run rather than silently bumping it.