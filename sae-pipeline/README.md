# sae-pipeline — SAE Feature Explorer offline pipeline

Offline (Python) half of interp M2 (SAE Explorer). See `../SAE_EXPLORER_BUILD_BRIEF.md`.

## Status

- **S2 — SAE load + basis verification (the correctness gate): DONE / PASS.**
  - `BASIS_CONTRACT.md` — the residual-basis contract the browser encoder (S7) must
    replicate, with derivation, evidence, and the layer recommendation.
  - `basis_transform.py` — `to_sae_input(resid)`: the reference transform
    (mean-center over d_model). Unit-tested in `test_basis_transform.py` (9 tests).
  - `verify_basis.py` — loads `gpt2-small-res-jb`, reproduces the browser residual from
    the fp32 ONNX graphs, asserts the transform, measures reconstruction per layer.
  - `reports/basis_report.json` — machine-readable results.
- **S3 — Encoder ONNX export + W_dec blob + parity: DONE / PASS.**
  - `export_sae_onnx.py` — bakes `feats = ReLU((x − b_dec) @ W_enc + b_enc)` into
    `sae_enc.onnx` (fp32) + an fp16 variant (fp32 IO on both, reusing the model-pipeline
    `keep_io_types` + `force_outputs_fp32` recipe). Centering is **not** baked in — the caller
    (S7) applies `to_sae_input` first. Also writes the decoder as raw blobs (`w_dec_fp16.bin`,
    `b_dec_fp32.bin`), a manifest, and a `content_hash`.
  - `test_encoder_parity.py` — onnxruntime vs torch `sae.encode` on real layer-8 residuals.
  - `reports/encoder_parity.json` — machine-readable parity numbers.
  - Artifacts (on D:, not in the repo): `D:/dev/sae-artifacts/L8/` — `sae_enc.onnx` 75.6 MB,
    `sae_enc_fp16.onnx` 37.8 MB, `w_dec_fp16.bin` 37.7 MB, `b_dec_fp32.bin` 3 KB,
    `manifest.json` (`content_hash=36b59552633dccb3`).
- **S4 — Bounded activation harvest → curated feature set: DONE.**
  - `harvest.py` — streams OpenWebText (`Skylion007/openwebtext`, matches the SAE's
    training distribution), runs GPT-2-small, extracts layer-8 `resid_pre` the
    `verify_basis` way (raw-HF residual == browser basis), applies `to_sae_input`
    (mean-center), encodes with the SAE, and accumulates **memory-bounded** per-feature
    stats over the stream: frequency, max, a shared-log-bin histogram, and top-K (16)
    activating snippets via per-feature min-heaps (token IDs + per-token acts, decoded
    to strings only for the curated set). Three resumable stages: `tokenize` (→ packed
    uint16 token file on D:), `harvest` (forward/encode + checkpoint every N chunks),
    `emit` (curated selection + dashboards + index + logit lens + report).
  - Ships dashboards for a **bounded curated set** (~300–500), NOT all 24,576. Curated =
    top-frequency alive features ∪ a diversity pass across log-frequency bands. The
    browser still computes live activations for un-dashboarded features (S7).
  - **BOS / position 0 is excluded** from every statistic (attention sink; degenerate).
  - Artifacts (on D:): `D:/dev/sae-artifacts/L8/dashboards/` — `features_XXXX.json`
    (256 features/file), `index.json` (ALL 24,576: `{id:{freq,chunk,hasDashboard}}`),
    `curated_features.json` (ids + selection reasons). Report:
    `reports/harvest_report.json`.
- **S5 — Haiku auto-labeling: DONE.**
  - `label.py` — for each of the 384 curated features, prompts **Claude Haiku 4.5** with the
    feature's top-10 activating snippets (peak token in «angle brackets», secondary ≥50%-of-max
    tokens in ‹angle brackets›) + logit-lens promoted tokens, and asks for a strict-JSON
    `{label (≤8 words), confidence: high|medium|low}`. Route: **Amazon Bedrock**
    `bedrock-runtime.converse`, model `us.anthropic.claude-haiku-4-5-20251001-v1:0`, ambient
    AWS creds (`us-east-1`), temperature 0. Defensive parse (strips code fences, retry-once,
    else `null`/`low`); resumable disk cache keyed by feature id; `--limit`/`--force`.
  - Result: **384/384 labeled, 0 null**; confidence **176 high / 128 medium / 80 low** (the
    low bucket is the genuinely-polysemantic high-frequency features — honest calibration).
    **343,736 in / 10,614 out tokens, est. $0.397**. `label` + `label_confidence` written into
    both `features_XXXX.json` and the 384 `hasDashboard` entries of `index.json`.
  - Artifacts (on D:): cache `D:/dev/sae-artifacts/L8/labels_cache.json`; updated dashboard
    JSONs in `D:/dev/sae-artifacts/L8/dashboards/`. Reports: `reports/labeling_report.md` +
    `.json`. **Flag for S6:** the dashboard JSONs changed, so their `content_hash` must be
    re-stamped at publish time.
- **S6 — content_hash stamp + HF dataset publish prep: DONE.**
  - `stamp_dashboards.py` — computes a `content_hash` over the dashboards payload
    (`index.json`, `features_000{0,1}.json`, `curated_features.json`) with the same
    scheme as `export_sae_onnx.py` / model-pipeline `stamp_manifests.py` (per-file sha256,
    sorted by name, combined, first 16 hex), and writes/updates the **dashboards manifest**
    `D:/dev/sae-artifacts/L8/dashboards/manifest.json`
    (`{layer, sae_release, d_sae, n_curated:384, total_features:24576, files, content_hash}`).
    Idempotent. Also re-verifies the S3 encoder/decoder `manifest.json` and refreshes it
    only if those binaries changed. **Run result:** dashboards `content_hash=8dd4f035f5826424`
    (fresh, post-labeling); encoder/decoder `content_hash=36b59552633dccb3` **unchanged**
    (still matches S3, binaries untouched).
  - **Publish is Zack's manual step** — not automated here. Full runbook (create the HF
    **dataset** repo `legendaryzoac/interp-sae-gpt2-L8`, re-stamp, `hf upload` the `L8/` tree
    preserving layout, sanity-check, `VITE_SAE_BASE_URL`, dataset card) is in
    `../DEPLOYMENT.md` §5. Dataset resolve URLs use the `datasets/…/resolve/main/…` scheme;
    CORS verified permissive for cross-origin browser fetch. Total publish ≈ 154.4 MB
    (10 files: dashboards + encoder/decoder; `harvest/`, logs, and the label cache are excluded).

## W_dec blob byte layout (S7/S9 read this from JS)

`w_dec_fp16.bin` = raw little-endian **float16**, shape `[d_sae=24576, d_in=768]`,
C-contiguous row-major. Feature `f`'s decoder vector is the 768 fp16 values at bytes
`f*1536 .. (f+1)*1536`. `b_dec_fp32.bin` = 768 little-endian float32 (decoder bias, only
needed for `x_hat = f @ W_dec + b_dec`). Steering (S9) = `resid' = resid + α·W_dec[f]`
(read row `f`, upcast to fp32, axpy into the raw residual — valid per BASIS_CONTRACT §S9).

## Environment (C: is ~98% full — everything lives on D:)

Own venv, **not** the parity venv (`D:/dev/interp-venv`).

```
export UV_CACHE_DIR=D:/dev/uv-cache UV_PYTHON_INSTALL_DIR=D:/dev/uv-python HF_HOME=D:/dev/hf-cache
uv venv D:/dev/sae-venv --python 3.12
uv pip install --python D:/dev/sae-venv/Scripts/python.exe -r requirements.txt
```

## Run the gate (S2)

```
export HF_HOME=D:/dev/hf-cache
D:/dev/sae-venv/Scripts/python.exe verify_basis.py --layers 6 7 8 9
D:/dev/sae-venv/Scripts/python.exe -m pytest test_basis_transform.py -q
```

## Build + verify the encoder (S3)

S3 additionally needs `onnx` + `onnxconverter-common` in the venv:
`uv pip install --python D:/dev/sae-venv/Scripts/python.exe onnx onnxconverter-common`.

```
export HF_HOME=D:/dev/hf-cache
D:/dev/sae-venv/Scripts/python.exe export_sae_onnx.py --layer 8 --out D:/dev/sae-artifacts/L8
D:/dev/sae-venv/Scripts/python.exe test_encoder_parity.py --artifacts D:/dev/sae-artifacts/L8
D:/dev/sae-venv/Scripts/python.exe -m pytest test_encoder_parity.py -q -s
```

## Run the harvest (S4)

Needs `datasets` in the venv (already present). Everything (token cache, checkpoints,
dashboards) lands on **D:** — C: is full. Default is a first ~2M-token run.

```
export HF_HOME=D:/dev/hf-cache PYTHONIOENCODING=utf-8
# one command runs tokenize -> harvest -> emit (each stage is resumable):
D:/dev/sae-venv/Scripts/python.exe harvest.py --tokens 2e6
# or a bigger run / individual stages:
D:/dev/sae-venv/Scripts/python.exe harvest.py --tokens 5e6 --curated-target 500
D:/dev/sae-venv/Scripts/python.exe harvest.py --stage emit   # re-emit from the last checkpoint
```

Key flags: `--tokens` (non-BOS budget), `--chunk-seqs` (forward batch), `--curated-target`,
`--min-acts/--max-freq/--min-max-act` (curation thresholds), `--device cpu|cuda`,
`--no-resume`. CPU is the default (the pinned venv has CPU torch); a 2M-token run is ~40 min
on this desktop's 4 CPU threads (GPT-2 forward is `stop_at_layer=8`, so blocks 8–11 and the
unembed are skipped). Progress + ETA print to stdout; a pickle checkpoint (`harvest_ckpt.pkl`
in the work dir) lets a crash resume without re-encoding.

## Run the labeling (S5)

Needs `boto3` in the venv (additive, on D:) and ambient AWS creds (IAM user `zachary`,
`us-east-1`; Bedrock Haiku 4.5 access confirmed). No secrets are hardcoded — the default
credential chain is used.

```
export HF_HOME=D:/dev/hf-cache PYTHONIOENCODING=utf-8   # AWS creds are ambient
# sanity: label + print the first 5, no write-back:
D:/dev/sae-venv/Scripts/python.exe label.py --limit 5 --sample 5
# full run: label all 384 (resumes from cache), then write back + verify:
D:/dev/sae-venv/Scripts/python.exe label.py
# merge an existing cache into the JSONs only / verify only:
D:/dev/sae-venv/Scripts/python.exe label.py --write-only
D:/dev/sae-venv/Scripts/python.exe label.py --verify
```

Key flags: `--limit N` (first N features), `--force` (ignore cache), `--workers` (in-flight
calls, default 4), `--no-write` (label only), `--anthropic` (direct Anthropic API fallback).
Each result is cached to `D:/dev/sae-artifacts/L8/labels_cache.json` keyed by feature id, so
reruns never re-pay. Full 384-feature run costs **~$0.40** and takes a few minutes on CPU.
Report: `reports/labeling_report.md` / `.json`.

### Emitted schemas (frozen for S5/S6/S7)

`dashboards/index.json` — the Inspector's map for ALL features (S5 adds `label` per entry):
```
{ "layer":8, "sae_release":"gpt2-small-res-jb", "d_sae":24576, "total_nonbos_tokens":N,
  "histogram_bins":[...21 edges...],
  "features": { "<id>": {"freq":float, "chunk":int|null, "hasDashboard":bool}, ... } }
```
`dashboards/features_XXXX.json` — array of curated per-feature dashboards (label/​
label_confidence added by S5):
```
{ "id":1234, "freq":0.0012, "max_act":7.2, "n_active":2400, "selection_reason":"...",
  "histogram": {"bins":[...21 edges...], "counts":[...20 ints...]},
  "top_examples": [ {"tokens":[str...≤24], "acts":[uint8...],   // act = acts/255*max_act
                     "max_act":7.2, "act_index":11}, ... ≤12 ],
  "logit_lens": {"promoted":[["tok",1.9],...10], "suppressed":[["tok",-1.4],...10]} }
```
`histogram` uses **shared** log-spaced bin edges (same for every feature; single-pass,
no second forward). `logit_lens = W_dec[f] @ W_U` with `W_U` from the default-processed
model (`fold_ln + center_unembed`) — the standard SAE logit lens.

## Run the stamper (S6)

Re-computes the dashboards `content_hash` (stale after S5's label write-back) and writes the
dashboards manifest; also re-verifies the S3 encoder/decoder manifest. Idempotent — safe to
re-run. No network, no venv extras beyond the base install.

```
D:/dev/sae-venv/Scripts/python.exe stamp_dashboards.py --artifacts D:/dev/sae-artifacts/L8
```

Writes `D:/dev/sae-artifacts/L8/dashboards/manifest.json` and prints both hashes. Current run:
dashboards `content_hash=8dd4f035f5826424`, encoder/decoder `content_hash=36b59552633dccb3`
(unchanged). **Re-run this before every HF publish** (any dashboard/label change flips the
hash). The manual publish steps (create dataset repo, `hf upload`, env var, dataset card) are
in `../DEPLOYMENT.md` §5 — publishing is Zack's action, not the pipeline's.

## Encoder parity note (S3)

Parity note: gates are **scale-normalized** (`max|Δ|/max|feats|`), not absolute — at layer 8
GPT-2's attention-sink activations make `|x|~3000` and `|feats|~500`, so an absolute 1e-4 diff
is float noise (onnx-fp32 is in fact closer to the fp64 truth than the torch reference). See
the docstring in `test_encoder_parity.py`. fp32 passes ~400× under gate; fp16 ~40× under,
with feature rankings (top-10) preserved.

## Result in one line

The browser produces **raw-HF** `hook_resid_pre`; the SAE expects it **mean-centered over
d_model** (`center_writing_weights`). Apply `x − mean(x)` and reconstruction is perfect;
skip it and every feature is silently wrong. `normalize_activations = "none"` (no scale);
`apply_b_dec_to_input = true` (encoder pre-subtracts b_dec — matters for S3).
