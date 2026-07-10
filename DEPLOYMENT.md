# Getting interp.zackwithers.com live: 0 → production checklist

All AWS resources are CDK-managed. Infra deploys run locally with your credentials;
CI only builds `web/` and ships the static site to S3 + CloudFront.

## 1. One-time AWS setup

- [ ] CDK bootstrap is already done account-wide
      (`CDKToolkit` in 545628619410/us-east-1) — nothing to run here.
- [ ] `npm install` (repo root — hydrates the `web` + `infra` workspaces)
- [ ] `npm run deploy -w infra` — first synth/deploy happens locally with your
      credentials. Creates the ACM cert (DNS-validated automatically against the
      `zackwithers.com` Route 53 hosted zone), the private S3 bucket with Origin
      Access Control, the CloudFront distribution (SPA 403/404 → `/index.html`),
      the `interp` A/AAAA alias records, and the GitHub OIDC deploy role.
      Cert validation adds a couple of DNS-propagation minutes on the first run.
- [ ] Note the stack outputs: `BucketName`, `DistributionId`, `CiRoleArn`.

> Synth needs no AWS context lookup: the hosted zone is referenced by its known id
> via `HostedZone.fromHostedZoneAttributes` (not `fromLookup`), so there is no
> `cdk.context.json` to commit. Actually reaching AWS (deploy, and the DNS
> validation that blocks on it) still requires your credentials — hence "locally".

## 2. GitHub repo

- [ ] Repo: [legendaryzoac/interp](https://github.com/legendaryzoac/interp)
      (public — portfolio piece), branch `main`
- [ ] Repo settings → Secrets and variables → Actions:
  - Variable `S3_BUCKET` — from stack output `BucketName`
  - Variable `CLOUDFRONT_DISTRIBUTION_ID` — from stack output `DistributionId`
  - Secret `AWS_ROLE_ARN` — from stack output `CiRoleArn`

## 3. AWS: OIDC role for GitHub Actions (no long-lived keys)

- [ ] The `token.actions.githubusercontent.com` identity provider already exists
      in the account (shared with the other portfolio projects) and is reused.
- [ ] The deploy role is **CDK-managed** (`GithubDeployRole` in
      `infra/lib/site-stack.ts`): trusted for
      `repo:legendaryzoac/interp:ref:refs/heads/main` only, with S3
      put/delete/list on the site bucket and `cloudfront:CreateInvalidation`
      on the distribution. No manual IAM steps.

## 4. Model weights (Hugging Face Hub)

The segmented ONNX graphs are **not** served from S3 — they live on the Hugging
Face Hub and the app fetches them at runtime from `VITE_MODEL_BASE_URL` (baked in
at build time by the deploy workflow). The browser code
(`web/src/lib/runner.ts`) fetches, per variant `fp32|fp16|int8`:

- `${VITE_MODEL_BASE_URL}/{variant}/{name}.onnx` — 14 graphs each
  (`embed`, `block_00`…`block_11`, `unembed`)
- `${VITE_MODEL_BASE_URL}/{variant}/manifest.json` — fetched with
  `cache: 'no-cache'`; its `content_hash` field versions the browser's Cache API
  namespace, so **the manifest must ship next to the .onnx files** or the app
  can't size the download or invalidate stale caches.

So the Hub repo just needs to mirror the local artifact tree exactly:
`<repo root>/{fp32,fp16,int8}/{14 .onnx + manifest.json}`. The three variants are
~652 / 331 / 252 MB (~1.2 GB total).

This is a **one-time manual publish** done by you (not CI — CI has no HF
credentials). Run it from a terminal on this Windows machine. Do it once, then
only step 4.7 (set the workflow var) is needed to activate.

### 4.1 Create a Hugging Face account

If you don't already have one, sign up at <https://huggingface.co/join> (free).
Pick a username — it becomes your repo namespace and **may differ from your
GitHub handle** (`legendaryzoac`). If `legendaryzoac` is taken on HF, whatever
username you land on is the `<user>` in every URL below; substitute accordingly.
Verify your email before continuing (uploads require a verified account).

### 4.2 Create a write-scoped access token

1. Go to <https://huggingface.co/settings/tokens> → **Create new token**.
2. Token type **Write** (or a fine-grained token with write access to your repos)
   — the default **Read** token cannot push.
3. Name it something like `interp-upload`, create it, and copy the value
   (`hf_…`). You won't be able to see it again; if you lose it, make a new one.

### 4.3 Install the HF CLI into the existing venv

The Hub CLI was renamed from `huggingface-cli` to **`hf`** in 2025; it ships in
the `huggingface_hub` PyPI package. Install it into the project venv at
`D:/dev/interp-venv` so it's version-pinned with the rest of the pipeline tooling
(this project installs with `uv`):

```powershell
py -m uv pip install --python D:/dev/interp-venv/Scripts/python.exe -U "huggingface_hub"
```

If you'd rather not use `uv`, the plain-pip equivalent works too:

```powershell
D:/dev/interp-venv/Scripts/python.exe -m pip install -U "huggingface_hub"
```

Verify the CLI resolves (call it through the venv so you get the pinned one):

```powershell
D:/dev/interp-venv/Scripts/hf.exe version
```

> If `hf.exe` isn't on the venv's `Scripts` path for some reason, `D:/dev/interp-venv/Scripts/python.exe -m huggingface_hub.cli --help` is the module-invocation fallback. All `hf …` commands below can be prefixed with `D:/dev/interp-venv/Scripts/` to guarantee the venv copy.

### 4.4 Log in

```powershell
D:/dev/interp-venv/Scripts/hf.exe auth login
```

Choose **Paste an access token** and paste the `hf_…` write token from 4.2 (or
pick the browser device-code flow it offers). Confirm it registered:

```powershell
D:/dev/interp-venv/Scripts/hf.exe auth whoami
```

This should print your HF username. That username is your `<user>` below.

### 4.5 Create the model repo

Public repo (this is a portfolio piece, so public is intended). Suggested id
`legendaryzoac/interp-gpt2-onnx` — **replace `legendaryzoac` with your actual HF
username from 4.4 if it differs**:

```powershell
D:/dev/interp-venv/Scripts/hf.exe repos create legendaryzoac/interp-gpt2-onnx
```

Model is the default repo type, and repos are public by default (no flag needed;
`--private` would make it private). If it already exists, add `--exist-ok`.

### 4.6 Upload the three variant folders (+ manifests)

`hf upload` signature is `hf upload <repo_id> <local_path> <path_in_repo>`. It
uploads a whole folder recursively and preserves structure, so upload each
variant directory to a matching subfolder in the repo. The `manifest.json` in
each folder is an ordinary file and gets uploaded along with the `.onnx` files —
no special handling needed. Run all three (adjust the repo id to your username):

```powershell
D:/dev/interp-venv/Scripts/hf.exe upload legendaryzoac/interp-gpt2-onnx D:/dev/interp-artifacts/onnx/fp32 fp32
D:/dev/interp-venv/Scripts/hf.exe upload legendaryzoac/interp-gpt2-onnx D:/dev/interp-artifacts/onnx/fp16 fp16
D:/dev/interp-venv/Scripts/hf.exe upload legendaryzoac/interp-gpt2-onnx D:/dev/interp-artifacts/onnx/int8 int8
```

Each command prints a `https://huggingface.co/<user>/interp-gpt2-onnx/tree/main/<variant>`
URL on success. `--repo-type model` is the default, so it's omitted above.

**Time caveat:** ~1.2 GB total. On a typical home upstream this is roughly
10–40 minutes wall-clock (upstream bandwidth, not download speed, is the limit);
the CLI shows per-file progress bars. `hf upload` handles large files and
resumes automatically if you re-run it, so a dropped connection is safe to retry
— it skips already-uploaded blobs.

### 4.7 Sanity-check one URL in a browser

The public download URL scheme is
`https://huggingface.co/<user>/<repo>/resolve/main/<path>`. Open the manifest for
one variant directly, e.g. (substitute your username):

```
https://huggingface.co/legendaryzoac/interp-gpt2-onnx/resolve/main/fp32/manifest.json
```

You should get the manifest JSON (it should include a `content_hash` and a
`files` map). Then confirm a graph resolves — pasting
`…/resolve/main/fp32/embed.onnx` should start a binary download. The `resolve`
endpoint transparently 302-redirects to HF's CDN; browser `fetch` (and the app)
follow that redirect automatically, and the endpoint returns
`Access-Control-Allow-Origin`, so cross-origin fetches from the CloudFront site
work. (This is the same mechanism Hugging Face's own Transformers.js uses to load
ONNX models in-browser from any origin.) The real cross-origin proof happens in
step 4.9 when the deployed app actually fetches these files — if a graph loads
there, CORS is fine.

### 4.8 Point the deploy workflow at the repo

In `.github/workflows/deploy.yml`, replace the placeholder
`VITE_MODEL_BASE_URL` (currently `https://huggingface.co/PLACEHOLDER/resolve/main`)
with your repo's resolve base — **no trailing slash, no variant segment**; the
app appends `/{variant}/{name}.onnx` itself:

```yaml
VITE_MODEL_BASE_URL: https://huggingface.co/legendaryzoac/interp-gpt2-onnx/resolve/main
```

It's already wired into the `npm run build -w web` step, so this one-line change
activates model loading on the next push to `main`.

### 4.9 (Optional) Local pre-deploy smoke test

To confirm the URLs work before pushing, point a local build at the same base via
`web/.env` (Vite reads `VITE_`-prefixed vars from there). Create/edit
`web/.env`:

```
VITE_MODEL_BASE_URL=https://huggingface.co/legendaryzoac/interp-gpt2-onnx/resolve/main
```

Then `npm run dev -w web` (or `npm run build -w web && npm run preview -w web`)
and load a model in the app — a successful run confirms fetch, CORS, and the
manifest/content_hash path end-to-end. `web/.env` is local-only; don't commit it
(CI gets the value from the workflow, not this file).

### 4.10 Re-publishing after a pipeline re-run

If you re-export the ONNX artifacts, re-run the three `hf upload` commands from
4.6 — same commands overwrite the changed files (unchanged blobs are skipped).
The stamped `content_hash` in each `manifest.json` changes when the graphs
change, which flips the browser's Cache API namespace
(`interp-onnx-<variant>-<hash>` in `runner.ts`), so returning users
**automatically refetch** the new graphs and evict the stale cache — no manual
cache-busting or CloudFront invalidation needed for the model files (they're not
on CloudFront anyway).

### 4.11 (Recommended) Add a README model card

For portfolio polish, add a one-paragraph `README.md` model card to the HF repo
so the repo page isn't blank. Create it and upload the single file:

```powershell
D:/dev/interp-venv/Scripts/hf.exe upload legendaryzoac/interp-gpt2-onnx D:/dev/interp-artifacts/onnx/README.md README.md
```

Suggested content:

```markdown
# interp-gpt2-onnx

Segmented ONNX export of GPT-2 (small) for **in-browser mechanistic
interpretability**. The model is split into 14 graphs — `embed`, twelve
transformer blocks (`block_00`–`block_11`) that also emit attention patterns, and
`unembed` for the logit lens — so a static web app can chain them entirely
client-side with onnxruntime-web (WebGPU/WASM). Provided in three precisions:
`fp32/`, `fp16/`, and `int8/`, each with a `manifest.json` (byte sizes +
`content_hash` for cache versioning).

Live demo: <https://interp.zackwithers.com> · Source:
<https://github.com/legendaryzoac/interp>
```

(Write that file to `D:/dev/interp-artifacts/onnx/README.md` first, or upload it
from wherever you save it — the second `hf upload` arg is just its local path.)

## 5. SAE Feature Explorer artifacts (Hugging Face **dataset**)

The SAE Feature Explorer (interp M2) ships a second HF-hosted payload, separate
from the model weights of §4: the **labeled SAE feature dashboards** plus the
**layer-8 SAE encoder/decoder** for `gpt2-small-res-jb`. The browser (Epic C)
fetches these at runtime from `VITE_SAE_BASE_URL` and versions its Cache API
namespace on each manifest's `content_hash` — the same scheme as the model
weights. This is a **one-time manual publish** done by you (not CI), from a
terminal on this Windows machine.

**Why a _dataset_ repo, not a model repo.** These are derived data (feature
statistics, activation snippets, labels, exported weights for in-browser use),
not a trained model, so a **dataset repo** is the right HF repo type. This
changes the public URL: a dataset's raw files resolve at

```
https://huggingface.co/datasets/<user>/<repo>/resolve/main/<path>
```

— note the extra **`datasets/`** path segment versus the model-weights URL in
§4.8 (`https://huggingface.co/<user>/<repo>/resolve/main/<path>`, no
`datasets/`). Everything else — the `resolve/main/…` layout, the 302→CDN
redirect, and CORS — behaves like §4.

> **CORS (verified 2026-07-10).** A cross-origin `curl` (with
> `Origin: https://interp.zackwithers.com`) against a public dataset's
> `…/resolve/main/…` URL returns `access-control-allow-origin:
> https://interp.zackwithers.com` on the 307/302 from `huggingface.co`, and the
> file bytes come back from the Xet/LFS CDN (`us.aws.cdn.hf.co/xet-bridge-us/…`)
> with `access-control-allow-origin: *` and `accept-ranges: bytes`. So
> whole-file browser `fetch()` from interp.zackwithers.com is permitted — the
> exact mechanism the §4 model weights already rely on. (Refs: HF Hub
> [downloading datasets](https://huggingface.co/docs/hub/datasets-downloading);
> the long-standing browser-hosting CORS request
> [huggingface_hub#468](https://github.com/huggingface/huggingface_hub/issues/468).
> The one documented gap is **preflighted HTTP _Range_** reads on the Xet bridge
> ([datasets#7931](https://github.com/huggingface/datasets/issues/7931)) — used
> by DuckDB-Wasm/parquet-wasm partial reads; the SAE app does plain full-file
> GETs, so it is unaffected.) The only thing not verifiable before upload is the
> live end-to-end fetch from the deployed app against **your** repo — confirm it
> in step 5.7's smoke test, exactly as §4.9 does for the model weights.

The local artifact tree to publish lives entirely on **D:** at
`D:/dev/sae-artifacts/L8/`. Only these **10 files** ship (the `harvest/`
checkpoint, `*.log`, and `labels_cache.json` are working files and are
excluded by the upload command below):

```
L8/sae_enc.onnx                       75,599,167 B   (fp32 encoder graph)
L8/sae_enc_fp16.onnx                  37,800,097 B   (fp16 encoder graph)
L8/w_dec_fp16.bin                     37,748,736 B   (decoder [24576,768] fp16)
L8/b_dec_fp32.bin                          3,072 B   (decoder bias, 768 fp32)
L8/manifest.json                           1,737 B   (encoder/decoder manifest, content_hash=36b59552633dccb3)
L8/dashboards/index.json               1,527,591 B   (all 24,576 features + labels on the 384 curated)
L8/dashboards/features_0000.json       1,115,004 B   (256 curated dashboards)
L8/dashboards/features_0001.json         562,018 B   (128 curated dashboards)
L8/dashboards/curated_features.json       46,726 B   (curated ids + selection reasons)
L8/dashboards/manifest.json                  319 B   (dashboards manifest, content_hash=8dd4f035f5826424)
```

Total upload ≈ **154.4 MB** (≈147 MiB): ~3.25 MB dashboards + ~151.15 MB
encoder/decoder. On a typical home upstream that's roughly **2–8 minutes**
wall-clock (upstream bandwidth is the limit); `hf upload` shows per-file
progress and resumes on re-run, so a dropped connection is safe to retry.

### 5.1 Reuse your HF account + write token

Reuse the same Hugging Face account and **Write** access token from §4.1–4.2 —
no new account or token is needed. If you did §4, you're already logged in
(§4.4). The `<user>` below is your HF username from `hf auth whoami` (it **may
differ** from the GitHub handle `legendaryzoac`; substitute your actual
username in every command and URL).

### 5.2 Verify the `hf` CLI in the SAE venv

This project uses its own venv at `D:/dev/sae-venv` (not the model-pipeline's
`D:/dev/interp-venv`). It already has `huggingface_hub` with the `hf` CLI:

```powershell
D:/dev/sae-venv/Scripts/hf.exe version
D:/dev/sae-venv/Scripts/hf.exe auth whoami   # should print your HF username
```

If `auth whoami` says you're not logged in, run
`D:/dev/sae-venv/Scripts/hf.exe auth login` and paste the §4.2 write token.

### 5.3 Create the dataset repo

Public (portfolio piece). Suggested id `legendaryzoac/interp-sae-gpt2-L8` —
**replace `legendaryzoac` with your HF username** if it differs. The
`--repo-type dataset` flag is what makes it a dataset repo (and puts
`datasets/` in the resolve URL):

```powershell
D:/dev/sae-venv/Scripts/hf.exe repos create legendaryzoac/interp-sae-gpt2-L8 --repo-type dataset --exist-ok
```

Repos are public by default (`--private` would make it private); `--exist-ok`
makes it safe to re-run.

### 5.4 Re-stamp the dashboards `content_hash` (do this first, every time)

The dashboards manifest's `content_hash` must reflect the current bytes. S5's
labeling rewrote `index.json` + `features_000{0,1}.json`, so re-stamp before
uploading (idempotent; safe to re-run):

```powershell
$env:HF_HOME="D:/dev/hf-cache"
D:/dev/sae-venv/Scripts/python.exe C:/Users/Zack/ClaudeCode/interp/sae-pipeline/stamp_dashboards.py --artifacts D:/dev/sae-artifacts/L8
```

It writes `D:/dev/sae-artifacts/L8/dashboards/manifest.json`
(`content_hash=8dd4f035f5826424` for the current files) and also re-verifies
the encoder/decoder `manifest.json` (should still report
`content_hash=36b59552633dccb3` — unchanged since S3).

### 5.5 Upload the `L8/` tree (one command, exact layout preserved)

`hf upload <repo_id> <local_path> <path_in_repo>` uploads a folder recursively
and preserves structure. Uploading `D:/dev/sae-artifacts/L8` into repo path
`L8` reproduces the `L8/dashboards/…` + `L8/*.onnx|*.bin|manifest.json` layout
the app expects. The three `--exclude` globs drop the ~94 MB of working files
(`harvest/` checkpoint + token cache, `*.log`, `labels_cache.json`) so only the
10 files above ship:

```powershell
D:/dev/sae-venv/Scripts/hf.exe upload legendaryzoac/interp-sae-gpt2-L8 D:/dev/sae-artifacts/L8 L8 --repo-type dataset --exclude "harvest/*" --exclude "*.log" --exclude "labels_cache.json"
```

On success it prints a
`https://huggingface.co/datasets/legendaryzoac/interp-sae-gpt2-L8/tree/main/L8`
URL. (Verified locally: those exact excludes select precisely the 10 published
files and skip all 6 working files.)

### 5.6 Sanity-check one URL in a browser

Open the dashboards manifest directly (substitute your username) — note the
`datasets/` segment and the `L8/` prefix:

```
https://huggingface.co/datasets/legendaryzoac/interp-sae-gpt2-L8/resolve/main/L8/dashboards/manifest.json
```

You should get the JSON with `content_hash`, `d_sae: 24576`, `n_curated: 384`,
and a `files` map. Then confirm a large blob resolves — pasting
`…/resolve/main/L8/sae_enc_fp16.onnx` should start a binary download (it
302-redirects to the HF CDN, which browser `fetch` follows automatically).

### 5.7 Point Epic C at the repo (do NOT edit the workflow yet)

The web app (Epic C) will read a base URL from a new build-time var
**`VITE_SAE_BASE_URL`** — **no trailing slash, no variant/subpath segment**; the
app appends `/dashboards/…`, `/sae_enc_fp16.onnx`, `/manifest.json`, etc.
itself. It must point at the `L8` folder:

```
VITE_SAE_BASE_URL = https://huggingface.co/datasets/legendaryzoac/interp-sae-gpt2-L8/resolve/main/L8
```

This gets wired into `.github/workflows/deploy.yml` (next to
`VITE_MODEL_BASE_URL`) **as part of Epic C**, once the web runner that consumes
it exists — **do not add it to the workflow now**. For a local pre-deploy smoke
test before then, put it in `web/.env` (Vite reads `VITE_`-prefixed vars there;
local-only, don't commit) and run the app — a successful dashboards + encoder
load confirms fetch, CORS, and the `content_hash` cache path end-to-end against
your repo.

### 5.8 (Recommended) Add a dataset card

For portfolio polish, add a one-paragraph `README.md` dataset card so the repo
page isn't blank. Write it to `D:/dev/sae-artifacts/L8/README.md`, then upload
the single file to the repo root:

```powershell
D:/dev/sae-venv/Scripts/hf.exe upload legendaryzoac/interp-sae-gpt2-L8 D:/dev/sae-artifacts/L8/README.md README.md --repo-type dataset
```

Suggested content:

```markdown
# interp-sae-gpt2-L8

Derived data for **in-browser mechanistic interpretability**: labeled
sparse-autoencoder (SAE) feature dashboards plus the SAE encoder/decoder for
**GPT-2 (small), layer-8 residual stream** (`hook_resid_pre`), SAE release
`gpt2-small-res-jb` (Joseph Bloom). Contains, under `L8/`: `dashboards/` — an
`index.json` over all 24,576 features and per-feature dashboards for 384 curated
features (top activating snippets, activation histograms, logit-lens tokens, and
auto-generated labels), each manifest carrying a `content_hash` for cache
versioning; and the exported encoder (`sae_enc.onnx` fp32 + `sae_enc_fp16.onnx`)
and raw decoder blobs (`w_dec_fp16.bin`, `b_dec_fp32.bin`) so a static web app
can compute and steer features entirely client-side with onnxruntime-web. These
are artifacts of an interpretability demo, not a general-purpose dataset.

Live demo: <https://interp.zackwithers.com> · Source:
<https://github.com/legendaryzoac/interp>
```

### 5.9 Re-publishing after a pipeline re-run

If you re-run the offline pipeline (re-harvest, re-label, or re-export the
encoder), **re-run 5.4 first** (re-stamp), then re-run the 5.5 upload — the same
command overwrites changed files and skips unchanged blobs. Each manifest's
`content_hash` flips only when its files actually change, which flips the
browser's Cache API namespace, so returning users automatically refetch the
changed payload and evict the stale cache — no manual cache-busting needed
(these files aren't on CloudFront anyway).

## 6. Deploys after that

- **Web changes**: push to `main` → Actions builds `web/` and syncs `web/dist`
  to S3, then invalidates `/index.html`. Hashed assets under `/assets/*` are
  content-addressed and cached immutably, so they never need invalidation.
- **Infra changes**: `npm run deploy -w infra` locally (`npm run diff -w infra`
  shows what changes first).
