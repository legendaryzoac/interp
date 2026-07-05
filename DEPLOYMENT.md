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

The segmented ONNX graphs and tokenizer are **not** served from S3 — they live on
the Hugging Face Hub and the app fetches them at runtime from
`VITE_MODEL_BASE_URL` (baked in at build time by the deploy workflow).

- [ ] Create the HF model repo and upload the pipeline artifacts.
- [ ] In `.github/workflows/deploy.yml`, replace the placeholder
      `VITE_MODEL_BASE_URL` (currently
      `https://huggingface.co/PLACEHOLDER/resolve/main`) with the real repo URL,
      e.g. `https://huggingface.co/<user>/interp-gpt2/resolve/main`. It is wired
      into the build step already — this is a one-line change to activate.

## 5. Deploys after that

- **Web changes**: push to `main` → Actions builds `web/` and syncs `web/dist`
  to S3, then invalidates `/index.html`. Hashed assets under `/assets/*` are
  content-addressed and cached immutably, so they never need invalidation.
- **Infra changes**: `npm run deploy -w infra` locally (`npm run diff -w infra`
  shows what changes first).
