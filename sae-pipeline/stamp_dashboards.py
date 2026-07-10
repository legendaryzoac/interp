"""Stamp the SAE dashboards payload with a content_hash over its file bytes.

The web runner (Epic C) versions its Cache API namespace on a manifest
`content_hash`, exactly like the segmented GPT-2 ONNX weights do
(model-pipeline/stamp_manifests.py). So the browser only re-downloads the
dashboards when they actually change. File sizes alone are not a safe version
signal, so we hash the bytes.

Two manifests live under D:/dev/sae-artifacts/L8/:

  * dashboards/manifest.json  — the dashboards payload (index.json,
    features_XXXX.json, curated_features.json). WRITTEN by this script. Its
    hash must be (re)computed now because S5 rewrote index.json +
    features_000{0,1}.json to add label/label_confidence.

  * manifest.json             — the S3 encoder/decoder set (sae_enc*.onnx,
    w_dec_fp16.bin, b_dec_fp32.bin), content_hash produced in S3. This script
    only VERIFIES it (recomputes over the same 4 binaries) and refreshes it iff
    the bytes changed. They shouldn't have — S6 does not touch them.

Same hash scheme as export_sae_onnx.py / stamp_manifests.py: per-file sha256,
files sorted by name, combined as "<name>:<hexdigest>\n", first 16 hex chars.

Idempotent; safe to re-run:
    python stamp_dashboards.py --artifacts D:/dev/sae-artifacts/L8
"""

import argparse
import hashlib
import json
from pathlib import Path

# The S3 encoder/decoder binaries (the other manifest's constituent files) and
# the hash S3 stamped over them, for the verify/refresh check.
S3_BINARIES = ["sae_enc.onnx", "sae_enc_fp16.onnx", "w_dec_fp16.bin", "b_dec_fp32.bin"]
S3_KNOWN_HASH = "36b59552633dccb3"


def content_hash(files: list[Path]) -> str:
    """Per-file sha256, sorted by name, combined, truncated to 16 hex chars.

    Byte-for-byte the scheme in export_sae_onnx.py and stamp_manifests.py.
    """
    combined = hashlib.sha256()
    for f in sorted(files, key=lambda p: p.name):
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        combined.update(f"{f.name}:{h.hexdigest()}\n".encode())
    return combined.hexdigest()[:16]


def stamp_dashboards(dash_dir: Path) -> str:
    """Compute + write dashboards/manifest.json. Returns the content_hash."""
    # Every published dashboard JSON except the manifest itself. Sorting by name
    # (done inside content_hash) makes the result stable + order-independent.
    files = sorted(
        p for p in dash_dir.glob("*.json") if p.name != "manifest.json"
    )
    if not files:
        raise SystemExit(f"no dashboard JSON files found in {dash_dir}")

    chash = content_hash(files)

    # sae_release is authoritative in index.json; fall back to the known value.
    index = json.loads((dash_dir / "index.json").read_text(encoding="utf-8"))
    sae_release = index.get("sae_release", "gpt2-small-res-jb")
    d_sae = int(index.get("d_sae", 24576))
    total_features = len(index.get("features", {})) or d_sae

    manifest = {
        "layer": 8,
        "sae_release": sae_release,
        "d_sae": d_sae,
        "n_curated": 384,
        "total_features": total_features,
        "files": {p.name: p.stat().st_size for p in files},
        "content_hash": chash,
    }
    (dash_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return chash


def verify_s3(root: Path) -> tuple[str, bool]:
    """Recompute the S3 encoder/decoder content_hash; refresh iff it changed.

    Returns (recomputed_hash, matches_stored).
    """
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binaries = [root / name for name in S3_BINARIES]
    missing = [b.name for b in binaries if not b.exists()]
    if missing:
        raise SystemExit(f"S3 binaries missing: {missing}")

    recomputed = content_hash(binaries)
    stored = manifest.get("content_hash")
    matches = recomputed == stored

    if not matches:
        # Bytes actually changed — refresh so the stamp stays truthful.
        manifest["content_hash"] = recomputed
        manifest["files"] = {b.name: b.stat().st_size for b in binaries}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return recomputed, matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="D:/dev/sae-artifacts/L8")
    args = ap.parse_args()

    root = Path(args.artifacts)
    dash_dir = root / "dashboards"

    dhash = stamp_dashboards(dash_dir)
    print(f"dashboards: content_hash={dhash}")
    print(f"  wrote {dash_dir / 'manifest.json'}")
    for name, size in json.loads(
        (dash_dir / "manifest.json").read_text(encoding="utf-8")
    )["files"].items():
        print(f"    {name}: {size} bytes")

    s3_hash, s3_ok = verify_s3(root)
    status = "MATCHES stored (unchanged)" if s3_ok else "CHANGED — manifest refreshed"
    print(f"S3 encoder/decoder: content_hash={s3_hash}  [{status}]")
    print(f"  known S3 hash = {S3_KNOWN_HASH}  ->  {'OK' if s3_hash == S3_KNOWN_HASH else 'DIFFERS'}")


if __name__ == "__main__":
    main()
