"""Stamp each variant manifest with a content_hash over its graph bytes.

The web runner versions its Cache API namespace on this hash, so returning
visitors re-download only when the artifacts actually change. File sizes alone
are not a safe version signal (a re-export can produce byte-different graphs
of identical size — exactly what happened with the fp16 fix).

Idempotent; run it as the last pipeline step, after export + quantize:
    python stamp_manifests.py --artifacts D:/dev/interp-artifacts/onnx \
                              --variants fp32 fp16 int8
"""

import argparse
import hashlib
import json
from pathlib import Path


def stamp(variant_dir: Path) -> str:
    manifest_path = variant_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    combined = hashlib.sha256()
    for f in sorted(variant_dir.glob("*.onnx")):
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        combined.update(f"{f.name}:{h.hexdigest()}\n".encode())
    content_hash = combined.hexdigest()[:16]
    manifest["content_hash"] = content_hash
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return content_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--variants", nargs="+", default=["fp32", "fp16", "int8"])
    args = parser.parse_args()
    for variant in args.variants:
        vdir = Path(args.artifacts) / variant
        print(f"{variant}: content_hash={stamp(vdir)}")


if __name__ == "__main__":
    main()
