"""Emit out/token_fixture.json: the HF GPT-2 tokenization of all 16 prompts.

The web app unit-tests its JavaScript BPE tokenizer against this fixture, so the
`tokens` field is the *raw* byte-level BPE surface form (convert_ids_to_tokens,
with the GPT-2 space marker U+0120 'G-with-dot' and newline marker U+010A) --
that is the deterministic, information-preserving tokenizer output a JS
implementation must reproduce exactly. `token_ids` are the vocab ids.

Usage:
    python token_fixture.py
"""

import json
from pathlib import Path

from transformers import GPT2TokenizerFast

HERE = Path(__file__).parent


def main():
    prompts = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    fixture = []
    for p in prompts:
        ids = tokenizer(p["text"])["input_ids"]
        fixture.append({
            "id": p["id"],
            "text": p["text"],
            "token_ids": ids,
            "tokens": tokenizer.convert_ids_to_tokens(ids),  # raw byte-level BPE
        })
        print(f"  {p['id']}: {len(ids)} tokens")

    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "token_fixture.json").write_text(
        json.dumps({"prompts": fixture}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"wrote {out_dir / 'token_fixture.json'}")


if __name__ == "__main__":
    main()
