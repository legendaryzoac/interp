"""basis_transform — map a browser-equivalent GPT-2 residual into the basis the
`gpt2-small-res-jb` SAEs were trained in.

This is the S2 correctness gate's single deliverable that the in-browser encoder
(S7) MUST replicate byte-for-byte. See BASIS_CONTRACT.md for the full derivation
and the measured reconstruction numbers that justify it.

TL;DR of the contract
---------------------
The browser (`web/src/lib/runner.ts::residualsEnteringBlocks()[L]`) produces GPT-2
`hook_resid_pre` at layer L from the **raw HuggingFace** weights — no TransformerLens
processing (proven ONNX == `from_pretrained_no_processing` by
`model-pipeline/validate.py`).

Joseph Bloom's `gpt2-small-res-jb` SAEs were trained on residuals from the
TransformerLens **default**-processed model (`HookedTransformer.from_pretrained`),
whose processing flags are `fold_ln`, `center_writing_weights`, `center_unembed`,
`fold_value_biases`. Of those, only `center_writing_weights` changes the *value* of
`hook_resid_pre`: it removes the per-position mean over d_model (LayerNorm makes the
model blind to that direction, so centering the writing weights is behaviour-
preserving but shifts the raw residual by exactly its d_model mean).

Therefore the raw -> SAE-input transform is a single, cheap, position-wise op:

    x_sae = x_raw - mean(x_raw over the d_model axis)

The SAE object itself then applies its own `b_dec` subtraction and any activation
normalization internally inside `SAE.encode`. For `gpt2-small-res-jb`,
`cfg.normalize_activations == "none"`, so no extra scale is needed (verified in
verify_basis.py). The optional `scale` argument exists only so a future release with
a constant-norm rescale can be supported without changing the call sites.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

D_MODEL = 768


def to_sae_input(
    resid: "np.ndarray",
    *,
    scale: Optional[float] = None,
) -> "np.ndarray":
    """Convert a raw-HF GPT-2 ``hook_resid_pre`` residual to SAE-ready input.

    Parameters
    ----------
    resid:
        Array with the last axis equal to ``d_model`` (768). Any leading shape
        ``[...]`` (e.g. ``[batch, seq, 768]`` or ``[n_tokens, 768]``) is preserved.
        These are the exact values ``runner.ts::residualsEnteringBlocks()[L]``
        yields in the browser.
    scale:
        Optional constant activation-normalization factor. ``None`` (the default,
        and the correct value for ``gpt2-small-res-jb`` whose
        ``normalize_activations == "none"``) applies no scaling. If a future SAE
        release uses a ``constant_norm_rescale``, pass that constant here and the
        browser encoder must multiply by the identical value.

    Returns
    -------
    np.ndarray
        float32 array, same shape as ``resid``, mean-centered over the last axis
        (and optionally scaled). This is what you feed to ``SAE.encode`` — or, in
        the browser, into the exported ``sae_enc.onnx`` graph.

    Notes
    -----
    * The operation is idempotent up to float error: centering an already-centered
      vector is a no-op.
    * It is purely position-wise; it never mixes information across tokens, so it is
      trivially correct for a single token, a sequence, or a batch.
    """
    x = np.asarray(resid, dtype=np.float32)
    if x.shape[-1] != D_MODEL:
        raise ValueError(
            f"expected last axis == d_model ({D_MODEL}), got shape {x.shape}"
        )
    x = x - x.mean(axis=-1, keepdims=True)
    if scale is not None:
        x = x * np.float32(scale)
    return x.astype(np.float32, copy=False)


__all__ = ["to_sae_input", "D_MODEL"]
