"""Unit tests for basis_transform.to_sae_input.

Pure-numpy, no model download required — run with:
    D:/dev/sae-venv/Scripts/python.exe -m pytest sae-pipeline/test_basis_transform.py -q

The model-in-the-loop equivalence check (that this transform actually reproduces the
SAE's training basis from a raw-HF residual) lives in verify_basis.py, which asserts
    max| to_sae_input(raw_resid) - default_processed_resid | < 1e-3
across real GPT-2 activations. These unit tests pin the pure mathematical contract.
"""
import numpy as np
import pytest

from basis_transform import D_MODEL, to_sae_input


def test_output_is_mean_centered_over_last_axis():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5, D_MODEL)).astype(np.float32) * 3.0 + 7.0
    y = to_sae_input(x)
    # Every position's mean over d_model must be ~0.
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-5)


def test_matches_manual_mean_subtraction():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, D_MODEL)).astype(np.float32)
    expected = x - x.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(to_sae_input(x), expected, rtol=0, atol=1e-6)


def test_shape_and_dtype_preserved():
    x = np.ones((2, 3, D_MODEL), dtype=np.float64)  # deliberately float64 input
    y = to_sae_input(x)
    assert y.shape == x.shape
    assert y.dtype == np.float32


def test_idempotent_on_centered_input():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((7, D_MODEL)).astype(np.float32)
    once = to_sae_input(x)
    twice = to_sae_input(once)
    np.testing.assert_allclose(once, twice, rtol=0, atol=1e-6)


def test_constant_vector_maps_to_zero():
    # A vector pointing purely along the all-ones direction carries no information
    # the model can read (LayerNorm removes it); the transform must annihilate it.
    x = np.full((3, D_MODEL), 4.2, dtype=np.float32)
    y = to_sae_input(x)
    np.testing.assert_allclose(y, 0.0, atol=1e-5)


def test_removes_only_the_mean_component():
    # Construct centered signal + known constant offset; transform must recover signal.
    rng = np.random.default_rng(3)
    signal = rng.standard_normal((5, D_MODEL)).astype(np.float32)
    signal -= signal.mean(axis=-1, keepdims=True)  # already centered
    offset = np.array([1.0, -2.0, 3.0, 0.5, -4.0], dtype=np.float32)[:, None]
    x = signal + offset
    np.testing.assert_allclose(to_sae_input(x), signal, atol=1e-5)


def test_single_token_1d():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(D_MODEL).astype(np.float32) + 10.0
    y = to_sae_input(x)
    assert y.shape == (D_MODEL,)
    assert abs(float(y.mean())) < 1e-5


def test_optional_scale_applied_after_centering():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((6, D_MODEL)).astype(np.float32) + 2.0
    scale = 0.37
    expected = (x - x.mean(axis=-1, keepdims=True)) * np.float32(scale)
    np.testing.assert_allclose(to_sae_input(x, scale=scale), expected, atol=1e-6)


def test_wrong_d_model_raises():
    with pytest.raises(ValueError):
        to_sae_input(np.zeros((3, 512), dtype=np.float32))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
