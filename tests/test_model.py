import pytest

pytest.importorskip("flax")

import jax
import jax.numpy as jnp
import numpy as np

from jax_mamba.model import MambaBlock, MambaLM, greedy_generate, init_mamba_cache


def test_mixer_step_matches_full_sequence():
    model = MambaBlock(d_model=8, d_state=4, d_conv=3, expand=2)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 8))
    variables = model.init(jax.random.PRNGKey(2), x)
    full = model.apply(variables, x)
    cache = init_mamba_cache(2, 8, d_state=4, d_conv=3, expand=2)
    outputs = []
    for i in range(x.shape[1]):
        out, cache = model.apply(variables, x[:, i], cache, method=model.step)
        outputs.append(out)
    np.testing.assert_allclose(full, jnp.stack(outputs, axis=1), rtol=3e-5, atol=3e-5)
    assert jnp.all(jnp.isfinite(jax.jit(lambda inputs: model.apply(variables, inputs))(x)))
    grads = jax.grad(lambda p: jnp.sum(model.apply({"params": p}, x) ** 2))(
        variables["params"]
    )
    assert bool(jnp.all(jnp.isfinite(grads["A_log"])))


def test_lm_token_step_and_generation():
    model = MambaLM(vocab_size=24, d_model=8, n_layers=2, d_state=4, d_conv=3)
    tokens = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    variables = model.init(jax.random.PRNGKey(3), tokens)
    full = model.apply(variables, tokens)
    cache = model.init_cache(tokens.shape[0])
    outputs = []
    for t in range(tokens.shape[1]):
        logits, cache = model.apply(variables, tokens[:, t], cache, method=model.step)
        outputs.append(logits)
    np.testing.assert_allclose(full, jnp.stack(outputs, axis=1), rtol=4e-5, atol=4e-5)
    generated = greedy_generate(model, variables, tokens[:, :2], max_new_tokens=3)
    assert generated.shape == (2, 5)
