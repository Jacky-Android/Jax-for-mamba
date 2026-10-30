import jax
import jax.numpy as jnp
import numpy as np

from jax_mamba.selective_scan import selective_scan, selective_step


def _inputs():
    k = jax.random.split(jax.random.PRNGKey(42), 5)
    u = jax.random.normal(k[0], (2, 9, 3))
    dt = jax.nn.softplus(jax.random.normal(k[1], (2, 9, 3)))
    A = -jnp.exp(jax.random.normal(k[2], (3, 4)))
    B = jax.random.normal(k[3], (2, 9, 4))
    C = jax.random.normal(k[4], (2, 9, 4))
    D = jnp.ones((3,))
    return u, dt, A, B, C, D


def test_scan_matches_step_and_chunking():
    u, dt, A, B, C, D = _inputs()
    ys, state = selective_scan(u, dt, A, B, C, D)
    x = jnp.zeros_like(state)
    expected = []
    for t in range(u.shape[1]):
        y, x = selective_step(x, u[:, t], dt[:, t], A, B[:, t], C[:, t], D)
        expected.append(y)
    np.testing.assert_allclose(ys, jnp.stack(expected, axis=1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(state, x, rtol=1e-5, atol=1e-5)
    first, xmid = selective_scan(u[:, :4], dt[:, :4], A, B[:, :4], C[:, :4], D)
    second, xend = selective_scan(u[:, 4:], dt[:, 4:], A, B[:, 4:], C[:, 4:], D, xmid)
    np.testing.assert_allclose(ys, jnp.concatenate([first, second], axis=1), atol=1e-5)
    np.testing.assert_allclose(state, xend, atol=1e-5)


def test_scan_jit_and_grad():
    args = _inputs()
    ref = selective_scan(*args)[0]
    compiled = jax.jit(selective_scan)(*args)[0]
    np.testing.assert_allclose(ref, compiled, atol=1e-5)
    grad = jax.grad(lambda u: jnp.sum(selective_scan(u, *args[1:])[0] ** 2))(args[0])
    assert grad.shape == args[0].shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_invalid_shape():
    import pytest
    u, dt, A, B, C, D = _inputs()
    with pytest.raises(ValueError):
        selective_scan(u[:, 0], dt, A, B, C, D)
