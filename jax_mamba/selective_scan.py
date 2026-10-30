"""Reference selective state-space scan (Mamba-1 convention) in pure JAX.

Shapes: u/delta = (batch, length, channels), A = (channels, state),
B/C = (batch, length, state), D = (channels,).  Delta is already positive.
The input term uses the Euler approximation delta * B * u, as in the
original Mamba selective_scan_ref; this is *not* exact ZOH for B.
"""

import jax
import jax.numpy as jnp


def selective_step(state, u_t, delta_t, A, B_t, C_t, D):
    """One SSM update. State (batch, channels, state_size)."""
    transition = jnp.exp(delta_t[..., None] * A[None, :, :])
    state = transition * state + (delta_t * u_t)[..., None] * B_t[:, None, :]
    y = jnp.sum(state * C_t[:, None, :], axis=-1) + u_t * D
    return y, state


def selective_scan(u, delta, A, B, C, D, initial_state=None):
    """Run an input-dependent SSM over (batch, length, channels).

    Returns (outputs, final_state). The scan carries only O(batch*channels*state)
    recurrent state, and is differentiable/JIT-compatible. No custom CUDA kernel.
    """
    if u.ndim != 3:
        raise ValueError("u must have shape (batch, length, channels)")
    batch, length, channels = u.shape
    state_size = A.shape[-1]
    if (delta.shape != u.shape or A.shape != (channels, state_size)
            or B.shape != (batch, length, state_size)
            or C.shape != (batch, length, state_size)
            or D.shape != (channels,)):
        raise ValueError("Incompatible shapes for selective scan")
    if initial_state is None:
        initial_state = jnp.zeros((batch, channels, state_size), dtype=u.dtype)

    def step(state, inputs):
        u_t, dt_t, b_t, c_t = inputs
        y, state = selective_step(state, u_t, dt_t, A, b_t, c_t, D)
        return state, y

    final_state, ys = jax.lax.scan(
        step, initial_state,
        (jnp.swapaxes(u, 0, 1), jnp.swapaxes(delta, 0, 1),
         jnp.swapaxes(B, 0, 1), jnp.swapaxes(C, 0, 1)),
    )
    return jnp.swapaxes(ys, 0, 1), final_state
