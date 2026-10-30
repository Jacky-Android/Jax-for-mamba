"""Readable, reference Mamba-1 implementation using JAX and Flax Linen."""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .selective_scan import selective_scan, selective_step


def _dt_bias_init(dt_min, dt_max):
    def init(key, shape, dtype=jnp.float32):
        log_dt = jax.random.uniform(
            key, shape, minval=math.log(dt_min), maxval=math.log(dt_max)
        )
        dt = jnp.exp(log_dt)
        return (dt + jnp.log(-jnp.expm1(-dt))).astype(dtype)
    return init


def _a_log_init(key, shape, dtype=jnp.float32):
    del key
    return jnp.broadcast_to(
        jnp.log(jnp.arange(1, shape[-1] + 1, dtype=dtype)), shape
    )


def init_mamba_cache(batch_size, d_model, d_state=16, expand=2, d_conv=4,
                     dtype=jnp.float32):
    """Pure, explicit cache; no mutable Flax collection is needed."""
    inner = d_model * expand
    return {
        "conv": jnp.zeros((batch_size, d_conv - 1, inner), dtype=dtype),
        "ssm": jnp.zeros((batch_size, inner, d_state), dtype=dtype),
    }


class RMSNorm(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        weight = self.param("weight", nn.initializers.ones, (x.shape[-1],))
        scale = jax.lax.rsqrt(
            jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
            + self.eps
        )
        return (x * scale.astype(x.dtype)) * weight


class MambaBlock(nn.Module):
    """Mamba-1 mixer: input projection -> causal DW conv -> selective SSM -> gate.

    Full sequence: (B, L, d_model). step consumes one (B, d_model)
    token and a returned explicit cache. Full and step paths share parameters.
    """
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None
    dt_min: float = 0.001
    dt_max: float = 0.1
    bias: bool = False

    def setup(self):
        if self.d_conv < 1 or self.d_state < 1 or self.expand < 1:
            raise ValueError("d_conv, d_state and expand must be >= 1")
        if self.dt_min <= 0 or self.dt_max <= self.dt_min:
            raise ValueError("Expected 0 < dt_min < dt_max")
        inner = self.d_model * self.expand
        rank = self.dt_rank or math.ceil(self.d_model / 16)
        self.in_proj = nn.Dense(inner * 2, use_bias=self.bias)
        self.conv_kernel = self.param(
            "conv_kernel", nn.initializers.lecun_normal(),
            (self.d_conv, 1, inner),
        )
        self.conv_bias = self.param("conv_bias", nn.initializers.zeros, (inner,))
        self.x_proj = nn.Dense(rank + 2 * self.d_state, use_bias=False)
        self.dt_proj = nn.Dense(
            inner, use_bias=True,
            kernel_init=nn.initializers.normal(rank ** -0.5),
            bias_init=_dt_bias_init(self.dt_min, self.dt_max),
        )
        self.A_log = self.param("A_log", _a_log_init, (inner, self.d_state))
        self.D = self.param("D", nn.initializers.ones, (inner,))
        self.out_proj = nn.Dense(self.d_model, use_bias=self.bias)

    def _ssm_inputs(self, x):
        projected = self.x_proj(x)
        dt_low_rank, B, C = jnp.split(
            projected,
            [projected.shape[-1] - 2 * self.d_state,
             projected.shape[-1] - self.d_state],
            axis=-1,
        )
        delta = jax.nn.softplus(self.dt_proj(dt_low_rank))
        return delta, B, C

    def __call__(self, x):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("Expected x with shape (batch, length, d_model)")
        inner = self.d_model * self.expand
        x, z = jnp.split(self.in_proj(x), 2, axis=-1)
        padded = jnp.pad(x, ((0, 0), (self.d_conv - 1, 0), (0, 0)))
        x = jax.lax.conv_general_dilated(
            padded, self.conv_kernel, (1,), "VALID",
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=inner,
        ) + self.conv_bias
        x = jax.nn.silu(x)
        delta, B, C = self._ssm_inputs(x)
        y, _ = selective_scan(x, delta, -jnp.exp(self.A_log), B, C, self.D)
        return self.out_proj(y * jax.nn.silu(z))

    def step(self, x_t, cache):
        """Process one token and return (output, updated_cache).

        This method is called with model.apply(variables, token, cache,
        method=model.step). Cache starts with init_mamba_cache.
        """
        if x_t.ndim != 2 or x_t.shape[-1] != self.d_model:
            raise ValueError("Expected x_t with shape (batch, d_model)")
        x, z = jnp.split(self.in_proj(x_t), 2, axis=-1)
        window = jnp.concatenate([cache["conv"], x[:, None, :]], axis=1)
        x = jnp.einsum("bkd,kd->bd", window, self.conv_kernel[:, 0, :])
        x = jax.nn.silu(x + self.conv_bias)
        delta, B, C = self._ssm_inputs(x)
        y, state = selective_step(
            cache["ssm"], x, delta, -jnp.exp(self.A_log), B, C, self.D
        )
        out = self.out_proj(y * jax.nn.silu(z))
        return out, {"conv": window[:, 1:, :], "ssm": state}


class MambaLM(nn.Module):
    """Small causal language model for experiments, not pretrained weights."""
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.d_model)
        self.norms = [RMSNorm() for _ in range(self.n_layers)]
        self.blocks = [MambaBlock(
            self.d_model, d_state=self.d_state,
            d_conv=self.d_conv, expand=self.expand,
        ) for _ in range(self.n_layers)]
        self.final_norm = RMSNorm()
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, tokens):
        x = self.embedding(tokens)
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        return self.lm_head(self.final_norm(x))

    def init_cache(self, batch_size, dtype=jnp.float32):
        return tuple(init_mamba_cache(
            batch_size, self.d_model, self.d_state, self.expand,
            self.d_conv, dtype,
        ) for _ in range(self.n_layers))

    def step(self, token_t, caches):
        """Consume one (batch,) token and return logits plus new caches."""
        x = self.embedding(token_t)
        new_caches = []
        for norm, block, cache in zip(self.norms, self.blocks, caches):
            update, new_cache = block.step(norm(x), cache)
            x = x + update
            new_caches.append(new_cache)
        return self.lm_head(self.final_norm(x)), tuple(new_caches)


def greedy_generate(model, variables, prompt, max_new_tokens):
    """Simple greedy decoding, intended as a readable reference (not optimized)."""
    if prompt.ndim != 2 or prompt.shape[1] < 1:
        raise ValueError("prompt must be (batch, length>=1)")
    cache = model.init_cache(prompt.shape[0])
    logits = None
    for t in range(prompt.shape[1]):
        logits, cache = model.apply(
            variables, prompt[:, t], cache, method=model.step
        )
    generated = []
    for _ in range(max_new_tokens):
        next_token = jnp.argmax(logits, axis=-1)
        generated.append(next_token)
        logits, cache = model.apply(
            variables, next_token, cache, method=model.step
        )
    if not generated:
        return prompt
    return jnp.concatenate([prompt, jnp.stack(generated, axis=1)], axis=1)
