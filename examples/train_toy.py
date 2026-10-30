"""Train on repeated short patterns; a demonstration, not a benchmark."""
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from jax_mamba.model import MambaLM


model = MambaLM(vocab_size=16, d_model=32, n_layers=2, d_state=8)
key = jax.random.PRNGKey(0)
dummy = jnp.zeros((8, 16), dtype=jnp.int32)
variables = model.init(key, dummy)
state = train_state.TrainState.create(
    apply_fn=model.apply, params=variables["params"],
    tx=optax.adam(learning_rate=2e-3),
)


@jax.jit
def train_step(state, tokens):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, tokens[:, :-1])
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, tokens[:, 1:]
        ).mean()
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


for i in range(51):
    key, subkey = jax.random.split(key)
    pattern = jax.random.randint(subkey, (8, 4), 0, 16)
    tokens = jnp.tile(pattern, (1, 4))
    state, loss = train_step(state, tokens)
    if i % 10 == 0:
        print(f"step={i:3d}  cross_entropy={float(loss):.4f}")
