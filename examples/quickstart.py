"""Initialize a Mamba model, run the same tokens via full and step modes."""
import jax
import jax.numpy as jnp

from jax_mamba.model import MambaBlock, MambaLM, greedy_generate


block = MambaBlock(d_model=16, d_state=8)
x = jax.random.normal(jax.random.PRNGKey(0), (2, 12, 16))
variables = block.init(jax.random.PRNGKey(1), x)
y = jax.jit(lambda inputs: block.apply(variables, inputs))(x)
print("MambaBlock:", x.shape, "->", y.shape)

lm = MambaLM(vocab_size=32, d_model=32, n_layers=2)
tokens = jnp.array([[1, 2, 3, 4]])
params = lm.init(jax.random.PRNGKey(2), tokens)
logits = lm.apply(params, tokens)
print("MambaLM logits:", logits.shape)
print("Greedy generation:", greedy_generate(lm, params, tokens, 4))
