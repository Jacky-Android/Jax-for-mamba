# Jax-for-mamba
# 使用 JAX 和 Flax 实现 S4
注意：本项目使用 JAX 结合 Flax 神经网络库。虽然我们个人主要使用 Torch，但 JAX 的函数式特性非常适合处理 S4 的一些复杂性。我们大量使用 vmap、scan、它们的神经网络对应函数，以及最重要的 jax.jit 来编译高效的 S4 层。
<img width="960" height="720" alt="image" src="https://github.com/user-attachments/assets/046509d4-ffb8-426a-acd7-15b2051309ae" />
