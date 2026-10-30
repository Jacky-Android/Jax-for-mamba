# Jax-for-mamba · JAX / Flax Mamba-1 参考复现

本仓库从 S4 / 线性时不变 SSM 教学继续，添加可测试、可训练、支持逐 token 推理的 Mamba-1 (Selective State Space Model) JAX/Flax 参考实现。重点是算法正确性、可读性、可复现性，不是 CUDA/Triton 的速度。

## 包含内容

- 原始 S4/SSM 教学 Notebook: [code.ipynb](code.ipynb)，保留原状。
- JAX selective scan: [jax_mamba/selective_scan.py](jax_mamba/selective_scan.py)，通过 lax.scan 实现输入相关的 delta、B、C 和稳定的对角 A。
- Flax Mamba-1: [jax_mamba/model.py](jax_mamba/model.py)，含输入投影、因果深度卷积、SiLU、选择性扫描、门控和输出投影。
- MambaLM: 多层残差 Mixer、RMSNorm、词嵌入、语言模型预测头。
- 逐 token 流式推理: 显式携带卷积历史缓存与 SSM 状态；提供 greedy_generate。
- tests、examples、pyproject.toml 与 GitHub Actions CPU CI。

本实现与官方 Mamba-1 selective_scan_ref 保持一致：先计算 state = exp(delta*A)*state + delta*B*u，再计算 y = C*state + D*u；这里 delta*B*u 是官方参考实现的近似输入离散化，并非精确 ZOH。逐元素指数仅适用于此处的对角 A，不能代替一般矩阵指数。

## 安装

Python >=3.10。默认使用 JAX CPU。GPU/TPU 请按照 JAX 官方安装说明配置。

    git clone https://github.com/Jacky-Android/Jax-for-mamba.git
    cd Jax-for-mamba
    python -m pip install -e '.[dev]'
    python -m pytest -q

## 运行

    python examples/quickstart.py
    python examples/train_toy.py

## 最小示例

    import jax
    import jax.numpy as jnp
    from jax_mamba.model import MambaBlock, init_mamba_cache

    block = MambaBlock(d_model=64, d_state=16, d_conv=4)
    x = jnp.ones((2, 32, 64))
    variables = block.init(jax.random.PRNGKey(0), x)
    y = block.apply(variables, x)  # (2, 32, 64)
    cache = init_mamba_cache(2, 64, d_state=16, d_conv=4)
    y0, cache = block.apply(variables, x[:, 0], cache, method=block.step)

## 模型结构

    Input (batch, length, d_model)
      -> Linear(d_model, 2 * d_inner) -> split x, z
      x -> Causal Depthwise Conv1D -> SiLU
        -> Linear(dt_rank + 2*d_state) -> dt, B, C
        -> softplus(Linear(dt)) -> positive delta
        -> selective scan with A = -exp(A_log) and D skip
      scan_output * SiLU(z) -> Linear(d_inner, d_model)

    MambaLM: Embedding -> N x [RMSNorm -> MambaBlock -> residual]
                       -> RMSNorm -> LM head

## 限制

这是 Mamba-1 教学参考实现，不包括 Mamba-2、官方预训练权重格式兼容、融合 CUDA/Triton 内核、分布式训练或正式 benchmark。训练速度可能低于官方优化版本。原始 code.ipynb 是 S4/SSM 教学内容，不等于完整 S4 的 DPLR/HiPPO 复现。

参考资料：[Mamba paper](https://arxiv.org/abs/2312.00752)、[Official Mamba](https://github.com/state-spaces/mamba)、[selective_scan_ref](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)。
