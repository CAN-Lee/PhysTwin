# PhysTwin: WarpMPMSimulator 集成说明文档

## 1. 概述 (Overview)
`WarpMPMSimulator` 是 PhysTwin 项目的高性能物理后端实现。它利用 NVIDIA Warp 的 JIT 编译能力，在 GPU 上直接执行 MPM 模拟逻辑，同时通过 `wp.Tape` 保持了完整的自动微分能力。

### 核心优势：
- **性能飞跃**：前向/反向速度提升约 15-20 倍。
- **微分闭环**：支持梯度从 Loss 回传至 PyTorch 神经网络（如 ResidualPGND）及物理专家参数。
- **混合本构 (MoE)**：在 Warp Kernel 中原生支持混合专家模型。
- **显存优化**：通过零拷贝 (Zero-copy) 技术共享张量，大幅降低显存占用。

---

## 2. 核心组件 (Core Components)

### 2.1 WarpMPMSimulator (`simulator_warp.py`)
作为 PyTorch `nn.Module` 的封装，管理 Warp 仿真器的生命周期：
- **Reset**: 初始化粒子位置、体积及控制器连接。
- **Step**: 执行一步 MPM 模拟，包括应力计算、网格更新和神经残差（Residual）注入。

### 2.2 MoE Stress Kernel (`moe_utils.py`)
在 Warp 中实现了与 PyTorch 版本一致的混合本构模型：
- 支持 `nh` (Neo-Hookean), `co` (Corotated), `st` (StVK), `fi` (Fiber)。
- 通过 `particle_weights` 动态混合各专家应力。

### 2.3 Controller Interaction (`warp_utils.py`)
在 Warp 中实现了 PD 控制器逻辑：
- 使用 `accumulate_pd_forces_kernel` 处理手与粒子的交互。
- 使用 `wp.atomic_add` 确保并行安全性。

---

## 3. 集成指南 (Integration Guide)

### 3.1 初始化与 Reset
```python
from phys_expert.model.diff_simulator.warp_solver.simulator_warp import WarpMPMSimulator

# 配置加载
simulator = WarpMPMSimulator(cfg.mpm).to("cuda")

# 重置状态 (包含点云填充后的 init_particles 和控制器位置)
simulator.reset(init_particles, controller_pos=controller_pos)