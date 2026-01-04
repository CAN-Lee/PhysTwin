# XPBD (Extended Position Based Dynamics) 理论推导

本文档从零推导 XPBD (Extended Position Based Dynamics) 的核心公式。XPBD 是对传统 PBD 的扩展，解决了 PBD 刚度依赖于时间步长 ($\Delta t$) 和迭代次数的问题，并引入了 "Compliance" (柔度) 的概念，使其具有严格的物理意义。

## 1. 背景：隐式欧拉积分与变分原理

大多数现代物理模拟是从牛顿第二定律出发的：
$$ M \ddot{x} = f_{ext} + f_{int}(x) $$

使用**隐式欧拉积分 (Implicit Euler Integration)** 进行离散化：
$$ x_{n+1} = x_n + \Delta t v_{n+1} $$
$$ v_{n+1} = v_n + \Delta t M^{-1} (f_{ext} + f_{int}(x_{n+1})) $$

这是一个非线性方程组。我们可以将其转化为一个**能量最小化问题 (Optimization Problem)**。
隐式欧拉积分等价于寻找 $x_{n+1}$，使得以下目标函数最小：

$$ x_{n+1} = \underset{x}{\text{argmin}} \left( \frac{1}{2} (x - \tilde{x})^T M (x - \tilde{x}) + \Delta t^2 U(x) \right) $$

其中：
*   $M$ 是质量矩阵。
*   $\tilde{x} = x_n + \Delta t v_n + \Delta t^2 M^{-1} f_{ext}$ 是仅考虑外力时的**预测位置 (Predicted Position)**。
*   $U(x)$ 是势能函数（如弹簧势能、重力势能等）。

这个公式的物理直觉是：**物体的新位置应该尽可能接近惯性预测位置 $\tilde{x}$，同时也要使系统的势能 $U(x)$ 尽可能小。**

## 2. 引入约束 (Constraints) 与 柔度 (Compliance)

在 XPBD 中，所有的相互作用（如弹簧、碰撞、体积维持）都被描述为**约束** $C(x)$。

### 软约束 (Soft Constraints)
对于一个约束 $C(x) = 0$，我们定义其势能为弹性势能：
$$ U(x) = \frac{1}{2} k C(x)^2 $$
其中 $k$ 是刚度 (Stiffness)。

为了处理无限刚度（刚体）的情况，XPBD 引入了 **柔度 (Compliance)** $\alpha = \frac{1}{k}$。
当 $\alpha = 0$ 时，代表无限刚度（硬约束）；当 $\alpha > 0$ 时，代表软约束。

此时势能写作：
$$ U(x) = \frac{1}{2\alpha} C(x)^2 $$

代入之前的最小化目标函数：
$$ E(x) = \frac{1}{2} (x - \tilde{x})^T M (x - \tilde{x}) + \sum_j \frac{\Delta t^2}{2\alpha_j} C_j(x)^2 $$

## 3. XPBD 核心推导

我们的目标是求 $E(x)$ 的极值。为了求解这个问题，XPBD 引入了拉格朗日乘子 $\lambda$。

### 为什么要引入拉格朗日乘子？

你可能会问，为什么不直接对 $E(x)$ 关于 $x$ 求导来解呢？引入 $\lambda$ 主要有三个原因：

1.  **统一硬约束和软约束**：
    *   如果直接优化 $E(x)$，当 $\alpha \to 0$ (刚体) 时，势能项 $\frac{1}{2\tilde{\alpha}} C(x)^2$ 会趋向无穷大，导致数值计算溢出或不稳定（刚度矩阵病态）。
    *   对于硬约束 ($\alpha=0$)，问题变成了带约束优化：$\min \frac{1}{2} (x - \tilde{x})^T M (x - \tilde{x}) \quad \text{s.t.} \quad C(x) = 0$。解决这类问题的标准数学方法就是**拉格朗日乘子法**。
    *   通过引入 $\lambda$，我们可以构建一个统一的数学框架，同时处理 $\alpha=0$ 和 $\alpha>0$ 的情况，而不会出现除以零的问题。

2.  **物理意义清晰**：
    *   拉格朗日乘子 $\lambda$ 在物理上直接对应于**约束力** (Constraint Force)。
    *   在软约束中，$\lambda \propto -C(x) / \tilde{\alpha}$，这正是胡克定律 $F = -k \Delta x$ 的体现。引入 $\lambda$ 让我们能显式地计算和控制物体受到的力。

3.  **便于求解**：
    *   引入 $\lambda$ 后，我们可以把非线性的位置优化问题转化为关于 $\lambda$ 的求根问题，这通常更容易通过牛顿法进行局部线性化求解。

### 推导过程

为了简化推导，我们考虑单个约束 $C(x)$。
基于上述理由，我们构造如下的方程组来描述最优解条件。
XPBD 论文证明，原优化问题等价于求解以下对偶形式的鞍点：

$$ \mathcal{L}(x, \lambda) = \frac{1}{2}(x - \tilde{x})^T M (x - \tilde{x}) + \lambda C(x) - \frac{1}{2} \tilde{\alpha} \lambda^2 $$

分别对 $x$ 和 $\lambda$ 求导并令其为 0：

1.  **对 $x$ 求导 (力的平衡)**：
    $$ \frac{\partial \mathcal{L}}{\partial x} = M(x - \tilde{x}) + \nabla C(x)^T \lambda = 0 $$
    $$ \implies x = \tilde{x} - M^{-1} \nabla C(x)^T \lambda $$
    *(注：这里 $\lambda$ 的符号定义可能因文献而异，XPBD通常取反向定义以便后续推导，记作 $x = \tilde{x} + M^{-1} \nabla C(x)^T \lambda$)*

2.  **对 $\lambda$ 求导 (约束方程)**：
    $$ \frac{\partial \mathcal{L}}{\partial \lambda} = C(x) - \tilde{\alpha} \lambda = 0 $$
    这里同样为了符合 XPBD 的标准形式 (Constraint Compliance)，通常写作：
    $$ C(x) + \tilde{\alpha} \lambda = 0 $$
    其中 $\tilde{\alpha} = \frac{\alpha}{\Delta t^2}$ 是时间步长归一化的柔度。

这个方程组 $C(x) + \tilde{\alpha} \lambda = 0$ 极其重要：
*   当 $\tilde{\alpha} = 0$ 时，退化为 $C(x) = 0$ (硬约束)。
*   当 $\tilde{\alpha} > 0$ 时，$\lambda = -C(x)/\tilde{\alpha}$，即力与形变成正比 (软约束)。

接下来的任务就是求解这个方程组。我们将位置更新公式代入约束方程...
其中 $\tilde{\alpha} = \frac{\alpha}{\Delta t^2}$ 是这种约束特有的**时间步长归一化柔度**。这使得 XPBD 的刚度表现与 $\Delta t$ 无关。

同时，力的平衡方程（梯度为0）给出：
$$ M(x - \tilde{x}) - \nabla C(x)^T \lambda = 0 $$
这里 $\nabla C(x)^T \lambda$ 实际上就是约束产生的力（冲量形式）。

由此得到位置更新公式：
$$ x = \tilde{x} + M^{-1} \nabla C(x)^T \lambda $$

### 求解 $\lambda$ (牛顿迭代)

我们将位置更新公式代入约束方程 $C(x) + \tilde{\alpha} \lambda = 0$ 中。
由于 $C(x)$ 是非线性的，我们对其在当前位置 $x_i$ 处进行一阶泰勒展开：

$$ C(x_{new}) \approx C(x_i) + \nabla C(x_i) \cdot \Delta x = 0 $$

我们要找一个修正量 $\Delta \lambda$，更新 $\lambda_{new} = \lambda_{old} + \Delta \lambda$。
对应的位置修正量为 $\Delta x = M^{-1} \nabla C(x)^T \Delta \lambda$。

代入约束方程的一阶近似：
$$ C(x_i) + \nabla C(x_i) \cdot (M^{-1} \nabla C(x_i)^T \Delta \lambda) + \tilde{\alpha} (\lambda_{old} + \Delta \lambda) = 0 $$

整理该式，求解 $\Delta \lambda$：

$$ \underbrace{C(x_i) + \tilde{\alpha} \lambda_{old}}_{\text{违反量}} + \underbrace{(\nabla C M^{-1} \nabla C^T + \tilde{\alpha})}_{\text{分母}} \Delta \lambda = 0 $$

于是得到 XPBD 的核心更新公式：

$$ \Delta \lambda = \frac{-C(x_i) - \tilde{\alpha} \lambda_{old}}{\nabla C(x_i) M^{-1} \nabla C(x_i)^T + \tilde{\alpha}} $$

得到 $\Delta \lambda$ 后，我们可以更新位置：

$$ \Delta x = M^{-1} \nabla C(x_i)^T \Delta \lambda $$

## 4. 算法流程 (Algorithm)

XPBD 的单步模拟流程如下：

1.  **预测 (Predict)**:
    $$ \tilde{x} = x_n + \Delta t v_n + \Delta t^2 M^{-1} f_{ext} $$
    初始化 $\lambda = 0$ (或者使用上一帧的 $\lambda$ 进行热启动 Warm Start)。

2.  **约束求解循环 (Constraint Solve Loop)**:
    对于每个约束 $j$ (迭代 $N$ 次):
    *   计算约束值 $C(x)$ 和梯度 $\nabla C(x)$。
    *   计算广义逆质量 $w = \nabla C M^{-1} \nabla C^T$。
    *   计算乘子增量：
        $$ \Delta \lambda = \frac{-C(x) - \tilde{\alpha} \lambda}{w + \tilde{\alpha}} $$
        *(注意：代码中通常只存储累积的 $\lambda$，或者每次迭代重置，具体取决于实现。Standard XPBD 更新的是累积的 $\lambda$)*
    *   更新位置：
        $$ \Delta x = M^{-1} \nabla C^T \Delta \lambda $$
        $$ x \leftarrow x + \Delta x $$
    *   更新累积乘子：
        $$ \lambda \leftarrow \lambda + \Delta \lambda $$

3.  **速度更新 (Velocity Update)**:
    $$ v_{n+1} = \frac{x_{final} - x_n}{\Delta t} $$

## 5. 关键参数总结

*   **Compliance ($\alpha$)**: 物理材质参数，$\alpha = 1/k$。单位是 $m/N$。
    *   $\alpha = 0$: 无限硬 (刚体约束)。
    *   $\alpha > 0$: 软体 (如弹簧、橡胶)。
*   **Time-step normalized Compliance ($\tilde{\alpha}$)**:
    $$ \tilde{\alpha} = \frac{\alpha}{\Delta t^2} $$
    这是保证模拟结果不随 $\Delta t$ 变化的关键。

## 6. 与可微物理 (Differentiable Physics) 的联系

在 Differentiable XPBD 中，我们希望对初始状态或参数（如 $\alpha$）求导。
由于 XPBD 的每一步都是可微的代数运算（加减乘除），我们可以通过自动微分 (Automatic Differentiation) 链式法则，计算 Loss 对 $\alpha$ 的梯度：

$$ \frac{\partial L}{\partial \alpha} = \frac{\partial L}{\partial x_{final}} \cdot \frac{\partial x_{final}}{\partial \dots} \cdot \frac{\partial \dots}{\partial \alpha} $$

这就是 `warp` 框架在幕后所做的事情。当我们定义一个 `wp.kernel` 并开启 `requires_grad=True` 时，它会记录上述的前向计算图 (Computation Graph)，并在反向传播时计算梯度。

