根据 `phys_expert/model/experts/mixture_model.py` 中的代码实现，该项目的混合本构模型（Mixture Constitutive Model）是一个由四个基础专家模型（Neo-Hookean, Corotated, St. Venant-Kirchhoff, Fiber）组成的加权求和模型。

---

### 1. 动机：为什么要提出混合本构模型？ (Motivation)

在现实世界中，可变形物体的物理行为极其复杂且具有**多尺度、非线性**的特征。传统的单一本构模型往往只能涵盖某种特定的物理特性，难以同时模拟复杂材质的多种表现：

1.  **材质特性的多样性**：
    *   **布料**：具有极强的抗拉伸性（纤维特性）但极易弯折（非线性弹性）。
    *   **绳索**：在一维方向上几乎不可压缩和拉伸，但在三维空间内具有复杂的扭转和旋转。
    *   **软体**：既需要保持体积，又需要在大旋转下保持数值稳定。
2.  **单一模型的局限性**：
    *   Neo-Hookean 适合大变形但缺乏方向性；Fiber 具有方向性但无法处理横向压缩；StVK 在大旋转下表现优异但在受压时容易数值崩溃。单一模型无法完美适配“数字孪生”对真实视频中复杂交互的拟合需求。
3.  **System ID 的灵活性需求**：
    *   为了从真实视频中自动识别物理参数，我们需要一个能够涵盖更广搜索空间的物理基底。通过混合本构，模型可以自动学习不同物理专家的比例，从而“进化”出最接近真实的复合材质。

---

### 2. 方法细节 (Methodology: What & How)

#### 混合本构模型通式 (What)

总的第一类 Piola-Kirchhoff 应力张量 $\mathbf{P}_{\text{total}}$ 定义为：

$$
\mathbf{P}_{\text{total}}(\mathbf{F}) = \sum_{k \in \mathcal{K}} w_k \cdot \mathbf{P}_k(\mathbf{F})
$$

其中：
* $\mathbf{F}$ 为变形梯度张量（Deformation Gradient）。
* $\mathcal{K} = \{\text{NH}, \text{Co}, \text{StVK}, \text{Fi}\}$ 为激活的专家集合。
* $w_k$ 为第 $k$ 个专家的权重系数（由模型优化得到，满足 $\sum w_k = 1$）。
* $\mathbf{P}_k(\mathbf{F})$ 为各个专家模型计算出的应力。

#### 物理参数化、分解与插值 (Physical Parameterization & Decomposition)

为了增强物理可解释性并确保数值稳定性，我们采取了以下策略：

1.  **参数映射 (Parameter Mapping)**：所有跨越多个数量级的核心物理量（$E, k_f, \nu$）均通过 **Log-Sigmoid** 映射，确保其在对数空间内均匀分布且始终保持在 $[\Theta_{min}, \Theta_{max}]$ 范围内：
    $$s = \text{Sigmoid}(\Theta_{raw}), \quad \Theta_{phys} = \exp\left( \ln(\Theta_{min}) + s \cdot (\ln(\Theta_{max}) - \ln(\Theta_{min})) \right)$$
    这种映射方式解决了跨量级优化中的梯度饱和问题，使模型能更灵敏地识别不同尺度的物理量。
2.  **权重归一化 (Weight Normalization)**：专家权重 $w_k$ 由可学习的 Logits $\mathbf{l}$ 通过 Softmax 函数计算得到，满足归一化约束：
    $$w_k = \frac{\exp(l_k)}{\sum_{j \in \mathcal{K}} \exp(l_j)}, \quad \sum_{k \in \mathcal{K}} w_k = 1$$
3.  **几何分解 (Geometry Decomposition)**：
    *   **Patch-level Optimization**：为了避免参数爆炸并增强优化的稳定性，我们利用 **最远点采样 (Farthest Point Sampling, FPS)** 将物体分解为 $K$ 个局部区域（Patches）。
    *   **中心点采样**：给定粒子集合 $\mathcal{P}$，我们迭代地选择 $K$ 个参数锚点（中心点） $\mathcal{C} = \{\mathbf{c}_1, \dots, \mathbf{c}_K\}$，满足：
        $$\mathbf{c}_i = \arg\max_{\mathbf{x} \in \mathcal{P}} \left( \min_{j < i} \|\mathbf{x} - \mathbf{c}_j\| \right)$$
    *   **局部参数化**：物理参数 $\Theta$ 和专家权重 $w_k$ 仅在这些中心点 $\mathbf{c}_k$ 上进行优化，从而将待优化参数量从 $\mathcal{O}(N_{particles})$ 降低到 $\mathcal{O}(K)$。
4.  **粒子级插值 (Particle-level Interpolation)**：
    *   利用 **K-最近邻 (K-Nearest Neighbors, KNN, K=3)** 算法寻找每个模拟粒子所属的邻近 Patch。
    *   采用 **反距离加权 (Inverse Distance Weighting, IDW)** 将 Patch 级的物理参数平滑地插值回粒子级，确保物理属性在空间上的连续性：
        $$\Theta_p = \sum_{j=1}^3 \alpha_j \Theta_{patch, j}, \quad \alpha_j = \frac{1/d_j}{\sum 1/d_j}$$
4.  **弹-塑性分层优化 (Elastic-Plastic Staged Optimization)**：
    为了处理现实视频中复杂的非线性行为并提高收敛稳定性，我们设计了一个由浅入深的两阶段优化方案：
    *   **阶段一：建立弹性基础 (Elastic Foundation Phase)**：
        *   **目标**：在不考虑永久形变的前提下，识别材料的基本物理刚度和最优的专家混合比例。
        *   **激活参数**：杨氏模量 ($E$)、泊松比 ($\nu$)、专家权重 ($\mathbf{w}$) 以及纤维参数 ($k_f, \mathbf{d}_0$)。
        *   **物理意义**：此阶段模型专注于拟合物体的“基本骨架”行为，如弯曲、拉伸和体积恢复。通过暂时冻结塑性，避免了优化初期因材料过早“屈服”而导致的弹性模量识别不准。
    *   **阶段二：引入塑性与联合精调 (Plasticity & Joint Fine-tuning Phase)**：
        *   **目标**：在稳定的弹性框架之上，拟合材料的能量耗散行为和残余永久形变。
        *   **激活参数**：正式引入屈服应力 ($\sigma_y$) 和塑性粘度 ($\eta$)，同时保持阶段一的所有参数继续参与优化。
        *   **塑性模型细节 (Viscoplasticity Model)**：
            我们采用基于 **Hencky 应变**（对数应变）的 **Von-Mises 屈服准则**。对于每一个模拟步，系统会计算弹性试探应变并执行**隐式回归映射 (Implicit Return Mapping)**：
            1.  **对数应变分解**：$\mathbf{\epsilon} = \ln(\mathbf{\Sigma})$，其中 $\mathbf{F} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$。
            2.  **偏应变计算**：$\mathbf{\epsilon}_{dev} = \mathbf{\epsilon} - \frac{1}{3}\text{tr}(\mathbf{\epsilon})\mathbf{I}$。
            3.  **屈服判定**：计算试探偏应力 $\mathbf{s}_{tr} = 2\mu \mathbf{\epsilon}_{dev}$，若满足 $\|\mathbf{s}_{tr}\| > \sqrt{2/3} \sigma_y$ 则发生塑性流动。
            4.  **映射回流**：通过下式计算修正后的偏应力幅值：
                $$\|\mathbf{s}_{new}\| = \|\mathbf{s}_{tr}\| - \frac{\|\mathbf{s}_{tr}\| - \sqrt{2/3}\sigma_y}{1 + \frac{\eta}{2\mu_{eff}\Delta t}}$$
            5.  **更新变形梯度**：根据修正后的应变重新构造 $\mathbf{F}_{new}$。
        *   **物理意义**：此阶段模型开始识别材料在剧烈交互下的“不可逆”行为（如布料的永久褶皱、软体的挤压形变）。这种“先稳骨架、再补细节”的策略极大地增强了 System ID 面对复杂真实数据时的数值健壮性。
3.  **空间异质性**：权重 $w_k$ 是基于 Patch 定义的，允许同一物体在不同部位表现出不同的物理特性（例如布料的加固边缘与中间区域）。

---

### 3. 混合本构的核心优势 (Key Benefits)

*   **物理先验的融合**：巧妙地将各向同性弹性（NH/Co/StVK）与各向异性约束（Fiber）结合，使得模型能够同时处理体积保持和方向性加强。
*   **数值稳定性增强**：通过专家间的“优势互补”，避免了单一模型在极端受力（如剧烈挤压或高速拉伸）下的数值奇点问题。
*   **高保真度拟合**：相比单一模型，混合本构极大地提升了模型对真实视频中物体动态（尤其是复杂的褶皱、垂坠和拉伸极限）的表达能力。
*   **自动化的 System ID**：无需人工指定复杂的本构方程，通过梯度下降自动组合出最合适的物理基底，实现了真正的“数据驱动物理”。

---

### 3. 本构模型参数表 (Constitutive Parameters Roster)

每个子专家模型和修正模块所依赖的可优化物理参数如下表所示：

| 模型分类 | 专家名称 (Key) | 核心物理参数 | 符号 | 物理含义 |
| :--- | :--- | :--- | :--- | :--- |
| **各向同性基底** | Neo-Hookean (nh) | 杨氏模量, 泊松比 | $E, \nu$ | 决定物体的整体刚度与体积变化率 |
| | Corotated (co) | 杨氏模量, 泊松比 | $E, \nu$ | 在大旋转下保持线性刚度，防止扭曲 |
| | StVK (st) | 杨氏模量, 泊松比 | $E, \nu$ | 处理小应变大旋转，提供强非线性 |
| **各向异性增强** | Fiber (fi) | 纤维刚度, 纤维方向 | $k_f, \mathbf{d}_0$ | 模拟沿特定方向（如布料经纬线）的抗拉增强 |
| **塑性修正模块** | Viscoplasticity | 屈服应力, 塑性粘度 | $\sigma_y, \eta$ | 决定材料何时发生永久变形及能量耗散速率 |
| **混合策略** | Mixture Model | 专家权重 | $\mathbf{w}$ | 决定上述各专家在空间每个点的贡献比例 |

---

### 4. 应力与变形梯度的关系 (Stress vs. Deformation Gradient)

为了直观理解不同本构模型（专家）在不同载荷下的响应，下图展示了单轴拉伸（Uniaxial Stretch）情况下，第一类 Piola-Kirchhoff 应力 $P$ 随拉伸倍数 $\lambda$ 的变化曲线：

![Constitutive Models Comparison](assets/constitutive_comparison.png)

*图：Neo-Hookean、Corotated、StVK 以及 Anisotropic Fiber 模型在单轴拉伸下的应力-拉伸响应对比。*

**关于坐标轴的说明**：
*   **横轴 Stretch $\lambda$**：代表变形梯度张量 $\mathbf{F}$ 在主方向上的分量（即 $F_{11}$）。
    *   $\lambda = 1.0$：无形变（原长）。
    *   $\lambda > 1.0$：**拉伸 (Extension)**。
    *   $\lambda < 1.0$：**压缩 (Compression)**。
*   **纵轴 $P_{11}$**：第一类 Piola-Kirchhoff 应力在拉伸方向的分量。曲线的斜率 $\frac{\partial P_{11}}{\partial \lambda}$ 即代表了材料在该状态下的切线模量。

---

### 5. 物理参数化与优化变量 (Physical Parameterization)

为了增强物理可解释性，模型并非直接优化拉梅常数 ($\mu, \lambda$)，而是将**杨氏模量 ($E$)** 和 **泊松比 ($\nu$)** 作为核心优化变量。因此，混合应力通式可以显式地写为关于物理参数 $\Theta = \{E, \nu, k_f, \mathbf{w}\}$ 的函数：

$$
\mathbf{P}_{\text{total}}(\mathbf{F}; E, \nu, k_f, \mathbf{w}) = \sum_{k \in \mathcal{K}} w_k \cdot \mathbf{P}_k\left(\mathbf{F}; \mu(E, \nu), \lambda(E, \nu), k_f\right)
$$

其中，显式优化变量集合为 $\Theta = \{E, \nu, k_f, \mathbf{w}, \sigma_y, \eta\}$：
*   $E$：杨氏模量 (Young's Modulus)，控制材料整体拉伸刚度。
*   $\nu$：泊松比 (Poisson's Ratio)，控制材料的横向收缩效应。
*   $k_f$：纤维刚度 (Fiber Stiffness)，控制各向异性增强。
*   $\mathbf{w} = \{w_{\text{NH}}, w_{\text{Co}}, w_{\text{StVK}}, w_{\text{Fi}}\}$：专家权重 (Expert Weights)，决定各本构模型的混合比例。
*   $\sigma_y$：屈服应力 (Yield Stress)，控制塑性变形发生的阈值。
*   $\eta$：塑性粘度 (Plastic Viscosity)，控制塑性流动的速率。

其中，拉梅常数与优化变量的转换关系为：
$$
\mu = \frac{E}{2(1+\nu)}, \quad \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
$$

> **注**：在实际代码实现中，为了保证物理参数的非负性和数值稳定性，优化变量 $E, k_f, \sigma_y, \eta$ 均通过 **Log-Sigmoid** 映射从原始参数空间 ($x_{\text{raw}}$) 映射到物理空间。

---

### 6. 损失函数与目标优化 (Loss Function & Optimization)

System Identification 的目标是通过梯度下降找到一组物理参数 $\Theta$，使得模拟的点云序列 $\mathcal{X}_{sim}$ 在时空上与观测到的真实点云 $\mathcal{X}_{gt}$ 最为接近。总损失函数定义为：

$$
\mathcal{L}(\Theta) = \lambda_{t} \mathcal{L}_{track} + \lambda_{c} \mathcal{L}_{chamfer} + \lambda_{s} \mathcal{L}_{smooth}
$$

其中：
*   **Tracking Loss ($\mathcal{L}_{track}$)**：计算对应粒子间的欧氏距离（仅针对有可靠跟踪的粒子）：
    $$\mathcal{L}_{track} = \sum_{t=1}^T \sum_{i \in \mathcal{P}_{track}} \| \mathbf{x}_{sim, i}^t - \mathbf{x}_{gt, i}^t \|^2$$
*   **Chamfer Loss ($\mathcal{L}_{chamfer}$)**：衡量整体点云形状的相似度（针对所有表面粒子）：
    $$\mathcal{L}_{chamfer}(\mathcal{S}_1, \mathcal{S}_2) = \sum_{x \in \mathcal{S}_1} \min_{y \in \mathcal{S}_2} \|x - y\|^2 + \sum_{y \in \mathcal{S}_2} \min_{x \in \mathcal{S}_1} \|x - y\|^2$$
*   **Smoothness Regularization ($\mathcal{L}_{smooth}$)**：惩罚物理参数在空间上的剧烈突变，保证 Patch 间参数的平滑性。

---

### 各专家模型的应力公式 $\mathbf{P}_k(\mathbf{F})$

#### 1. Neo-Hookean (NH)
基于对数应变的 Neo-Hookean 模型，适用于大变形。

$$
\mathbf{P}_{\text{NH}} = \mu (\mathbf{F} - \mathbf{F}^{-T}) + \lambda \ln(J) \mathbf{F}^{-T}
$$

* $J = \det(\mathbf{F})$
* $\mu, \lambda$ 为拉梅常数（Lame parameters）。

#### 2. Corotated Linear Elasticity (Co)
共旋线性弹性模型，去除旋转分量后应用胡克定律，适用于模拟旋转物体。

$$
\mathbf{P}_{\text{Co}} = 2\mu (\mathbf{F} - \mathbf{R}) + \lambda (J - 1) J \mathbf{F}^{-T}
$$

* $\mathbf{R}$ 为极分解 $\mathbf{F} = \mathbf{R}\mathbf{S}$ 中的旋转矩阵。

#### 3. St. Venant-Kirchhoff (StVK)
基于 Green-Lagrange 应变的经典模型，适用于小应变大旋转。

$$
\mathbf{P}_{\text{StVK}} = \mathbf{F} \mathbf{S} = \mathbf{F} \left( 2\mu \mathbf{E} + \lambda \text{tr}(\mathbf{E}) \mathbf{I} \right)
$$

* $\mathbf{E} = \frac{1}{2}(\mathbf{F}^T \mathbf{F} - \mathbf{I})$ 为 Green-Lagrange 应变张量。
* $\mathbf{S}$ 为第二类 Piola-Kirchhoff 应力。

#### 4. Anisotropic Fiber (Fi)
各向异性纤维模型，仅在纤维方向受拉伸时产生应力（各向异性增强）。

$$
\mathbf{P}_{\text{Fi}} = \begin{cases} 
k \frac{\lambda_f - 1}{\lambda_f} \mathbf{F} (\mathbf{d}_0 \otimes \mathbf{d}_0) & \text{if } \lambda_f > 1 \\
\mathbf{0} & \text{if } \lambda_f \le 1 
\end{cases}
$$

* $k$ 为纤维刚度系数（Fiber Stiffness）。
* $\mathbf{d}_0$ 为初始构型下的纤维方向单位向量。
* $\lambda_f = \|\mathbf{F}\mathbf{d}_0\|$ 为沿纤维方向的伸长率。
* $\otimes$ 表示张量积（外积）。

---

### 塑性修正 (Plasticity Return Mapping)

在计算出弹性试探应力（Elastic Trial Stress）后，还会应用基于 Hencky 应变的 Von-Mises 屈服准则进行塑性修正（Return Mapping）：

$$
\mathbf{P}_{\text{final}} \leftarrow \text{ReturnMapping}(\mathbf{P}_{\text{total}}, \sigma_y, \eta)
$$

* $\sigma_y$ 为屈服应力（Yield Stress）。
* $\eta$ 为塑性粘度（Plastic Viscosity）。

---

### 附录：关于应力符号 $\mathbf{P}$ vs $\sigma$ 的说明

在撰写论文时，需明确区分 **第一类 Piola-Kirchhoff 应力 ($\mathbf{P}$)** 与 **柯西应力 ($\sigma$, Cauchy Stress)**：

1. **定义区别**：
   * **$\mathbf{P}$ (First Piola-Kirchhoff Stress)**：描述**当前构型下的力**作用在**参考构型（初始形状）的面积**上。本构模型通常通过对变形梯度 $\mathbf{F}$ 求导得到（$\mathbf{P} = \frac{\partial \Psi}{\partial \mathbf{F}}$），因此上述本构公式均使用 $\mathbf{P}$。
   * **$\sigma$ (Cauchy Stress)**：描述**当前构型下的力**作用在**当前构型（变形后）的面积**上，代表材料内部真实的物理受力状态（真应力）。

2. **数学关系**：
   $$
   \sigma = \frac{1}{J} \mathbf{P} \mathbf{F}^T
   $$
   其中 $J = \det(\mathbf{F})$。

3. **写作建议**：
   * **描述本构方程时**：**必须使用 $\mathbf{P}$**。因为它是 $\mathbf{F}$ 的直接函数，数学形式更简洁且符合超弹性势能的推导逻辑。如果你将其写为 $\sigma$，则公式右边需要额外乘以 $\frac{1}{J}\mathbf{F}^T$，这会使公式变得冗余且不直观。
   * **展示结果/云图时**：**建议使用 $\sigma$**（或基于 $\sigma$ 计算的 Von-Mises 应力）。因为这反映了材料在当前时刻真实的受力强度，符合物理直觉。
