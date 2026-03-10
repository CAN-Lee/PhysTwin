# MeshSequence: 基于单目视频的 4D Mesh 序列一致性重建

# (MeshSequence: Consistency-aware 4D Mesh Reconstruction from Monocular Videos)

## 1. 问题背景 (Problem Statement)

在从单目视频进行 3D 重建的任务中，虽然可以利用现有的 **Image-to-3D** 模型（如 One-2-3-45, MeshLRM 等）对每一帧视频独立生成 3D Mesh，但存在以下核心痛点：

* **拓扑不一致性：** 每一帧生成的 Mesh 顶点数量、连接关系（Faces）各不相同。
* **缺乏对应关系 (Correspondence)：** 不同帧之间的顶点没有时间上的逻辑关联，无法进行纹理追踪、运动分析或下游动画处理。
* **时间不连续：** 独立生成的 Mesh 在时间维度上存在严重的闪烁（Flickering）和几何突变。

## 2. 核心目标 (Research Objective)

给定单目视频序列，通过一个参考网格（$\mathcal{M}_{can}$，通常由首帧生成）和预计算的运动信息，重建出一组具有统一拓扑结构且顶点对齐的 4D Mesh 序列 $\{\mathcal{M}_t\}_{t=0}^T$。

---

## 3. 技术路线 (Methodology)

本方法采用 **参考网格变形 (Template-based Deformation)** 的策略，利用 **SE(3) 运动聚类场** 驱动 Canonical Mesh 进行非刚性形变。

### 3.1 初始表示 (Initialization)

* **Canonical Mesh ($\mathcal{M}_{can}$):** 对视频首帧进行高质量 Image-to-3D 重建（或其他来源），获得基础几何拓扑。若无外部网格，默认使用视频首帧生成的点云作为 canonical 表达。
* **SE(3) 运动场预计算:** 利用 Procrustes Alignment 对密集追踪的点轨迹进行聚类与配准。
    1. **运动聚类:** 对所有点的 3D 轨迹使用 K-means 聚类，将运动相似的点归为同一簇 $k$。
    2. **刚性配准 (Procrustes):** 对于每一簇，计算从 Canonical frame $\mathcal{M}_{can}$ 到每一帧 $t$ 的最优 SE(3) 变换 $\mathbf{T}_{k,t}$。

### 3.2 变形驱动机制 (Deformation Mechanism)

为了实现平滑且保物理特性的变形，我们采用**神经蒙皮 (Neural Skinning)** 结合 **SE(3) 场** 的方案：

1. **权重分配 (Skinning Weights Assignment):**
对于 $\mathcal{M}_{can}$ 中的每一个顶点 $v_i$，计算其相对于 $K$ 个运动聚类的隶属度权重 $w_{i,k}$。
    * *实现方式：* 使用基于距离的 Softmax 函数或 MLP 预测权重，满足 $\sum_{k=1}^K w_{i,k} = 1$。


3. **运动混合 (Linear Blend Skinning, LBS):**
利用预计算的 $SE(3)$ 矩阵对顶点坐标进行变换：
$$v_{i,t} = \sum_{k=1}^{K} w_{i,k} \cdot (\mathbf{R}_{k,t} v_{i,can} + \mathbf{t}_{k,t})$$

3. **双四元数插值 (Dual Quaternion Skinning, 可选优化):**
若运动包含大幅度旋转，采用 DQS 替代 LBS 以避免关节处的“糖纸效应” (Volume loss)，确保变形后的几何体积感。

### 3.3 优化与约束 (Loss Functions)

为了弥补单目估计的误差，通过以下 Loss 对变形过程进行微调：

* **Chamfer Distance Loss:** 使变形后的 Mesh 表面尽可能贴合该帧独立生成的原始 Point Cloud/Mesh。
* **ARAP Regularization (As-Rigid-As-Possible):** 约束局部顶点的旋转一致性，防止网格出现病态拉伸。
* **Temporal Smoothness:** 对  变换矩阵施加一阶或二阶平滑约束，消除运动抖动。

---

## 4. 方案优势 (Advantages)

* **天然一致性：** 所有帧共享同一套顶点索引，解决了 Correspondence 问题。
* **鲁棒性高：** 相比于直接预测每个顶点的 $\Delta v$（位移），$SE(3)$ 聚类场提供了更强的几何先验，能够处理大幅度的刚性与半刚性运动。
* **端到端潜力：** 该框架可进一步集成到微分渲染器（如 PyTorch3D, GSplat）中，利用图像颜色损失进行反向优化。

---

## 5. 后续实验规划 (Next Steps)

1. [ ] **可视化聚类场：** 验证预计算的 SE(3) 聚类是否与物体的语义结构（如手臂、躯干）吻合。
2. [ ] **消融实验：** 对比“直接位移预测”与“SE(3) 场驱动”在长序列重建中的漂移量。
3. [ ] **渲染评估：** 检查变形后的 Mesh 在重投影后的 Mask IoU 和 PSNR。

---