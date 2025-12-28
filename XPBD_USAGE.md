# XPBD 系统使用说明

## 概述

项目现在支持两种物理模拟系统：
1. **SpringMassSystemWarp** - 原有的质量-弹簧系统（显式力计算）
2. **XPBDSystemWarp** - 新的可微分 XPBD 系统（基于约束求解）

两个系统保持完全相同的接口，可以无缝替换使用。

## XPBD 系统特点

- **更稳定**：使用约束求解而不是显式力计算，数值稳定性更好
- **可微分**：完全支持自动微分，可用于梯度优化
- **兼容性**：与 SpringMassSystemWarp 接口完全兼容
- **扩展性**：支持距离约束、弯曲约束和体积约束

## 使用方法

### 1. 导入系统

```python
from qqtt.model.diff_simulator import SpringMassSystemWarp, XPBDSystemWarp

# 使用原有系统
simulator = SpringMassSystemWarp(...)

# 或使用新的 XPBD 系统
simulator = XPBDSystemWarp(...)
```

### 2. 基础参数说明

XPBDSystemWarp 的参数与 SpringMassSystemWarp 完全相同，额外支持：

- `xpbd_iterations` (默认=1): XPBD 约束求解的迭代次数。增加迭代次数可以提高稳定性，但会增加计算时间。

### 3. 参数转换

XPBD 系统内部会将 `spring_Y`（刚度参数）自动转换为 `compliance`（柔度参数）：
- `compliance = 1 / stiffness`
- `stiffness = exp(spring_Y)`

这意味着：
- 较大的 `spring_Y` → 较小的 `compliance` → 更硬的约束
- 较小的 `spring_Y` → 较大的 `compliance` → 更软的约束

### 4. 在训练代码中使用

在 `trainer_warp.py` 或 `cma_optimize_warp.py` 中，只需替换导入：

```python
# 原来
from qqtt.model.diff_simulator import SpringMassSystemWarp
self.simulator = SpringMassSystemWarp(...)

# 改为
from qqtt.model.diff_simulator import XPBDSystemWarp
self.simulator = XPBDSystemWarp(...)
```

## 约束类型

XPBD系统支持三种约束类型：

### 1. 距离约束（Distance Constraints）- 默认启用

- **用途**：模拟拉伸/压缩，所有弹簧连接都使用距离约束
- **参数**：通过 `spring_Y` 控制刚度

### 2. 弯曲约束（Bending Constraints）- 可选

- **用途**：防止布料过度弯曲，产生更自然的褶皱
- **适用对象**：布料、纸张等薄片状物体
- **参数**：
  - `bending_edges`: (n_bending, 4) - 共享边的两个三角形顶点索引 [i0, i1, i2, i3]
  - `rest_bending_angles`: (n_bending,) - 初始二面角（弧度）
  - `bending_compliance`: float 或 (n_bending,) - 弯曲柔度，默认 1e-6

**生成弯曲约束示例**：
```python
import numpy as np

def generate_bending_constraints(vertices, faces):
    """从三角形网格生成弯曲约束"""
    bending_edges = []
    rest_angles = []
    
    # 构建边到面的映射
    edge_to_faces = {}
    for i, face in enumerate(faces):
        for j in range(3):
            edge = tuple(sorted([face[j], face[(j+1)%3]]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append((i, face[(j+2)%3]))
    
    # 找到共享边的三角形对
    for edge, face_info in edge_to_faces.items():
        if len(face_info) == 2:
            i0, i1 = edge
            _, i2 = face_info[0]
            _, i3 = face_info[1]
            
            # 计算初始二面角
            e0 = vertices[i1] - vertices[i0]
            n1 = np.cross(e0, vertices[i2] - vertices[i0])
            n2 = np.cross(e0, vertices[i3] - vertices[i0])
            n1 = n1 / (np.linalg.norm(n1) + 1e-6)
            n2 = n2 / (np.linalg.norm(n2) + 1e-6)
            cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            bending_edges.append([i0, i1, i2, i3])
            rest_angles.append(angle)
    
    return np.array(bending_edges), np.array(rest_angles)
```

### 3. 体积约束（Volume Constraints）- 可选

- **用途**：保持3D物体的体积，防止过度压缩或膨胀
- **适用对象**：3D实体物体（如毛绒玩具、肌肉）
- **参数**：
  - `tetrahedra`: (n_tets, 4) - 四面体顶点索引 [i0, i1, i2, i3]
  - `rest_volumes`: (n_tets,) - 初始体积
  - `volume_compliance`: float 或 (n_tets,) - 体积柔度，默认 1e-6

**生成体积约束示例**：
```python
from scipy.spatial import Delaunay

def generate_volume_constraints(vertices):
    """从点云生成四面体网格和体积约束"""
    tri = Delaunay(vertices)
    tetrahedra = tri.simplices
    
    rest_volumes = []
    for tet in tetrahedra:
        x0, x1, x2, x3 = vertices[tet]
        e1 = x1 - x0
        e2 = x2 - x0
        e3 = x3 - x0
        volume = np.abs(np.dot(e1, np.cross(e2, e3))) / 6.0
        rest_volumes.append(volume)
    
    return tetrahedra, np.array(rest_volumes)
```

## 使用示例

### 基础使用（只有距离约束）

```python
simulator = XPBDSystemWarp(
    init_vertices=vertices,
    init_springs=springs,
    init_rest_lengths=rest_lengths,
    init_masses=masses,
    dt=0.01,
    num_substeps=10,
    spring_Y=10000.0,
    collide_elas=0.7,
    collide_fric=0.3,
    dashpot_damping=0.1,
    drag_damping=0.1,
    xpbd_iterations=2,
)
```

### 添加弯曲约束（用于布料）

```python
bending_edges, rest_angles = generate_bending_constraints(vertices, faces)

simulator = XPBDSystemWarp(
    # ... 基础参数 ...
    bending_edges=bending_edges,
    rest_bending_angles=rest_angles,
    bending_compliance=1e-6,  # 较硬的弯曲
)
```

### 添加体积约束（用于3D实体）

```python
tetrahedra, rest_volumes = generate_volume_constraints(vertices)

simulator = XPBDSystemWarp(
    # ... 基础参数 ...
    tetrahedra=tetrahedra,
    rest_volumes=rest_volumes,
    volume_compliance=1e-6,  # 接近不可压缩
)
```

### 组合使用

```python
simulator = XPBDSystemWarp(
    # ... 基础参数 ...
    # 弯曲约束（用于布料）
    bending_edges=bending_edges,
    rest_bending_angles=rest_angles,
    bending_compliance=1e-5,
    # 体积约束（用于3D实体）
    tetrahedra=tetrahedra,
    rest_volumes=rest_volumes,
    volume_compliance=1e-6,
)
```

## 可优化的变量

XPBD系统中所有可优化的变量（`requires_grad=True`）如下：

| 变量名 | 维度 | 是否总是优化 | 说明 |
|--------|------|--------------|------|
| `wp_spring_Y` | (n_springs,) | ✅ 是 | 弹簧刚度（对数空间），`stiffness = exp(spring_Y)` |
| `wp_compliance` | (n_springs,) | ✅ 是 | 距离约束柔度，`compliance = 1 / stiffness`，从 `spring_Y` 自动转换 |
| `wp_bending_compliance` | (n_bending,) | ⚠️ 可选 | 弯曲约束柔度，需提供 `bending_edges` 参数 |
| `wp_volume_compliance` | (n_volume,) | ⚠️ 可选 | 体积约束柔度，需提供 `tetrahedra` 参数 |
| `wp_collide_elas` | (1,) | ⚠️ 条件 | 地面碰撞弹性系数，需 `cfg.collision_learn=True` |
| `wp_collide_fric` | (1,) | ⚠️ 条件 | 地面碰撞摩擦系数，需 `cfg.collision_learn=True` |
| `wp_collide_object_elas` | (1,) | ⚠️ 条件 | 对象间碰撞弹性系数，需 `cfg.collision_learn=True` |
| `wp_collide_object_fric` | (1,) | ⚠️ 条件 | 对象间碰撞摩擦系数，需 `cfg.collision_learn=True` |

### 说明

- **总是优化**：这些参数在每次创建 XPBDSystemWarp 时都会设置为可优化
- **可选优化**：只有在提供相应约束参数（如 `bending_edges` 或 `tetrahedra`）时才会创建并优化
- **条件优化**：根据配置 `cfg.collision_learn` 决定是否优化，默认 `True`

### 优化策略建议

- **密集参数**：`wp_spring_Y` / `wp_compliance`（每个弹簧一个参数，通常有数千个）适合**一阶优化**（梯度下降）
- **稀疏参数**：碰撞参数（4个标量）适合**零阶优化**（CMA-ES）
- **可选参数**：弯曲和体积约束参数根据是否启用而定，如果启用，建议使用梯度优化

### 注意

- 状态变量（`wp_x`, `wp_v`, `wp_lambda` 等）虽然 `requires_grad=True`，但它们是中间状态，不是优化目标
- 损失变量（`loss`, `chamfer_loss` 等）是输出，不是优化目标
- 真正的优化参数是上述物理参数，通过梯度下降或 CMA-ES 等方法优化

## 参数调优建议

### 弯曲柔度（bending_compliance）
- **1e-8 ~ 1e-7**: 非常硬的弯曲（如硬纸板）
- **1e-6 ~ 1e-5**: 中等弯曲（如普通布料）
- **1e-4 ~ 1e-3**: 软弯曲（如丝绸、薄纱）

### 体积柔度（volume_compliance）
- **1e-8 ~ 1e-7**: 接近不可压缩（如橡胶、生物组织）
- **1e-6 ~ 1e-5**: 硬体积约束（如泡沫）
- **1e-4 ~ 1e-3**: 软体积约束（如气体）

### 迭代次数（xpbd_iterations）
- **1**: 简单场景，快速计算
- **2-3**: 复杂场景，提高稳定性

## 实现细节

### XPBD 算法流程

1. **预测阶段**：根据当前速度和重力预测新位置
2. **约束求解**：迭代求解距离约束、弯曲约束、体积约束，修正位置
3. **碰撞处理**：处理对象间碰撞和地面碰撞
4. **速度更新**：从修正后的位置更新速度

### 核心 Kernel

- `xpbd_predict_positions`: 预测位置和速度
- `xpbd_distance_constraint`: 距离约束求解（可微分）
- `xpbd_bending_constraint`: 弯曲约束求解（可微分）
- `xpbd_volume_constraint`: 体积约束求解（可微分）
- `xpbd_update_velocities`: 更新速度

### 复用的功能

XPBD 系统复用了原有系统的以下功能：
- 碰撞检测和处理
- 损失计算（chamfer loss, track loss, acc loss）
- 控制点管理
- CUDA graph 优化

## 注意事项

1. **参数兼容性**：`spring_Y` 参数仍然使用，但内部转换为 `compliance`
2. **dashpot_damping**：保留用于兼容性，但 XPBD 主要通过 `compliance` 控制阻尼
3. **数值稳定性**：XPBD 通常比显式力计算更稳定，但需要适当的 `compliance` 范围（建议 >= 1e-8）
4. **性能影响**：弯曲约束和体积约束会增加计算开销，约束数量越多，计算时间越长
5. **网格质量**：弯曲约束需要良好的三角形网格，体积约束需要良好的四面体网格
6. **兼容性**：如果不提供约束参数，系统会退化为只有距离约束，与原有接口完全兼容
