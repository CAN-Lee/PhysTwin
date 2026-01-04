"""
XPBD (Extended Position Based Dynamics) 可微分物理模拟系统

本模块实现了基于XPBD的可微分物理模拟器，支持以下约束类型：
1. 距离约束（Distance Constraints）- 默认启用，模拟拉伸/压缩
2. 弯曲约束（Bending Constraints）- 可选，防止布料过度弯曲
3. 体积约束（Volume Constraints）- 可选，保持3D物体体积

主要特性：
- 完全可微分，支持自动微分和梯度优化
- 与SpringMassSystemWarp接口完全兼容
- 数值稳定性优于显式力计算方法
- 支持CUDA加速和CUDA graph优化

使用示例：
    # 基础使用（只有距离约束）
    simulator = XPBDSystemWarp(
        init_vertices=vertices,
        init_springs=springs,
        init_rest_lengths=rest_lengths,
        init_masses=masses,
        dt=0.01,
        num_substeps=10,
        spring_Y=10000.0,
        ...
    )
    
    # 添加弯曲约束（用于布料）
    simulator = XPBDSystemWarp(
        ...,
        bending_edges=bending_edges,  # (n_bending, 4)
        rest_bending_angles=rest_angles,  # (n_bending,)
        bending_compliance=1e-6,
    )
    
    # 添加体积约束（用于3D实体）
    simulator = XPBDSystemWarp(
        ...,
        tetrahedra=tetrahedra,  # (n_tets, 4)
        rest_volumes=rest_volumes,  # (n_tets,)
        volume_compliance=1e-6,
    )

参数说明：
- spring_Y: 弹簧刚度（对数空间），内部转换为compliance
- xpbd_iterations: 约束求解迭代次数，默认1，复杂场景建议2-3
- bending_compliance: 弯曲柔度，1e-6~1e-5为中等弯曲，1e-4~1e-3为软弯曲
- volume_compliance: 体积柔度，1e-6~1e-5为硬体积约束，1e-4~1e-3为软体积约束

详细使用说明请参考 XPBD_USAGE.md
"""

import torch
from qqtt.utils import logger, cfg
import warp as wp

wp.init()
wp.set_device("cuda:0")
if not cfg.use_graph:
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verify_autograd_array_access = True

# Import shared kernels from spring_mass_warp
# XPBD系统复用mass-spring系统的辅助kernel，避免重复实现
from .spring_mass_warp import (
    # 数据复制kernel（enable_backward=False，不需要梯度）
    copy_vec3,          # 复制vec3数组（用于复制位置、速度等3D向量）
    copy_int,           # 复制int32数组（用于复制可见性、mask等整数数据）
    copy_float,         # 复制float32数组（用于复制参数值）
    
    # 控制点管理
    set_control_points, # 在子步之间插值控制点位置（用于驱动物体运动）
    
    # 循环辅助（用于碰撞检测）
    loop,               # 循环辅助函数（用于遍历碰撞对）
    
    # 碰撞检测和处理
    update_potential_collision,  # 使用空间哈希查找潜在碰撞对（离散操作，不可微）
    object_collision,            # 处理对象间碰撞（可微分，支持弹性/摩擦参数优化）
    integrate_ground_collision,  # 处理地面碰撞并更新位置/速度（可微分）
    
    # 损失计算辅助（enable_backward=False，离散操作）
    compute_distances,      # 计算预测点与真实点的距离矩阵（用于Chamfer loss）
    compute_neigh_indices,   # 查找最近邻索引（离散操作，不可微）
    
    # 损失计算kernel（可微分，用于反向传播）
    compute_chamfer_loss,   # 计算Chamfer距离损失（预测点云与真实点云的匹配误差）
    compute_track_loss,     # 计算跟踪损失（Smooth L1，跟踪点的位置误差）
    compute_acc_loss,       # 计算加速度损失（Smooth L1，加速度平滑性约束）
    compute_final_loss,     # 合并所有损失：loss = chamfer_loss + track_loss + acc_loss
    compute_simple_loss,     # 简单损失（用于synthetic数据，L2距离）
    
    # 辅助工具
    set_int,    # 设置整数标量（用于设置标志位）
    update_acc, # 更新加速度（计算速度差，用于acc_loss）
)


class XPBDState:
    """XPBD状态类，扩展了State以支持XPBD约束"""
    def __init__(self, wp_init_vertices, num_control_points, n_springs, n_bending=0, n_volume=0):
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_x_pred = wp.zeros_like(wp_init_vertices, requires_grad=True)  # XPBD预测位置
        self.wp_x_corrected = wp.zeros_like(wp_init_vertices, requires_grad=True)  # 约束修正后的位置
        self.wp_v_before_collision = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_ground = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=True)
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)
        # XPBD约束拉格朗日乘数
        self.wp_lambda = wp.zeros(
            (n_springs,), dtype=wp.float32, requires_grad=True
        )
        # 弯曲约束拉格朗日乘数
        if n_bending > 0:
            self.wp_lambda_bend = wp.zeros(
                (n_bending,), dtype=wp.float32, requires_grad=True
            )
        else:
            self.wp_lambda_bend = None
        # 体积约束拉格朗日乘数
        if n_volume > 0:
            self.wp_lambda_vol = wp.zeros(
                (n_volume,), dtype=wp.float32, requires_grad=True
            )
        else:
            self.wp_lambda_vol = None

    def clear_forces(self):
        self.wp_vertice_forces.zero_()
    
    def clear_lambda(self):
        """重置拉格朗日乘数"""
        self.wp_lambda.zero_()
        if self.wp_lambda_bend is not None:
            self.wp_lambda_bend.zero_()
        if self.wp_lambda_vol is not None:
            self.wp_lambda_vol.zero_()

    @property
    def requires_grad(self):
        """Indicates whether the state arrays have gradient computation enabled."""
        return self.wp_x.requires_grad


@wp.kernel
def xpbd_predict_positions(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    x_pred: wp.array(dtype=wp.vec3),
    v_pred: wp.array(dtype=wp.vec3),
):
    """XPBD第一步：预测位置和速度（考虑重力和阻尼）"""
    tid = wp.tid()
    
    # 应用重力和阻尼
    gravity = wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    drag_factor = wp.exp(-dt * drag_damping)
    
    # 预测速度（考虑阻尼）
    v_pred[tid] = v[tid] * drag_factor + gravity * dt
    
    # 预测位置
    x_pred[tid] = x[tid] + v_pred[tid] * dt


@wp.kernel
def xpbd_distance_constraint(
    x_pred: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    num_object_points: int,
    springs: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    compliance: wp.array(dtype=float),  # 柔度 = 1/stiffness
    masses: wp.array(dtype=wp.float32),
    dt: float,
    lambda_array: wp.array(dtype=float),
    x_corrected: wp.array(dtype=wp.vec3),
):
    """XPBD距离约束求解（可微分）"""
    tid = wp.tid()
    
    idx1 = springs[tid][0]
    idx2 = springs[tid][1]
    
    # 获取位置
    if idx1 >= num_object_points:
        x1 = control_x[idx1 - num_object_points]
        m1 = 1.0  # 控制点质量设为1（或使用实际质量）
    else:
        x1 = x_pred[idx1]
        m1 = masses[idx1]
    
    if idx2 >= num_object_points:
        x2 = control_x[idx2 - num_object_points]
        m2 = 1.0
    else:
        x2 = x_pred[idx2]
        m2 = masses[idx2]
    
    # 计算当前距离
    delta = x2 - x1
    distance = wp.length(delta)
    rest_length = rest_lengths[tid]
    
    # 约束值 C = |x2 - x1| - rest_length
    C = distance - rest_length
    
    # 计算约束梯度（归一化方向）
    if distance > 1e-6:
        n = delta / distance
    else:
        # 避免除零，使用零向量
        n = wp.vec3(0.0, 0.0, 0.0)
        C = 0.0  # 如果距离太小，约束值也为0
    
    # 质量权重（使用质量倒数）
    w1 = 1.0 / wp.max(m1, 1e-6)
    w2 = 1.0 / wp.max(m2, 1e-6)
    
    # XPBD约束求解
    # alpha = compliance / dt^2
    # 添加数值保护：确保compliance不为零或负数
    comp = wp.max(compliance[tid], 1e-8)  # 最小compliance保护
    alpha = comp / (dt * dt)
    
    # 计算约束梯度范数的平方
    grad_norm_sq = w1 + w2
    
    # XPBD更新公式：delta_lambda = -(C + alpha * lambda) / (grad_norm_sq + alpha)
    # 添加数值保护：避免除零
    denominator = grad_norm_sq + alpha
    if denominator < 1e-10:
        return  # 跳过这个约束，避免除零
    delta_lambda = -(C + alpha * lambda_array[tid]) / denominator
    
    # 更新拉格朗日乘数
    lambda_array[tid] += delta_lambda
    
    # 位置修正（根据质量权重分配）
    correction = delta_lambda * n
    
    if idx1 < num_object_points:
        wp.atomic_add(x_corrected, idx1, -w1 * correction)
    if idx2 < num_object_points:
        wp.atomic_add(x_corrected, idx2, w2 * correction)


@wp.kernel
def xpbd_update_velocities(
    x: wp.array(dtype=wp.vec3),
    x_corrected: wp.array(dtype=wp.vec3),
    dt: float,
    v_new: wp.array(dtype=wp.vec3),
):
    """XPBD最后一步：从修正后的位置更新速度"""
    tid = wp.tid()
    v_new[tid] = (x_corrected[tid] - x[tid]) / dt


@wp.kernel
def xpbd_bending_constraint(
    x_pred: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    num_object_points: int,
    bending_edges: wp.array(dtype=wp.vec4i),  # [i0, i1, i2, i3] 共享边的两个三角形
    rest_angles: wp.array(dtype=float),  # 初始二面角
    compliance: wp.array(dtype=float),  # 弯曲柔度
    masses: wp.array(dtype=wp.float32),
    dt: float,
    lambda_bend: wp.array(dtype=float),
    x_corrected: wp.array(dtype=wp.vec3),
):
    """
    XPBD弯曲约束（二面角约束）
    用于防止布料过度弯曲，产生更自然的褶皱
    """
    tid = wp.tid()
    
    i0 = bending_edges[tid][0]
    i1 = bending_edges[tid][1]  # 共享边的两个顶点
    i2 = bending_edges[tid][2]  # 第一个三角形的第三个顶点
    i3 = bending_edges[tid][3]  # 第二个三角形的第三个顶点
    
    # 获取位置（不能使用嵌套函数，需要内联）
    if i0 >= num_object_points:
        x0 = control_x[i0 - num_object_points]
        m0 = 1.0
    else:
        x0 = x_pred[i0]
        m0 = masses[i0]
    
    if i1 >= num_object_points:
        x1 = control_x[i1 - num_object_points]
        m1 = 1.0
    else:
        x1 = x_pred[i1]
        m1 = masses[i1]
    
    if i2 >= num_object_points:
        x2 = control_x[i2 - num_object_points]
        m2 = 1.0
    else:
        x2 = x_pred[i2]
        m2 = masses[i2]
    
    if i3 >= num_object_points:
        x3 = control_x[i3 - num_object_points]
        m3 = 1.0
    else:
        x3 = x_pred[i3]
        m3 = masses[i3]
    
    # 计算两个三角形的法向量
    e0 = x1 - x0
    n1 = wp.cross(e0, x2 - x0)
    n2 = wp.cross(e0, x3 - x0)
    
    n1_len = wp.length(n1)
    n2_len = wp.length(n2)
    
    if n1_len < 1e-6 or n2_len < 1e-6:
        return  # 退化情况，跳过
    
    n1 = n1 / n1_len
    n2 = n2 / n2_len
    
    # 计算二面角
    cos_angle = wp.dot(n1, n2)
    cos_angle = wp.clamp(cos_angle, -1.0, 1.0)
    angle = wp.acos(cos_angle)
    
    # 约束值：当前角度与初始角度的差
    rest_angle = rest_angles[tid]
    C = angle - rest_angle
    
    # 计算约束梯度（简化但稳定的方法）
    # 使用二面角约束的梯度公式
    sin_angle = wp.sin(angle)
    if wp.abs(sin_angle) < 1e-6:
        return  # 避免除零
    
    # 计算垂直于共享边的方向
    e0_len = wp.length(e0)
    if e0_len < 1e-6:
        return
    
    e0_norm = e0 / e0_len
    
    # 计算垂直于n1和e0的方向（用于梯度计算）
    # 这是一个简化的梯度近似
    q1 = x2 - x0
    q2 = x3 - x0
    
    # 计算梯度方向（简化版本）
    # 使用法向量差的方向作为修正方向
    n_diff = n2 - n1
    grad_magnitude = wp.length(n_diff)
    
    if grad_magnitude < 1e-6:
        return
    
    # 质量权重
    w0 = 1.0 / wp.max(m0, 1e-6)
    w1 = 1.0 / wp.max(m1, 1e-6)
    w2 = 1.0 / wp.max(m2, 1e-6)
    w3 = 1.0 / wp.max(m3, 1e-6)
    
    # XPBD约束求解
    # 添加数值保护
    comp = wp.max(compliance[tid], 1e-8)
    alpha = comp / (dt * dt)
    
    # 计算梯度范数的平方（简化版本）
    grad_norm_sq = w0 + w1 + w2 + w3
    
    # XPBD更新
    denominator = grad_norm_sq + alpha
    if denominator < 1e-10:
        return  # 跳过这个约束，避免除零
    delta_lambda = -(C + alpha * lambda_bend[tid]) / denominator
    lambda_bend[tid] += delta_lambda
    
    # 位置修正（使用法向量差的方向）
    correction_dir = n_diff / grad_magnitude
    correction_magnitude = delta_lambda / sin_angle
    
    # 根据质量权重分配修正到各个顶点
    # 这是一个简化的分配策略
    if i0 < num_object_points:
        correction = correction_magnitude * w0 * correction_dir * 0.25
        wp.atomic_add(x_corrected, i0, correction)
    if i1 < num_object_points:
        correction = correction_magnitude * w1 * correction_dir * 0.25
        wp.atomic_add(x_corrected, i1, correction)
    if i2 < num_object_points:
        correction = correction_magnitude * w2 * correction_dir * 0.25
        wp.atomic_add(x_corrected, i2, correction)
    if i3 < num_object_points:
        correction = correction_magnitude * w3 * correction_dir * 0.25
        wp.atomic_add(x_corrected, i3, correction)


@wp.kernel
def xpbd_volume_constraint(
    x_pred: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    num_object_points: int,
    tets: wp.array(dtype=wp.vec4i),  # 四面体索引 [i0, i1, i2, i3]
    rest_volumes: wp.array(dtype=float),  # 初始体积
    compliance: wp.array(dtype=float),  # 体积柔度
    masses: wp.array(dtype=wp.float32),
    dt: float,
    lambda_vol: wp.array(dtype=float),
    x_corrected: wp.array(dtype=wp.vec3),
):
    """
    XPBD体积约束（四面体体积约束）
    用于保持3D物体的体积，防止过度压缩或膨胀
    """
    tid = wp.tid()
    
    i0 = tets[tid][0]
    i1 = tets[tid][1]
    i2 = tets[tid][2]
    i3 = tets[tid][3]
    
    # 获取位置（不能使用嵌套函数，需要内联）
    if i0 >= num_object_points:
        x0 = control_x[i0 - num_object_points]
        m0 = 1.0
    else:
        x0 = x_pred[i0]
        m0 = masses[i0]
    
    if i1 >= num_object_points:
        x1 = control_x[i1 - num_object_points]
        m1 = 1.0
    else:
        x1 = x_pred[i1]
        m1 = masses[i1]
    
    if i2 >= num_object_points:
        x2 = control_x[i2 - num_object_points]
        m2 = 1.0
    else:
        x2 = x_pred[i2]
        m2 = masses[i2]
    
    if i3 >= num_object_points:
        x3 = control_x[i3 - num_object_points]
        m3 = 1.0
    else:
        x3 = x_pred[i3]
        m3 = masses[i3]
    
    # 计算四面体体积
    # V = (1/6) * |det(x1-x0, x2-x0, x3-x0)|
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0
    
    # 计算行列式（标量三重积）
    cross_e2_e3 = wp.cross(e2, e3)
    volume = wp.dot(e1, cross_e2_e3) / 6.0
    
    rest_volume = rest_volumes[tid]
    
    # 约束值：当前体积与初始体积的差
    C = volume - rest_volume
    
    # 计算约束梯度
    # dV/dx0 = -(1/6) * (cross(x2-x0, x3-x0) + cross(x3-x0, x1-x0) + cross(x1-x0, x2-x0))
    # dV/dx1 = (1/6) * cross(x2-x0, x3-x0)
    # dV/dx2 = (1/6) * cross(x3-x0, x1-x0)
    # dV/dx3 = (1/6) * cross(x1-x0, x2-x0)
    
    grad_x0 = -(cross_e2_e3 + wp.cross(e3, e1) + wp.cross(e1, e2)) / 6.0
    grad_x1 = cross_e2_e3 / 6.0
    grad_x2 = wp.cross(e3, e1) / 6.0
    grad_x3 = wp.cross(e1, e2) / 6.0
    
    # 质量权重
    w0 = 1.0 / wp.max(m0, 1e-6)
    w1 = 1.0 / wp.max(m1, 1e-6)
    w2 = 1.0 / wp.max(m2, 1e-6)
    w3 = 1.0 / wp.max(m3, 1e-6)
    
    # 计算梯度范数的平方
    grad_norm_sq = (
        w0 * wp.length_sq(grad_x0) +
        w1 * wp.length_sq(grad_x1) +
        w2 * wp.length_sq(grad_x2) +
        w3 * wp.length_sq(grad_x3)
    )
    
    # XPBD约束求解
    # 添加数值保护
    comp = wp.max(compliance[tid], 1e-8)
    alpha = comp / (dt * dt)
    denominator = grad_norm_sq + alpha
    if denominator < 1e-10:
        return  # 跳过这个约束，避免除零
    delta_lambda = -(C + alpha * lambda_vol[tid]) / denominator
    lambda_vol[tid] += delta_lambda
    
    # 位置修正
    if i0 < num_object_points:
        correction = delta_lambda * w0 * grad_x0
        wp.atomic_add(x_corrected, i0, correction)
    if i1 < num_object_points:
        correction = delta_lambda * w1 * grad_x1
        wp.atomic_add(x_corrected, i1, correction)
    if i2 < num_object_points:
        correction = delta_lambda * w2 * grad_x2
        wp.atomic_add(x_corrected, i2, correction)
    if i3 < num_object_points:
        correction = delta_lambda * w3 * grad_x3
        wp.atomic_add(x_corrected, i3, correction)


@wp.kernel
def copy_x_pred_to_corrected(
    x_pred: wp.array(dtype=wp.vec3),
    x_corrected: wp.array(dtype=wp.vec3),
):
    """将预测位置复制到修正位置（初始化）"""
    tid = wp.tid()
    x_corrected[tid] = x_pred[tid]


@wp.kernel
def add_correction_to_pred(
    x_pred: wp.array(dtype=wp.vec3),
    x_correction: wp.array(dtype=wp.vec3),
    x_final: wp.array(dtype=wp.vec3),
):
    """将修正添加到预测位置"""
    tid = wp.tid()
    x_final[tid] = x_pred[tid] + x_correction[tid]


class XPBDSystemWarp:
    """
    基于XPBD（Extended Position Based Dynamics）的可微分物理模拟系统
    
    与SpringMassSystemWarp保持相同的接口，可以无缝替换使用。
    XPBD使用约束求解而不是显式力计算，通常更稳定。
    """
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        collision_dist=0.02,
        init_velocities=None,
        num_object_points=None,
        num_surface_points=None,
        num_original_points=None,
        controller_points=None,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        gt_object_points=None,
        gt_object_visibilities=None,
        gt_object_motions_valid=None,
        self_collision=False,
        disable_backward=False,
        xpbd_iterations=1,  # XPBD约束迭代次数
        # 弯曲约束参数（可选）
        bending_edges=None,  # 形状: (n_bending, 4) - [i0, i1, i2, i3] 共享边的两个三角形顶点索引
        rest_bending_angles=None,  # 形状: (n_bending,) - 初始二面角
        bending_compliance=None,  # 形状: (n_bending,) 或标量 - 弯曲柔度
        # 体积约束参数（可选）
        tetrahedra=None,  # 形状: (n_tets, 4) - [i0, i1, i2, i3] 四面体顶点索引
        rest_volumes=None,  # 形状: (n_tets,) - 初始体积
        volume_compliance=None,  # 形状: (n_tets,) 或标量 - 体积柔度
    ):
        logger.info(f"[SIMULATION]: Initialize the XPBD System")
        self.device = cfg.device
        self.xpbd_iterations = xpbd_iterations  # XPBD约束求解迭代次数

        # Record the parameters
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:num_object_points].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
        else:
            self.wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping  # 保留用于兼容性，但XPBD中主要通过compliance控制
        self.drag_damping = drag_damping
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        if controller_points is None:
            assert num_object_points == self.n_vertices
        else:
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices
        self.num_object_points = num_object_points
        self.num_control_points = (
            controller_points.shape[1] if controller_points is not None else 0
        )
        self.controller_points = controller_points

        # Deal with the any collision detection
        self.object_collision_flag = 0
        if init_masks is not None:
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1

        if self_collision:
            assert init_masks is None
            self.object_collision_flag = 1
            # Make all points as the collision points
            init_masks = torch.arange(
                self.n_vertices, dtype=torch.int32, device=self.device
            )

        if self.object_collision_flag:
            self.wp_masks = wp.from_torch(
                init_masks[:num_object_points].int(),
                dtype=wp.int32,
                requires_grad=False,
            )

            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

        # Initialize the GT for calculating losses
        self.gt_object_points = gt_object_points
        if cfg.data_type == "real":
            self.gt_object_visibilities = gt_object_visibilities.int()
            self.gt_object_motions_valid = gt_object_motions_valid.int()

        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        if num_original_points is None:
            self.num_original_points = self.num_object_points

        # Initialize warp arrays
        self.wp_springs = wp.from_torch(
            init_springs, dtype=wp.vec2i, requires_grad=False
        )
        self.wp_rest_lengths = wp.from_torch(
            init_rest_lengths, dtype=wp.float32, requires_grad=False
        )
        self.wp_masses = wp.from_torch(
            init_masses[:num_object_points], dtype=wp.float32, requires_grad=False
        )
        
        if cfg.data_type == "real":
            self.prev_acc = wp.zeros_like(self.wp_init_vertices, requires_grad=False)
            self.acc_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)

        self.wp_current_object_points = wp.from_torch(
            self.gt_object_points[1].clone(), dtype=wp.vec3, requires_grad=False
        )
        if cfg.data_type == "real":
            self.wp_current_object_visibilities = wp.from_torch(
                self.gt_object_visibilities[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_current_object_motions_valid = wp.from_torch(
                self.gt_object_motions_valid[0].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.num_valid_visibilities = int(self.gt_object_visibilities[1].sum())
            self.num_valid_motions = int(self.gt_object_motions_valid[0].sum())

            self.wp_original_control_point = wp.from_torch(
                self.controller_points[0].clone(), dtype=wp.vec3, requires_grad=False
            )
            self.wp_target_control_point = wp.from_torch(
                self.controller_points[1].clone(), dtype=wp.vec3, requires_grad=False
            )

            self.chamfer_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.track_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.acc_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # 初始化弯曲约束（如果提供）
        self.use_bending = bending_edges is not None and rest_bending_angles is not None
        if self.use_bending:
            self.n_bending = bending_edges.shape[0]
            self.wp_bending_edges = wp.from_torch(
                torch.tensor(bending_edges, dtype=torch.int32, device=self.device),
                dtype=wp.vec4i,
                requires_grad=False
            )
            self.wp_rest_bending_angles = wp.from_torch(
                torch.tensor(rest_bending_angles, dtype=torch.float32, device=self.device),
                requires_grad=False
            )
            if bending_compliance is None:
                # 默认弯曲柔度（较小的值，表示较硬的弯曲）
                bending_compliance = torch.ones(self.n_bending, dtype=torch.float32, device=self.device) * 1e-6
            elif isinstance(bending_compliance, (int, float)):
                bending_compliance = torch.ones(self.n_bending, dtype=torch.float32, device=self.device) * bending_compliance
            self.wp_bending_compliance = wp.from_torch(
                bending_compliance,
                requires_grad=True
            )
        else:
            self.n_bending = 0
            self.wp_bending_edges = None
            self.wp_rest_bending_angles = None
            self.wp_bending_compliance = None
        
        # 初始化体积约束（如果提供）
        self.use_volume = tetrahedra is not None and rest_volumes is not None
        if self.use_volume:
            self.n_volume = tetrahedra.shape[0]
            self.wp_tetrahedra = wp.from_torch(
                torch.tensor(tetrahedra, dtype=torch.int32, device=self.device),
                dtype=wp.vec4i,
                requires_grad=False
            )
            self.wp_rest_volumes = wp.from_torch(
                torch.tensor(rest_volumes, dtype=torch.float32, device=self.device),
                requires_grad=False
            )
            if volume_compliance is None:
                # 默认体积柔度（较小的值，表示较硬的体积约束）
                volume_compliance = torch.ones(self.n_volume, dtype=torch.float32, device=self.device) * 1e-6
            elif isinstance(volume_compliance, (int, float)):
                volume_compliance = torch.ones(self.n_volume, dtype=torch.float32, device=self.device) * volume_compliance
            self.wp_volume_compliance = wp.from_torch(
                volume_compliance,
                requires_grad=True
            )
        else:
            self.n_volume = 0
            self.wp_tetrahedra = None
            self.wp_rest_volumes = None
            self.wp_volume_compliance = None
        
        # Initialize the warp states
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = XPBDState(
                self.wp_init_velocities, 
                self.num_control_points, 
                self.n_springs,
                self.n_bending,
                self.n_volume
            )
            self.wp_states.append(state)
        if cfg.data_type == "real":
            self.distance_matrix = wp.zeros(
                (self.num_original_points, self.num_surface_points), requires_grad=False
            )
            self.neigh_indices = wp.zeros(
                (self.num_original_points), dtype=wp.int32, requires_grad=False
            )

        # Parameter to be optimized
        # Convert spring_Y (stiffness) to compliance (1/stiffness)
        spring_Y_tensor = torch.tensor(spring_Y, dtype=torch.float32, device=self.device)
        if isinstance(spring_Y, (int, float)):
            spring_Y_tensor = spring_Y_tensor * torch.ones(
                self.n_springs, dtype=torch.float32, device=self.device
            )
        
        # Calculate compliance from stiffness
        # compliance = 1 / stiffness, where stiffness = exp(spring_Y)
        stiffness = torch.clamp(
            torch.exp(spring_Y_tensor),
            min=spring_Y_min,
            max=spring_Y_max
        )
        compliance = 1.0 / stiffness
        # 添加数值保护：确保compliance在合理范围内，避免数值不稳定
        # compliance应该在 [1e-8, 1e-2] 范围内
        compliance = torch.clamp(compliance, min=1e-8, max=1e-2)
        
        # Store spring_Y for compatibility (used in set_spring_Y)
        self.wp_spring_Y = wp.from_torch(
            torch.log(stiffness),
            requires_grad=True,
        )
        
        # Store compliance for XPBD
        self.wp_compliance = wp.from_torch(
            compliance,
            requires_grad=True,
        )
        
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            torch.tensor(
                [collide_object_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            torch.tensor(
                [collide_object_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )

        # Create the CUDA graph to accelerate
        if cfg.use_graph:
            if cfg.data_type == "real":
                if not disable_backward:
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_loss()
                self.graph = capture.graph
            elif cfg.data_type == "synthetic":
                if not disable_backward:
                    # For synthetic data, we compute simple loss
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_simple_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_simple_loss()
                self.graph = capture.graph
            else:
                raise NotImplementedError

            with wp.ScopedCapture() as forward_capture:
                self.step()
            self.forward_graph = forward_capture.graph
        else:
            self.tape = wp.Tape()

    def set_controller_target(self, frame_idx, pure_inference=False):
        if self.controller_points is not None:
            # Set the controller points
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx - 1]],
                outputs=[self.wp_original_control_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx]],
                outputs=[self.wp_target_control_point],
            )

        if not pure_inference:
            # Set the target points
            wp.launch(
                copy_vec3,
                dim=self.num_original_points,
                inputs=[self.gt_object_points[frame_idx]],
                outputs=[self.wp_current_object_points],
            )

            if cfg.data_type == "real":
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_visibilities[frame_idx]],
                    outputs=[self.wp_current_object_visibilities],
                )
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_motions_valid[frame_idx - 1]],
                    outputs=[self.wp_current_object_motions_valid],
                )

                self.num_valid_visibilities = int(
                    self.gt_object_visibilities[frame_idx].sum()
                )
                self.num_valid_motions = int(
                    self.gt_object_motions_valid[frame_idx - 1].sum()
                )

    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v, pure_inference=False):
        # Detach and clone and set requires_grad=True
        assert (
            self.num_object_points == wp_x.shape[0]
            and self.num_object_points == self.wp_states[0].wp_x.shape[0]
        )

        if not pure_inference:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_x, requires_grad=False)],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_v, requires_grad=False)],
                outputs=[self.wp_states[0].wp_v],
            )
        else:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_x],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_v],
                outputs=[self.wp_states[0].wp_v],
            )

    def set_acc_count(self, acc_count):
        if acc_count:
            input = 1
        else:
            input = 0
        wp.launch(
            set_int,
            dim=1,
            inputs=[input],
            outputs=[self.acc_count],
        )

    def update_acc(self):
        wp.launch(
            update_acc,
            dim=self.num_object_points,
            inputs=[
                wp.clone(self.wp_states[0].wp_v, requires_grad=False),
                wp.clone(self.wp_states[-1].wp_v, requires_grad=False),
            ],
            outputs=[self.prev_acc],
        )

    def update_collision_graph(self):
        assert self.object_collision_flag
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_x,
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        """XPBD步进函数"""
        for i in range(self.num_substeps):
            # 重置lambda（每个子步重新开始）
            self.wp_states[i].clear_lambda()
            self.wp_states[i].clear_forces()
            
            # 设置控制点（如果存在）
            if self.controller_points is not None:
                wp.launch(
                    set_control_points,
                    dim=self.num_control_points,
                    inputs=[
                        self.num_substeps,
                        self.wp_original_control_point,
                        self.wp_target_control_point,
                        i,
                    ],
                    outputs=[self.wp_states[i].wp_control_x],
                )

            # XPBD步骤1：预测位置和速度
            wp.launch(
                kernel=xpbd_predict_positions,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_masses,
                    self.dt / self.num_substeps,
                    self.drag_damping,
                    self.reverse_factor,
                ],
                outputs=[
                    self.wp_states[i].wp_x_pred,
                    self.wp_states[i].wp_v_before_collision,  # 临时存储预测速度
                ],
            )
            
            # 初始化修正位置为预测位置
            wp.launch(
                copy_x_pred_to_corrected,
                dim=self.num_object_points,
                inputs=[self.wp_states[i].wp_x_pred],
                outputs=[self.wp_states[i].wp_x_corrected],
            )
            
            # XPBD步骤2：迭代求解约束（可以多次迭代提高稳定性）
            for iter in range(self.xpbd_iterations):
                # 重置修正（每次迭代重新计算）
                self.wp_states[i].wp_x_corrected.zero_()
                
                # 求解距离约束（计算修正）
                wp.launch(
                    kernel=xpbd_distance_constraint,
                    dim=self.n_springs,
                    inputs=[
                        self.wp_states[i].wp_x_pred,
                        self.wp_states[i].wp_control_x,
                        self.num_object_points,
                        self.wp_springs,
                        self.wp_rest_lengths,
                        self.wp_compliance,
                        self.wp_masses,
                        self.dt / self.num_substeps,
                        self.wp_states[i].wp_lambda,
                    ],
                    outputs=[self.wp_states[i].wp_x_corrected],
                )
                
                # 求解弯曲约束（如果启用）
                if self.use_bending:
                    wp.launch(
                        kernel=xpbd_bending_constraint,
                        dim=self.n_bending,
                        inputs=[
                            self.wp_states[i].wp_x_pred,
                            self.wp_states[i].wp_control_x,
                            self.num_object_points,
                            self.wp_bending_edges,
                            self.wp_rest_bending_angles,
                            self.wp_bending_compliance,
                            self.wp_masses,
                            self.dt / self.num_substeps,
                            self.wp_states[i].wp_lambda_bend,
                        ],
                        outputs=[self.wp_states[i].wp_x_corrected],
                    )
                
                # 求解体积约束（如果启用）
                if self.use_volume:
                    wp.launch(
                        kernel=xpbd_volume_constraint,
                        dim=self.n_volume,
                        inputs=[
                            self.wp_states[i].wp_x_pred,
                            self.wp_states[i].wp_control_x,
                            self.num_object_points,
                            self.wp_tetrahedra,
                            self.wp_rest_volumes,
                            self.wp_volume_compliance,
                            self.wp_masses,
                            self.dt / self.num_substeps,
                            self.wp_states[i].wp_lambda_vol,
                        ],
                        outputs=[self.wp_states[i].wp_x_corrected],
                    )
                
                # 将修正添加到预测位置（累积修正）
                wp.launch(
                    add_correction_to_pred,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x_pred,
                        self.wp_states[i].wp_x_corrected,
                    ],
                    outputs=[self.wp_states[i].wp_x_pred],
                )
            
            # 最终修正后的位置就是 x_pred，复制到 x_corrected 用于后续处理
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[self.wp_states[i].wp_x_pred],
                outputs=[self.wp_states[i].wp_x_corrected],
            )

            # 碰撞处理（保持原有逻辑）
            if self.object_collision_flag:
                # 更新速度（用于碰撞检测）
                wp.launch(
                    kernel=xpbd_update_velocities,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_x_corrected,
                        self.dt / self.num_substeps,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_collision],
                )
                
                # 对象碰撞处理
                wp.launch(
                    kernel=object_collision,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x_corrected,
                        self.wp_states[i].wp_v_before_collision,
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )
            else:
                # 如果没有对象碰撞，直接使用约束修正后的速度
                wp.launch(
                    kernel=xpbd_update_velocities,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_x_corrected,
                        self.dt / self.num_substeps,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            # 地面碰撞和最终位置更新
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x_corrected,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.wp_collide_fric,
                    self.dt / self.num_substeps,
                    self.reverse_factor,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )

    def calculate_loss(self):
        # Compute the chamfer loss
        # Precompute the distances matrix for the chamfer loss
        wp.launch(
            compute_distances,
            dim=(self.num_original_points, self.num_surface_points),
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
            ],
            outputs=[self.distance_matrix],
        )

        wp.launch(
            compute_neigh_indices,
            dim=self.num_original_points,
            inputs=[self.distance_matrix],
            outputs=[self.neigh_indices],
        )

        wp.launch(
            compute_chamfer_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.num_valid_visibilities,
                self.neigh_indices,
                cfg.chamfer_weight,
            ],
            outputs=[self.chamfer_loss],
        )

        # Compute the tracking loss
        wp.launch(
            compute_track_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_motions_valid,
                self.num_valid_motions,
                cfg.track_weight,
            ],
            outputs=[self.track_loss],
        )

        wp.launch(
            compute_acc_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_v,
                self.wp_states[-1].wp_v,
                self.prev_acc,
                self.num_object_points,
                self.acc_count,
                cfg.acc_weight,
            ],
            outputs=[self.acc_loss],
        )

        wp.launch(
            compute_final_loss,
            dim=1,
            inputs=[self.chamfer_loss, self.track_loss, self.acc_loss],
            outputs=[self.loss],
        )

    def calculate_simple_loss(self):
        wp.launch(
            compute_simple_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.num_object_points,
            ],
            outputs=[self.loss],
        )

    def clear_loss(self):
        if cfg.data_type == "real":
            self.distance_matrix.zero_()
            self.neigh_indices.zero_()
            self.chamfer_loss.zero_()
            self.track_loss.zero_()
            self.acc_loss.zero_()
        self.loss.zero_()

    # Functions used to load the parameters
    def set_spring_Y(self, spring_Y):
        """设置弹簧刚度参数（兼容性接口）"""
        # 更新spring_Y
        wp.launch(
            copy_float,
            dim=self.n_springs,
            inputs=[spring_Y],
            outputs=[self.wp_spring_Y],
        )
        
        # 同步更新compliance
        # 这里需要在CPU上计算，因为需要exp操作
        spring_Y_torch = wp.to_torch(self.wp_spring_Y)
        stiffness = torch.clamp(
            torch.exp(spring_Y_torch),
            min=self.spring_Y_min,
            max=self.spring_Y_max
        )
        compliance = 1.0 / stiffness
        # 添加数值保护
        compliance = torch.clamp(compliance, min=1e-8, max=1e-2)
        compliance_wp = wp.from_torch(compliance, requires_grad=True)
        
        wp.launch(
            copy_float,
            dim=self.n_springs,
            inputs=[compliance_wp],
            outputs=[self.wp_compliance],
        )

    def set_collide(self, collide_elas, collide_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collide_object(self, collide_object_elas, collide_object_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_elas],
            outputs=[self.wp_collide_object_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_fric],
            outputs=[self.wp_collide_object_fric],
        )

