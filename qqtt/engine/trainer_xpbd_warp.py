"""
XPBD训练器

专门用于训练XPBD系统的训练器，主要改动：
1. 使用XPBDSystemWarp替代SpringMassSystemWarp
2. 优化器包含wp_compliance（主参数）和可选的弯曲/体积约束参数
3. 参数同步：优化compliance后同步更新spring_Y（用于兼容性）
4. 模型保存包含所有compliance参数
"""

from qqtt.data import RealData, SimpleData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt.model.diff_simulator import XPBDSystemWarp
from qqtt.model.diff_simulator.spring_mass_warp import copy_float
import open3d as o3d
import numpy as np
import torch
import wandb
import os
from tqdm import tqdm
import warp as wp
from scipy.spatial import Delaunay
import pickle
import warnings
import cv2
import logging

# 抑制Warp关于enable_backward=False的警告（这些kernel确实不需要梯度）
# 设置warp的日志级别来抑制警告
warp_logger = logging.getLogger('warp')
warp_logger.setLevel(logging.ERROR)
# 使用warnings过滤器抑制UserWarning
warnings.filterwarnings('ignore', message='.*enable_backward=False.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Running the tape backwards.*', category=UserWarning)
# Optional imports for headless environments
try:
    from pynput import keyboard
except (ImportError, OSError):
    keyboard = None
try:
    import pyrender
except (ImportError, OSError):
    pyrender = None
try:
    import trimesh
except (ImportError, OSError):
    trimesh = None
import matplotlib.pyplot as plt

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.dynamic_utils import (
    interpolate_motions_speedup,
    knn_weights,
    knn_weights_sparse,
    get_topk_indices,
    calc_weights_vals_from_indices,
)
from gaussian_splatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from gs_render import (
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)
from gaussian_splatting.rotation_utils import quaternion_multiply, matrix_to_quaternion

from sklearn.cluster import KMeans
import copy
import time
import threading


class InvPhyTrainerXPBD:
    """
    XPBD训练器
    
    与InvPhyTrainerWarp的主要区别：
    1. 使用XPBDSystemWarp
    2. 优化器包含wp_compliance等XPBD特有参数
    3. 参数同步机制：优化compliance后同步spring_Y
    4. 支持可选的弯曲和体积约束
    """
    def __init__(
        self,
        data_path,
        base_dir,
        train_frame=None,
        mask_path=None,
        velocity_path=None,
        pure_inference_mode=False,
        device="cuda:0",
        # XPBD特有参数
        use_bending=False,  # 是否使用弯曲约束
        use_volume=False,  # 是否使用体积约束
        xpbd_iterations=1,  # XPBD迭代次数
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        cfg.train_frame = train_frame
        
        self.use_bending = use_bending
        self.use_volume = use_volume
        self.xpbd_iterations = xpbd_iterations

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False, save_gt=False)
            # Get the object points and controller points
            self.object_points = self.dataset.object_points
            self.object_colors = self.dataset.object_colors
            self.object_visibilities = self.dataset.object_visibilities
            self.object_motions_valid = self.dataset.object_motions_valid
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_original_points = self.dataset.num_original_points
            self.num_surface_points = self.dataset.num_surface_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            self.dataset = SimpleData(visualize=False)
            self.object_points = self.dataset.data
            self.object_colors = None
            self.object_visibilities = None
            self.object_motions_valid = None
            self.controller_points = None
            self.structure_points = self.dataset.data[0]
            self.num_original_points = None
            self.num_surface_points = None
            self.num_all_points = len(self.dataset.data[0])
            # Prepare for the multiple object case
            if mask_path is not None:
                mask = np.load(mask_path)
                self.init_masks = torch.tensor(
                    mask, dtype=torch.float32, device=cfg.device
                )
            if velocity_path is not None:
                velocity = np.load(velocity_path)
                self.init_velocities = torch.tensor(
                    velocity, dtype=torch.float32, device=cfg.device
                )
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

        # Initialize the vertices, springs, rest lengths and masses
        if self.controller_points is None:
            firt_frame_controller_points = None
        else:
            firt_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            firt_frame_controller_points,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )

        # 生成弯曲约束（如果启用）
        bending_edges = None
        rest_bending_angles = None
        if self.use_bending:
            # 需要从点云或网格生成弯曲约束
            # 这里简化处理：如果有网格数据，可以生成；否则跳过
            logger.info("[XPBD]: Bending constraints requested but not implemented yet. Skipping.")
            self.use_bending = False
        
        # 生成体积约束（如果启用）
        tetrahedra = None
        rest_volumes = None
        if self.use_volume:
            try:
                # 使用Delaunay三角剖分生成四面体
                vertices_np = self.init_vertices.cpu().numpy()
                tri = Delaunay(vertices_np)
                tetrahedra = tri.simplices
                
                # 计算初始体积
                rest_volumes = []
                for tet in tetrahedra:
                    x0, x1, x2, x3 = vertices_np[tet]
                    e1 = x1 - x0
                    e2 = x2 - x0
                    e3 = x3 - x0
                    volume = np.abs(np.dot(e1, np.cross(e2, e3))) / 6.0
                    rest_volumes.append(volume)
                
                tetrahedra = torch.tensor(tetrahedra, dtype=torch.int32, device=cfg.device)
                rest_volumes = torch.tensor(rest_volumes, dtype=torch.float32, device=cfg.device)
                logger.info(f"[XPBD]: Generated {len(tetrahedra)} volume constraints")
            except Exception as e:
                logger.warning(f"[XPBD]: Failed to generate volume constraints: {e}. Skipping.")
                self.use_volume = False

        # 创建XPBD模拟器
        self.simulator = XPBDSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
            xpbd_iterations=self.xpbd_iterations,
            # 可选约束
            bending_edges=bending_edges,
            rest_bending_angles=rest_bending_angles,
            bending_compliance=1e-6 if self.use_bending else None,
            tetrahedra=tetrahedra.numpy() if self.use_volume else None,
            rest_volumes=rest_volumes.numpy() if self.use_volume else None,
            volume_compliance=1e-6 if self.use_volume else None,
        )

        if not pure_inference_mode:
            # 构建优化器参数列表
            optimizer_params = [
                wp.to_torch(self.simulator.wp_compliance),  # 主参数：距离约束柔度
            ]
            
            # 添加碰撞参数（如果可优化）
            if cfg.collision_learn:
                optimizer_params.extend([
                    wp.to_torch(self.simulator.wp_collide_elas),
                    wp.to_torch(self.simulator.wp_collide_fric),
                    wp.to_torch(self.simulator.wp_collide_object_elas),
                    wp.to_torch(self.simulator.wp_collide_object_fric),
                ])
            
            # 添加弯曲约束参数（如果启用）
            if self.simulator.use_bending:
                optimizer_params.append(wp.to_torch(self.simulator.wp_bending_compliance))
            
            # 添加体积约束参数（如果启用）
            if self.simulator.use_volume:
                optimizer_params.append(wp.to_torch(self.simulator.wp_volume_compliance))
            
            self.optimizer = torch.optim.Adam(
                optimizer_params,
                lr=cfg.base_lr,
                betas=(0.9, 0.99),
            )

            if "debug" not in cfg.run_name:
                wandb.init(
                    project="final_pipeline",
                    name=cfg.run_name + "_xpbd",
                    config=cfg.to_dict(),
                )
            else:
                wandb.init(
                    project="Debug",
                    name=cfg.run_name + "_xpbd",
                    config=cfg.to_dict(),
                )
            if not os.path.exists(f"{cfg.base_dir}/train"):
                os.makedirs(f"{cfg.base_dir}/train")

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        """初始化顶点、弹簧、静止长度和质量（与原有方法相同）"""
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)

            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                num_object_points = len(points)
                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )

    def _sync_spring_Y_from_compliance(self):
        """
        从compliance同步更新spring_Y（用于兼容性）
        在optimizer.step()之后调用
        """
        compliance_torch = wp.to_torch(self.simulator.wp_compliance, requires_grad=False)
        # 添加数值保护：确保compliance在合理范围内
        compliance_torch = torch.clamp(compliance_torch, min=1e-8, max=1e-2)
        stiffness = 1.0 / compliance_torch
        stiffness = torch.clamp(stiffness, min=self.simulator.spring_Y_min, max=self.simulator.spring_Y_max)
        spring_Y = torch.log(stiffness)
        spring_Y_wp = wp.from_torch(spring_Y, requires_grad=False)
        
        # 更新wp_spring_Y（不参与梯度计算，仅用于兼容性）
        wp.launch(
            copy_float,
            dim=self.simulator.n_springs,
            inputs=[spring_Y_wp],
            outputs=[self.simulator.wp_spring_Y],
        )

    def train(self, start_epoch=-1, skip_visualization=False):
        """训练XPBD系统"""
        # Render the initial visualization (skip in headless mode)
        if not skip_visualization:
            try:
                video_path = f"{cfg.base_dir}/train/init_xpbd.mp4"
                self.visualize_sim(save_only=True, video_path=video_path)
            except Exception as e:
                logger.warning(f"[XPBD]: Failed to create initial visualization: {e}. Skipping visualization.")
                skip_visualization = True

        best_loss = None
        best_epoch = None
        # Train the model with the physical simulator
        for i in range(start_epoch + 1, cfg.iterations):
            total_loss = 0.0
            if cfg.data_type == "real":
                total_chamfer_loss = 0.0
                total_track_loss = 0.0
            self.simulator.set_init_state(
                self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
            )
            with wp.ScopedTimer("backward"):
                for j in tqdm(range(1, cfg.train_frame)):
                    self.simulator.set_controller_target(j)
                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()

                    if cfg.use_graph:
                        wp.capture_launch(self.simulator.graph)
                    else:
                        if cfg.data_type == "real":
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_loss()
                            self.simulator.tape.backward(self.simulator.loss)
                        else:
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_simple_loss()
                            self.simulator.tape.backward(self.simulator.loss)

                    self.optimizer.step()
                    
                    # 同步更新spring_Y（从compliance计算，用于兼容性）
                    self._sync_spring_Y_from_compliance()

                    if cfg.data_type == "real":
                        chamfer_loss = wp.to_torch(
                            self.simulator.chamfer_loss, requires_grad=False
                        )
                        track_loss = wp.to_torch(
                            self.simulator.track_loss, requires_grad=False
                        )
                        total_chamfer_loss += chamfer_loss.item()
                        total_track_loss += track_loss.item()

                    loss = wp.to_torch(self.simulator.loss, requires_grad=False)
                    total_loss += loss.item()

                    if cfg.use_graph:
                        # Only need to clear the gradient, the tape is created in the graph
                        self.simulator.tape.zero()
                    else:
                        # Need to reset the compute graph and clear the gradient
                        self.simulator.tape.reset()
                    self.simulator.clear_loss()
                    # Set the intial state for the next step
                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                    )

            total_loss /= cfg.train_frame - 1
            if cfg.data_type == "real":
                total_chamfer_loss /= cfg.train_frame - 1
                total_track_loss /= cfg.train_frame - 1
            
            # 记录日志
            log_dict = {
                "loss": total_loss,
                "chamfer_loss": (
                    total_chamfer_loss if cfg.data_type == "real" else 0
                ),
                "track_loss": total_track_loss if cfg.data_type == "real" else 0,
            }
            
            if cfg.collision_learn:
                log_dict.update({
                    "collide_elas": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ).item(),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ).item(),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ).item(),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ).item(),
                })
            
            # 记录compliance统计信息
            compliance_torch = wp.to_torch(self.simulator.wp_compliance, requires_grad=False)
            log_dict.update({
                "compliance_mean": compliance_torch.mean().item(),
                "compliance_std": compliance_torch.std().item(),
                "compliance_min": compliance_torch.min().item(),
                "compliance_max": compliance_torch.max().item(),
            })
            
            wandb.log(log_dict, step=i)

            logger.info(f"[Train XPBD]: Iteration: {i}, Loss: {total_loss}")

            if (i % cfg.vis_interval == 0 or i == cfg.iterations - 1) and not skip_visualization:
                try:
                    video_path = f"{cfg.base_dir}/train/sim_xpbd_iter{i}.mp4"
                    self.visualize_sim(save_only=True, video_path=video_path)
                    wandb.log(
                        {
                            "video": wandb.Video(
                                video_path,
                                format="mp4",
                                fps=cfg.FPS,
                            ),
                        },
                        step=i,
                    )
                except Exception as e:
                    logger.warning(f"[XPBD]: Failed to create visualization at iteration {i}: {e}. Skipping visualization.")
                    skip_visualization = True
                # Save the parameters
                cur_model = {
                    "epoch": i,
                    "num_object_springs": self.num_object_springs,
                    "spring_Y": torch.exp(
                        wp.to_torch(self.simulator.wp_spring_Y, requires_grad=False)
                    ),
                    "compliance": wp.to_torch(
                        self.simulator.wp_compliance, requires_grad=False
                    ),
                    "collide_elas": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                
                # 保存可选约束参数
                if self.simulator.use_bending:
                    cur_model["bending_compliance"] = wp.to_torch(
                        self.simulator.wp_bending_compliance, requires_grad=False
                    )
                if self.simulator.use_volume:
                    cur_model["volume_compliance"] = wp.to_torch(
                        self.simulator.wp_volume_compliance, requires_grad=False
                    )
                
                if best_loss == None or total_loss < best_loss:
                    # Remove old best model file if it exists
                    if best_loss is not None:
                        old_best_model_path = (
                            f"{cfg.base_dir}/train/best_xpbd_{best_epoch}.pth"
                        )
                        if os.path.exists(old_best_model_path):
                            os.remove(old_best_model_path)

                    # Update best loss and best epoch
                    best_loss = total_loss
                    best_epoch = i

                    # Save new best model
                    best_model_path = f"{cfg.base_dir}/train/best_xpbd_{best_epoch}.pth"
                    torch.save(cur_model, best_model_path)
                    logger.info(
                        f"Latest best XPBD model saved: epoch {best_epoch} with loss {best_loss}"
                    )

                torch.save(cur_model, f"{cfg.base_dir}/train/iter_xpbd_{i}.pth")
                logger.info(
                    f"[Visualize]: Visualize the XPBD simulation at iteration {i} and save the model"
                )

        wandb.finish()

    def visualize_sim(self, save_only=True, video_path=None, save_trajectory=False, save_path=None):
        """可视化模拟结果"""
        logger.info("Visualizing the XPBD simulation")
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        vertices = [
            wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
        ]

        with wp.ScopedTimer("simulate"):
            for i in tqdm(range(1, frame_len)):
                if cfg.data_type == "real":
                    self.simulator.set_controller_target(i, pure_inference=True)
                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(self.simulator.forward_graph)
                else:
                    self.simulator.step()
                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                vertices.append(x.cpu())
                # Set the intial state for the next step
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

        vertices = torch.stack(vertices, dim=0)

        if save_trajectory:
            logger.info(f"Save the trajectory to {save_path}")
            vertices_to_save = vertices.cpu().numpy()
            with open(save_path, "wb") as f:
                pickle.dump(vertices_to_save, f)

        if not save_only:
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=True,
            )
        else:
            assert video_path is not None, "Please provide the video path to save"
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

