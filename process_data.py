"""
数据处理主脚本：从原始视频数据生成物理仿真所需的训练数据

功能概述：
1. 视频分割：使用 GroundedSAM2 分割物体和控制器（手）
2. 形状先验生成（可选）：使用 Trellis 生成3D mesh
3. 密集跟踪：使用 Co-tracker 跟踪点轨迹
4. 3D重建：从深度图生成点云
5. 对齐（可选）：将形状先验与观测对齐
6. 最终数据生成：生成用于物理仿真的点云数据

输入数据要求：
- color/: RGB图像（多视角、多帧）
- depth/: 深度图（.npy格式）
- calibrate.pkl: 相机外参（c2w矩阵）
- metadata.json: 相机内参、图像尺寸

输出数据：
- mask/: 分割mask和mask_info
- shape/: 形状先验（如果启用）
- cotracker/: 密集跟踪数据
- pcd/: 3D点云数据
- final_data.pkl: 最终训练数据
- split.json: 训练/测试集划分
"""

import os
from argparse import ArgumentParser
import time
import logging
import json
import glob

# ==================== 命令行参数解析 ====================
parser = ArgumentParser(description="处理原始视频数据，生成物理仿真训练数据")
parser.add_argument(
    "--base_path",
    type=str,
    default="./data/different_types",
    help="数据基础路径，包含所有案例的文件夹"
)
parser.add_argument(
    "--case_name",
    type=str,
    required=True,
    help="案例名称，对应 base_path 下的子文件夹名称"
)
parser.add_argument(
    "--category",
    type=str,
    required=True,
    help="物体类别（如 'cloth', 'rope', 'sloth'），用于分割提示词"
)
parser.add_argument(
    "--shape_prior",
    action="store_true",
    default=False,
    help="是否使用形状先验（Trellis生成的3D mesh）"
)
args = parser.parse_args()

# ==================== 处理流程控制标志 ====================
# 可以通过设置这些标志来跳过某些处理步骤（用于调试）
PROCESS_SEG = True          # 视频分割：识别物体和控制器
PROCESS_SHAPE_PRIOR = True  # 形状先验生成：使用Trellis生成3D mesh
PROCESS_TRACK = True        # 密集跟踪：使用Co-tracker跟踪点轨迹
PROCESS_3D = True           # 3D重建：从深度图生成点云
PROCESS_ALIGN = True        # 对齐：将形状先验与观测对齐
PROCESS_FINAL = True        # 最终数据生成：生成训练用的点云数据

# ==================== 关键参数设置 ====================
base_path = args.base_path          # 数据基础路径
case_name = args.case_name          # 案例名称
category = args.category            # 物体类别
TEXT_PROMPT = f"{category}.hand"    # 分割提示词：用于GroundedSAM2识别物体和手
CONTROLLER_NAME = "hand"            # 控制器名称（通常是"hand"）
SHAPE_PRIOR = args.shape_prior      # 是否使用形状先验

# ==================== 日志系统 ====================
logger = None


def setup_logger(log_file="timer.log"):
    """
    设置全局日志记录器
    
    参数:
        log_file: 日志文件路径，默认 "timer.log"
    """
    global logger 

    if logger is None:
        logger = logging.getLogger("GlobalLogger")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)


setup_logger()


def existDir(dir_path):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        dir_path: 目录路径
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class Timer:
    """
    计时器上下文管理器，用于记录每个处理步骤的耗时
    """
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )


# ==================== 步骤1: 视频分割 ====================
if PROCESS_SEG:
    """
    功能：使用 GroundedSAM2 对视频进行分割，识别物体和控制器（手）
    
    输入：
        - color/: RGB图像
        - TEXT_PROMPT: 分割提示词（如 "cloth.hand"），格式为 "{category}.hand"，点号分隔不同对象类别，用于指导分割模型识别物体和手
    
    输出：
        - mask/: 分割mask图像
        - mask/mask_info_*.json: mask信息（物体和控制器的索引）
    """
    with Timer("Video Segmentation"):
        os.system(
            f"python ./data_process/segment.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT}"
        )


# ==================== 步骤2: 形状先验生成（可选） ====================
if PROCESS_SHAPE_PRIOR and SHAPE_PRIOR:
    """
    功能：生成物体的3D形状先验（mesh），用于指导点云重建
    
    流程：
        1. 图像上采样：使用SDXL提升图像分辨率
        2. 图像分割：提取物体mask
        3. 形状生成：使用Trellis从单张图像生成3D mesh
    
    输入：
        - color/0/0.png: 第一帧第一视角的RGB图像
        - mask/: 物体mask
        - category: 物体类别，用于分割提示词
        - CONTROLLER_NAME: "hand"，用于区分物体和控制器
    
    输出：
        - shape/high_resolution.png: 高分辨率图像
        - shape/masked_image.png: 物体mask图像
        - shape/matching/final_mesh.glb: 生成的3D mesh
    """
    # 从mask_info中获取物体索引（排除控制器）
    with open(f"{base_path}/{case_name}/mask/mask_info_{0}.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    mask_path = f"{base_path}/{case_name}/mask/0/{obj_idx}/0.png"

    existDir(f"{base_path}/{case_name}/shape")
    
    # 2.1 图像上采样：提升分辨率以便生成更好的3D模型
    with Timer("Image Upscale"):
        if not os.path.isfile(f"{base_path}/{case_name}/shape/high_resolution.png"):
            os.system(
                f"python ./data_process/image_upscale.py --img_path {base_path}/{case_name}/color/0/0.png --mask_path {mask_path} --output_path {base_path}/{case_name}/shape/high_resolution.png --category {category}"
            )

    # 2.2 图像分割：提取物体的mask
    with Timer("Image Segmentation"):
        os.system(
            f"python ./data_process/segment_util_image.py --img_path {base_path}/{case_name}/shape/high_resolution.png --TEXT_PROMPT {category} --output_path {base_path}/{case_name}/shape/masked_image.png"
        )

    # 2.3 形状生成：使用Trellis从单张图像生成3D mesh
    with Timer("Shape Prior Generation"):
        os.system(
            f"python ./data_process/shape_prior.py --img_path {base_path}/{case_name}/shape/masked_image.png --output_dir {base_path}/{case_name}/shape"
        )

# ==================== 步骤3: 密集跟踪 ====================
if PROCESS_TRACK:
    """
    功能：使用 Co-tracker 对视频进行密集点跟踪
    
    输入：
        - color/: RGB图像序列
        - mask/: 分割mask
        - 使用默认的Co-tracker配置
    
    输出：
        - cotracker/: 跟踪数据（每个视角一个文件）
            - tracks: 点轨迹 [frame, point, (x, y)]
            - visibility: 可见性矩阵 [frame, point]
    """
    with Timer("Dense Tracking"):
        os.system(
            f"python ./data_process/dense_track.py --base_path {base_path} --case_name {case_name}"
        )

# ==================== 步骤4: 3D重建 ====================
if PROCESS_3D:
    """
    功能：从多视角深度图重建3D点云，并处理跟踪数据
    
    子步骤：
        4.1 Lift to 3D: 从深度图生成点云
        4.2 Mask后处理: 过滤噪声
        4.3 跟踪数据处理: 将2D跟踪转换为3D点云轨迹
    """
    
    # 4.1 从深度图生成3D点云（世界坐标系）
    with Timer("Lift to 3D"):
        """
        功能：将深度图转换为3D点云
        
        输入：
            - depth/: 深度图（.npy格式）
            - calibrate.pkl: 相机外参（c2w）
            - metadata.json: 相机内参
        
        输出：
            - pcd/: 点云数据（.npz格式，每帧一个文件）
                - points: 点云坐标 [num_cam, num_points, 3]
                - colors: 点云颜色 [num_cam, num_points, 3]
                - masks: mask信息
        """
        os.system(
            f"python ./data_process/data_process_pcd.py --base_path {base_path} --case_name {case_name}"
        )

    # 4.2 Mask后处理：过滤噪声，提高mask质量
    with Timer("Mask Post-Processing"):
        """
        功能：对物体和控制器mask进行后处理，去除噪声
        
        输入：
            - pcd/: 点云数据
            - mask/: 原始mask
            - CONTROLLER_NAME: "hand"，用于识别控制器mask
        
        输出：
            - mask/processed_masks.pkl: 处理后的mask
        """
        os.system(
            f"python ./data_process/data_process_mask.py --base_path {base_path} --case_name {case_name} --controller_name {CONTROLLER_NAME}"
        )

    # 4.3 跟踪数据处理：将2D跟踪转换为3D点云轨迹
    with Timer("Data Tracking"):
        """
        功能：将Co-tracker的2D跟踪结果转换为3D点云轨迹
        
        输入：
            - cotracker/: 2D跟踪数据
            - pcd/: 点云数据
            - mask/processed_masks.pkl: 处理后的mask
        
        输出：
            - 更新 pcd/ 中的点云数据，添加跟踪信息
            - 生成物体点和控制器点的轨迹
        """
        os.system(
            f"python ./data_process/data_process_track.py --base_path {base_path} --case_name {case_name}"
        )

# ==================== 步骤5: 对齐（可选，需要形状先验） ====================
if PROCESS_ALIGN and SHAPE_PRIOR:
    """
    功能：将形状先验（3D mesh）与观测点云对齐
    
    输入：
        - shape/matching/final_mesh.glb: 形状先验mesh
        - pcd/: 观测点云
        - CONTROLLER_NAME: "hand"，用于排除控制器点
    
    输出：
        - shape/matching/aligned_mesh.glb: 对齐后的mesh（如果生成）
        - 更新点云数据，融合形状先验信息
    """
    with Timer("Alignment"):
        os.system(
            f"python ./data_process/align.py --base_path {base_path} --case_name {case_name} --controller_name {CONTROLLER_NAME}"
        )

# ==================== 步骤6: 最终数据生成 ====================
if PROCESS_FINAL:
    """
    功能：生成用于物理仿真的最终点云数据
    
    输入：
        - pcd/: 处理后的点云数据
        - shape/matching/final_mesh.glb: 形状先验（如果启用）
        - SHAPE_PRIOR: 是否使用形状先验来指导点云采样
    
    输出：
        - final_data.pkl: 最终训练数据，包含：
            - object_points: 物体点云 [frame, point, 3]
            - controller_points: 控制器点云 [frame, point, 3]
            - object_colors: 点云颜色
            - object_visibilities: 可见性矩阵
            - object_motions_valid: 运动有效性
        - split.json: 训练/测试集划分
    """
    with Timer("Final Data Generation"):
        if SHAPE_PRIOR:
            os.system(
                f"python ./data_process/data_process_sample.py --base_path {base_path} --case_name {case_name} --shape_prior"
            )
        else:
            os.system(
                f"python ./data_process/data_process_sample.py --base_path {base_path} --case_name {case_name}"
            )

    # 保存训练/测试集划分（70%训练，30%测试）
    frame_len = len(glob.glob(f"{base_path}/{case_name}/pcd/*.npz"))
    split = {}
    split["frame_len"] = frame_len
    split["train"] = [0, int(frame_len * 0.7)]  # 前70%用于训练
    split["test"] = [int(frame_len * 0.7), frame_len]  # 后30%用于测试
    with open(f"{base_path}/{case_name}/split.json", "w") as f:
        json.dump(split, f)
