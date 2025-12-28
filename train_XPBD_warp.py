"""
XPBD训练脚本

与train_warp.py的主要区别：
1. 使用XPBDSystemWarp替代SpringMassSystemWarp
2. 不需要CMA-ES零阶优化阶段，直接进行梯度下降
3. 优化器包含wp_compliance等XPBD特有参数
4. 支持可选的弯曲和体积约束

使用方法：
    python train_XPBD_warp.py --base_path <data_path> --case_name <case> --train_frame <frames>
example:
    python train_XPBD_warp.py \
    --base_path ./data/different_types \
    --case_name double_lift_cloth_1 \
    --train_frame 81
    --skip_visualization # 跳过可视化（适用于SSH远程开发等headless环境）
可选参数：
    --use_bending: 启用弯曲约束（需要网格数据）
    --use_volume: 启用体积约束
    --xpbd_iterations: XPBD约束求解迭代次数（默认1）
"""

# Set environment variables for headless mode before importing any OpenGL-dependent libraries
import os
# Prevent OpenGL initialization attempts in headless environments
if 'DISPLAY' not in os.environ or os.environ.get('DISPLAY') == '':
    os.environ['PYGLET_HEADLESS'] = '1'
    os.environ['DISPLAY'] = ''

from qqtt.engine.trainer_xpbd_warp import InvPhyTrainerXPBD
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import os
import json


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="数据根目录路径")
    parser.add_argument("--case_name", type=str, required=True, help="案例名称")
    parser.add_argument("--train_frame", type=int, required=True, help="训练帧数")
    parser.add_argument("--use_bending", action="store_true", help="启用弯曲约束（需要网格数据）")
    parser.add_argument("--use_volume", action="store_true", help="启用体积约束")
    parser.add_argument("--xpbd_iterations", type=int, default=1, help="XPBD约束求解迭代次数（默认1，建议2-3）")
    parser.add_argument("--skip_visualization", action="store_true", help="跳过可视化（适用于SSH远程开发等headless环境）")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame
    use_bending = args.use_bending
    use_volume = args.use_volume
    xpbd_iterations = args.xpbd_iterations

    # 加载配置文件
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    print(f"[DATA TYPE]: {cfg.data_type}")
    print(f"[XPBD]: Iterations={xpbd_iterations}, Bending={use_bending}, Volume={use_volume}")

    base_dir = f"experiments_XPBD/{case_name}"

    # XPBD不需要CMA-ES零阶优化阶段，直接使用配置中的初始参数
    # 如果需要，可以从optimal_params.pkl加载碰撞参数等稀疏参数
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    if os.path.exists(optimal_path):
        import pickle
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        # 只使用碰撞相关参数，不覆盖spring_Y（因为XPBD使用compliance）
        if "collide_elas" in optimal_params:
            cfg.collide_elas = optimal_params["collide_elas"]
        if "collide_fric" in optimal_params:
            cfg.collide_fric = optimal_params["collide_fric"]
        if "collide_object_elas" in optimal_params:
            cfg.collide_object_elas = optimal_params["collide_object_elas"]
        if "collide_object_fric" in optimal_params:
            cfg.collide_object_fric = optimal_params["collide_object_fric"]
        logger.info(f"[XPBD]: Loaded collision parameters from {optimal_path}")
    else:
        logger.info(f"[XPBD]: No optimal_params.pkl found, using default collision parameters")

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        import pickle
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    logger.set_log_file(path=base_dir, name="inv_phy_xpbd_log")
    trainer = InvPhyTrainerXPBD(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
        use_bending=use_bending,
        use_volume=use_volume,
        xpbd_iterations=xpbd_iterations,
    )
    trainer.train(skip_visualization=args.skip_visualization)

