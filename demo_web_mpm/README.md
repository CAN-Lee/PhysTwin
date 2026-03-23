# PhysTwin 网页鼠标拖拽 Demo（无头服务器）

在浏览器里用鼠标点选粒子并拖拽，后端在 GPU 上跑 Warp MPM（与训练一致的混合本构 + 可选 ResidualPGND）。

## 依赖

在 `phystwin` 环境中：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phystwin
pip install -r demo_web_mpm/requirements.txt
```

## 启动

在 **PhysTwin 仓库根目录**：

```bash
python demo_web_mpm/interactive_mpm_server.py \
  --case_name single_push_rope \
  --config configs/rope.yaml \
  --checkpoint ./output_3/rope_warp/single_push_rope/best_checkpoint.pt \
  --host 0.0.0.0 \
  --port 8765
```

- 不传 `--checkpoint` 时默认：`{output_dir}/{case_name}/best_checkpoint.pt`（与 yaml 中 `output_dir` 一致）。
- `--gpu`：指定可见 GPU（默认 `0`）。
- `--steps_per_tick`：每次拖拽/空闲推进的子步数（默认 `56`，越大越稳、越慢）。
- `--max_send_particles`：每帧最多向浏览器发送的粒子数（多粒子场景会自动 stride 下采样）。
- `--use_neural`：开启 ResidualPGND（更慢、更接近完整推理）。
- `--settle_iters`：从数据**第一帧**出发、无控制器只做重力与接触，松弛到地面的迭代次数（默认 `220`；越大越贴地、首帧越慢）。

**初始姿态**：首次「连接」后的「重置」会在服务端把物体从 `final_data` 首帧松弛到地面再缓存，之后每次重置都回到该贴地姿态。

**可视化**：黄色球/光 = **Action**；红色小球 = **被 PD 弹簧连接的粒子**。  
**坐标系**：仿真为 **XY 地面、Z 竖直**（重力 −Z）；网页 Three.js 为 **XZ 地面、Y 竖直**，前端已做 `(x,y,z)_sim → (x, z, -y)_three`，网格与地面对齐。

默认端口 **8765**；若已被占用，程序会自动改用 8766、8767… 并在终端打印实际端口。  
本机浏览器打开终端里提示的地址（例如 **http://127.0.0.1:8765**）。

## 使用

1. 点击 **连接服务器**（默认 `ws://当前主机/ws`）。
2. 连接成功后会自动 **重置物体**。
3. **左键**点中某个粒子后拖拽；松开鼠标后物体会按物理继续运动。
4. **重置物体** 恢复初始姿态。

旋转视角：拖拽空白处（未点中粒子时）使用轨道控制器。

## 说明

- 配置里必须是 **`mpm.use_warp: true`**。
- 仅作交互演示；抓取会 **从初始帧重新挂接** controller（与训练时 PD 弹簧一致）。
- 若端口或反向代理不同，连接时按提示改 WebSocket URL。
