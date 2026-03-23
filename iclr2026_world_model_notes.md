# ICLR 2026 paperlist 中 world model 相关论文整理

说明：我按“标题或关键词里明确出现 world model / world modeling / world simulation”等严格口径先筛，再做人工归类。

严格口径候选数：52 篇。

## 推荐优先读（18 篇）

1. **Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling**（#73，page score 7.33）  
   - 对象中心、自监督、从视频直接学 keypoints/boxes/masks，偏“视觉世界模型本体”。
2. **WoW!: World Models in a Closed-Loop World**（#205，page score 7.00）  
   - 强调闭环评测：不只看视频质量，而是看能否真的帮助具身任务成功。
3. **FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction**（#384，page score 6.50）  
   - 把视频生成和隐式3D场联合建模，突出 3D/几何一致性。
4. **Vid2World: Crafting Video Diffusion Models to Interactive World Models**（#3113，page score 5.20）  
   - 把大规模视频 diffusion model 改造成 interactive world model。
5. **Astra: General Interactive World Model with Autoregressive Denoising**（#3775，page score 5.00）  
   - 基于 autoregressive denoising 的通用交互式视频世界模型。
6. **Learning to Be Uncertain: Pre-training World Models with Horizon-Calibrated Uncertainty**（#4414，page score 4.67）  
   - 显式建模“越往未来越不确定”的 world model 预训练。
7. **Learning Massively Multitask World Models for Continuous Control**（#1121，page score 6.00）  
   - 多任务连续控制 world model，大 benchmark + language-conditioned model。
8. **Mixture-of-World Models: Scaling Multi-Task Reinforcement Learning with Modular Latent Dynamics**（#5216，page score 4.00）  
   - Mixture-of-World Models，模块化 latent dynamics 做多任务扩展。
9. **From Observations to Events: Event-Aware World Models for Reinforcement Learning**（#4347，page score 4.67）  
   - event-aware 表征，让 world model 更鲁棒地泛化到结构相似场景。
10. **WorldGym: World Model as An Environment for Policy Evaluation**（#678，page score 6.50）  
   - 把 world model 当 policy evaluation environment。
11. **Ctrl-World: A Controllable Generative World Model for Robot Manipulation**（#1252，page score 6.00）  
   - 可控、多视角、长时交互的机器人操作世界模型。
12. **WMPO: World Model-based Policy Optimization for Vision-Language-Action Models**（#3210，page score 5.00）  
   - 用 pixel-space video world model 对 VLA 做 imagined on-policy RL。
13. **ViMo: A Generative Visual GUI World Model for App Agents**（#894，page score 6.00）  
   - 首个 visual GUI world model，直接生成未来 GUI 图像。
14. **R-WoM: Retrieval-augmented World Model For Computer-use Agents**（#3843，page score 5.00）  
   - 检索增强 world model，给 computer-use agent 补充外部教程知识。
15. **DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving**（#492，page score 6.50）  
   - 驾驶 world models 的综合 benchmark。
16. **DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving**（#2231，page score 5.50）  
   - world modeling 作为 VLA 驾驶模型的 dense self-supervision。
17. **ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving**（#4267，page score 4.67）  
   - Temporal Residual World Model，抓动态对象，服务端到端 driving planning。
18. **VCWorld: A Biological World Model for Virtual Cell Simulation**（#4590，page score 4.50）  
   - 把 biological knowledge + LLM reasoning 结合成 virtual cell world model。

## 完整长名单（按主题）

### 核心生成式 / 表征 / 基础世界模型

- **Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling**（#73，score 7.33）  
  - 对象中心、自监督、从视频直接学 keypoints/boxes/masks，偏“视觉世界模型本体”。
- **WoW!: World Models in a Closed-Loop World**（#205，score 7.00）  
  - 强调闭环评测：不只看视频质量，而是看能否真的帮助具身任务成功。
- **FantasyWorld: Geometry-Consistent World Modeling via Unified Video and 3D Prediction**（#384，score 6.50）  
  - 把视频生成和隐式3D场联合建模，突出 3D/几何一致性。
- **Verification of the Implicit World Model in a Generative Model via Adversarial Sequences**（#388，score 6.50）  
  - 验证生成模型里的“隐式世界模型”是否真的学到规则结构。
- **Building spatial world models from sparse transitional episodic memories**（#571，score 6.50）  
  - 从稀疏 episodic memories 构建空间 world model，偏认知/导航。
- **FlashWorld: High-quality 3D Scene Generation within Seconds**（#1131，score 6.00）  
  - 高质量 3D scene generation，站在 world models 语境下做加速。
- **Composition of Memory Experts for Diffusion World Models**（#1453，score 6.00）  
  - 给 diffusion world model 加短期/长期/空间记忆专家。
- **Code World Models for General Game Playing**（#1503，score 6.00）  
  - 把自然语言规则翻成可执行 Python world model，再接 MCTS。
- **Vid2World: Crafting Video Diffusion Models to Interactive World Models**（#3113，score 5.20）  
  - 把大规模视频 diffusion model 改造成 interactive world model。
- **Astra: General Interactive World Model with Autoregressive Denoising**（#3775，score 5.00）  
  - 基于 autoregressive denoising 的通用交互式视频世界模型。
- **Learning to Be Uncertain: Pre-training World Models with Horizon-Calibrated Uncertainty**（#4414，score 4.67）  
  - 显式建模“越往未来越不确定”的 world model 预训练。
- **One Life to Learn: Inferring Symbolic World Models for Stochastic Environments from Unguided Exploration**（#4428，score 4.67）  
  - 从 unguided exploration 中归纳随机环境的符号世界模型。
- **Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling**（#4515，score 4.50）  
  - 把视频 diffusion 和 3D 表征结合，提升 consistent world modeling。

### RL / 控制 / 机器人中的世界模型

- **Efficient Reinforcement Learning by Guiding World Models with Non-Curated Data**（#6，score 8.00）  
  - 用 non-curated offline data 去指导 world model，提升 online RL sample efficiency。
- **Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning**（#636，score 6.50）  
  - 把 video models/world models 微调到 visuomotor control 与 planning。
- **WorldGym: World Model as An Environment for Policy Evaluation**（#678，score 6.50）  
  - 把 world model 当 policy evaluation environment。
- **Unified Vision-Language-Action Model**（#679，score 6.50）  
  - VLA 中显式融入 world modeling，强调长时因果动态。
- **RIG: Synergizing Reasoning and Imagination in End-to-End Generalist Policy**（#911，score 6.00）  
  - 把 reasoning 与 imagination 统一进 end-to-end generalist policy。
- **Learning Massively Multitask World Models for Continuous Control**（#1121，score 6.00）  
  - 多任务连续控制 world model，大 benchmark + language-conditioned model。
- **Empowering Multi-Robot Cooperation via Sequential World Models**（#1137，score 6.00）  
  - 顺序 world models 用于多机器人协作。
- **Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control**（#1194，score 6.00）  
  - 面向 humanoid control 的大规模预训练 + world model 微调。
- **Ctrl-World: A Controllable Generative World Model for Robot Manipulation**（#1252，score 6.00）  
  - 可控、多视角、长时交互的机器人操作世界模型。
- **R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation**（#1486，score 6.00）  
  - 不依赖 decoder/augmentation 的轻量化 world model 路线。
- **Sparse Imagination for Efficient Visual World Model Planning**（#2568，score 5.50）  
  - 通过 sparse token imagination 降低 visual world model planning 成本。
- **Horizon Imagination: Efficient On-Policy Training in Diffusion World Models**（#2812，score 5.33）  
  - 扩散世界模型中的高效 on-policy 训练。
- **Object-Centric World Models from Few-Shot Annotations for Sample-Efficient Reinforcement Learning**（#2976，score 5.33）  
  - 少量标注辅助的 object-centric world model，用于 sample-efficient RL。
- **WMPO: World Model-based Policy Optimization for Vision-Language-Action Models**（#3210，score 5.00）  
  - 用 pixel-space video world model 对 VLA 做 imagined on-policy RL。
- **One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning**（#3293，score 5.00）  
  - 多任务规划里共享一个高效 world model。
- **Scalable Offline Model-Based RL with Action Chunks**（#3543，score 5.00）  
  - offline model-based RL + action chunks。
- **Deep SPI: Safe Policy Improvement via World Models**（#3882，score 5.00）  
  - 带安全改进约束的 SPI + world model。
- **WIMLE: Uncertainty‑Aware World Models with IMLE for Sample‑Efficient Continuous Control**（#3988，score 5.00）  
  - 用 IMLE/uncertainty 改善连续控制里的 world model 可靠性。
- **From Observations to Events: Event-Aware World Models for Reinforcement Learning**（#4347，score 4.67）  
  - event-aware 表征，让 world model 更鲁棒地泛化到结构相似场景。
- **Mixture-of-World Models: Scaling Multi-Task Reinforcement Learning with Modular Latent Dynamics**（#5216，score 4.00）  
  - Mixture-of-World Models，模块化 latent dynamics 做多任务扩展。

### LLM / GUI / 数字环境 Agent

- **ViMo: A Generative Visual GUI World Model for App Agents**（#894，score 6.00）  
  - 首个 visual GUI world model，直接生成未来 GUI 图像。
- **Speech World Model: Causal State–Action Planning with Explicit Reasoning for Speech**（#2988，score 5.33）  
  - Speech world model，把语音理解表述为 state-action planning。
- **Dual-Scale World Models for LLM Agents towards Hard-Exploration Problems**（#3544，score 5.00）  
  - LLM agent 的双尺度世界模型 / world memory，用于 hard exploration。
- **R-WoM: Retrieval-augmented World Model For Computer-use Agents**（#3843，score 5.00）  
  - 检索增强 world model，给 computer-use agent 补充外部教程知识。

### 自动驾驶世界模型

- **FlowAD: Ego-Scene Interactive Modeling for Autonomous Driving**（#476，score 6.50）  
  - 自动驾驶的 ego-scene 交互式建模，带明显 world model 色彩。
- **DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving**（#492，score 6.50）  
  - 驾驶 world models 的综合 benchmark。
- **DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving**（#2231，score 5.50）  
  - world modeling 作为 VLA 驾驶模型的 dense self-supervision。
- **Rethinking Driving World Model as Synthetic Data Generator for Perception Tasks**（#2271，score 5.50）  
  - 把 driving world model 当合成数据生成器重新审视其对 perception 的价值。
- **ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving**（#4267，score 4.67）  
  - Temporal Residual World Model，抓动态对象，服务端到端 driving planning。
- **ConsisDrive: Identity-Preserving Driving World Models for Video Generation by Instance Mask**（#4446，score 4.50）  
  - identity-preserving driving world model，偏视频生成一致性。

### 评测 / 数据集 / world-modeling 视角的基准

- **OmniWorld: A Multi-Domain and Multi-Modal Dataset for 4D World Modeling**（#3031，score 5.33）  
  - 多领域多模态 4D world modeling dataset。
- **ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction**（#4173，score 4.80）  
  - 用 egocentric interaction 来评测 embodied cognition / world modeling 能力。

### 跨学科 / 邻接方向

- **ChronoEdit: Towards Temporal Reasoning for In-Context Image Editing and World Simulation**（#816，score 6.00）  
  - 把 image editing 重构为具时间推理的 world simulation 问题。
- **VCWorld: A Biological World Model for Virtual Cell Simulation**（#4590，score 4.50）  
  - 把 biological knowledge + LLM reasoning 结合成 virtual cell world model。
- **MicroVerse: A Preliminary Exploration Toward a Micro-World Simulation**（#5102，score 4.00）  
  - 微观现象的视频生成/微世界模拟，邻接 world simulation。
