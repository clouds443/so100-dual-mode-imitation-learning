# SO100 Dual-Mode Imitation Learning

基于 LeRobot / GenkiBot 改造的 SO100 机械臂模仿学习项目。

本项目保留原有 **主臂 teacher / leader -> 从臂 student / follower** 的遥操作采集与训练流程，同时新增 **只使用从臂** 的手拖示教流程。也就是说，即使没有主臂，也可以只靠学生端机械臂完成：

```text
手拖示教采集数据 -> 训练 ACT 模仿学习策略 -> 加载模型控制真实机械臂
```

## 核心特性

- **双模式采集**：支持传统主从遥操作 `teleoperate`，也支持从臂单独手拖示教 `kinesthetic`。
- **从臂单独模仿学习**：录制时关闭从臂非夹爪关节扭矩，用手拖动机械臂生成示教轨迹，不需要额外购买 teacher / leader 主臂。
- **保持 LeRobot 数据格式**：采集数据仍包含 `observation.state`、双相机 `observation.images.*` 和 `action`，可继续使用原有 ACT 训练和评估流程。
- **双相机输入**：默认使用 `laptop` 与 `phone` 两路 OpenCV 摄像头。
- **夹爪键盘控制**：从臂手拖采集时可长按 `O` 打开夹爪、长按 `C` 关闭夹爪，夹爪轨迹会自然写入第 6 维 action。
- **上机安全限制**：学生端配置 `so100_student.yaml` 设置 `max_relative_target: 5`，限制模型控制时每步关节目标变化。

## 适用硬件

### 模式一：主从遥操作

- teacher / leader 主臂
- student / follower 从臂
- 摄像头：可选，默认支持 laptop + phone 双相机

使用配置：

```text
lerobot/configs/robot/so100.yaml
```

### 模式二：只有从臂

- 只需要 student / follower 从臂
- Feetech `sts3215` 电机
- 默认串口：`com5`
- 摄像头：`camera_index: 0` 和 `camera_index: 1`

使用配置：

```text
lerobot/configs/robot/so100_student.yaml
```

## 环境安装

建议使用 Python 3.10 和 conda 环境：

```powershell
conda create -n hostprotocol python=3.10
conda activate hostprotocol

cd <repo-dir>
pip install -e .
pip install -r requirements.txt
pip install -e ".[feetech]"
```

如果需要在本机使用 GPU 加载模型，建议使用与本项目兼容的 PyTorch 组合：

```text
torch==2.6.0
torchvision==0.21.0
```

注意：本项目的视频读取流程依赖 `torchvision.io.VideoReader`，请避免安装 `torchvision>=0.22`。

## 只有从臂的完整流程

进入项目并设置环境变量：

```powershell
cd <repo-dir>
conda activate hostprotocol
$env:PYTHONUTF8="1"
$env:PYTHONPATH=(Resolve-Path ".").Path
```

### 1. 校准学生端从臂

```powershell
python lerobot/scripts/control_robot.py calibrate `
  --robot-path lerobot/configs/robot/so100_student.yaml `
  --arms main_follower
```

### 2. 手拖示教采集数据

录制时请用手扶住机械臂。程序会关闭非夹爪关节扭矩，机械臂可能下垂或松动。

```powershell
python lerobot/scripts/control_robot.py record `
  --robot-path lerobot/configs/robot/so100_student.yaml `
  --record-control-mode kinesthetic `
  --fps 30 `
  --root data `
  --repo-id pick/so100_student `
  --tags pickbottle `
  --warmup-time-s 5 `
  --episode-time-s 30 `
  --reset-time-s 6 `
  --num-episodes 50 `
  --kinesthetic-gripper-keyboard 1 `
  --gripper-open-angle 90 `
  --gripper-closed-angle -90 `
  --gripper-open-key o `
  --gripper-close-key c `
  --gripper-speed 30 `
  --push-to-hub 0
```

夹爪控制：

- 长按 `O`：逐步打开夹爪
- 长按 `C`：逐步关闭夹爪
- 松开按键：停止夹爪运动
- 如果夹爪方向相反，交换 `--gripper-open-angle` 和 `--gripper-closed-angle`

### 3. 查看数据

```powershell
python lerobot/scripts/visualize_dataset.py `
  --repo-id pick/so100_student `
  --root ./data `
  --local-files-only 1 `
  --episode-index 0
```

### 4. 回放数据

```powershell
python lerobot/scripts/control_robot.py replay `
  --robot-path lerobot/configs/robot/so100_student.yaml `
  --fps 30 `
  --root data `
  --repo-id pick/so100_student `
  --episode 0 `
  --robot-overrides "~cameras"
```

### 5. 训练 ACT 模型

```powershell
$env:DATA_DIR="./data"
python lerobot/scripts/train.py `
  dataset_repo_id=pick/so100_student `
  policy=act_so100_real `
  env=so100_real `
  hydra.run.dir=outputs/train/act_so100_student `
  hydra.job.name=act_so100_student `
  device=cuda `
  wandb.enable=false
```

### 6. 让模型控制真实从臂

```powershell
python lerobot/scripts/control_robot.py record `
  --robot-path lerobot/configs/robot/so100_student.yaml `
  --fps 30 `
  --root data `
  --repo-id pick/eval_so100_student `
  --tags pickbottle `
  --warmup-time-s 5 `
  --episode-time-s 300 `
  --reset-time-s 10 `
  --num-episodes 2 `
  --push-to-hub 0 `
  -p outputs/train/act_so100_student/checkpoints/last/pretrained_model
```

## 主从遥操作流程

如果你同时拥有主臂和从臂，可以继续使用传统 teacher / student 遥操作方式：

```powershell
python lerobot/scripts/control_robot.py teleoperate `
  --robot-path lerobot/configs/robot/so100.yaml `
  --robot-overrides "~cameras" `
  --display-cameras 0
```

主从采集数据：

```powershell
python lerobot/scripts/control_robot.py record `
  --robot-path lerobot/configs/robot/so100.yaml `
  --fps 30 `
  --root data `
  --repo-id pick/so100_test `
  --tags pickbottle `
  --episode-time-s 30 `
  --reset-time-s 6 `
  --num-episodes 50 `
  --push-to-hub 0
```


## 更多操作说明

完整命令、排错记录和数据采集细节见：

```text
操作文档.md
```

## 开源许可

本项目基于 LeRobot / GenkiBot 及相关开源项目改造，请遵守各项目许可证要求。主要依赖项目包括：

- LeRobot：Apache License 2.0
- Diffusion Policy：MIT License
- ALOHA：MIT License
- DETR：Apache License 2.0

详情请查看本仓库中的 `LICENSE` 文件。
