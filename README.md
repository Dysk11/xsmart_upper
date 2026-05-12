# X-SmartCar 上位机视觉巡线项目

## 项目用途

这是一个面向“全国大学生智能汽车竞赛 X-SmartCar 人工智能模型组”的 Python 上位机项目，运行平台为 RK3588S Linux + Python 3.10+。

项目职责聚焦在上位机主链路：

- 从摄像头或视频文件读取图像；
- 在 AR 混合赛道中识别蓝色航道；
- 提取航道中心线并计算横向误差、航向误差、曲率、置信度；
- 生成目标速度、目标转向等高层控制量；
- 通过桥接层发送给下位机；
- 为后续扩展目标检测 / OCR / 红绿灯 / 金币规划模块预留接口。

注意：

- 下位机 TC264 已经由队友完成位置环、速度环等底层闭环；
- 本项目**不实现**底层 PID、PWM 输出、电机闭环；
- 上位机只输出高层目标量，例如 `target_speed` 和 `steer_deg`；
- 若后续需要切换为 TC264 的真实协议，只需替换 `core/protocol.py` 与 `core/bridge.py`。

## 目录结构

```text
xsmart_upper/
  main.py
  config/
    config.yaml
  core/
    camera.py
    preprocess.py
    lane_detector.py
    lane_tracker.py
    planner.py
    bridge.py
    protocol.py
    logger.py
    visualizer.py
  utils/
    math_utils.py
    image_utils.py
    fps.py
  tests/
    demo_video_test.py
  README.md
  requirements.txt
```

## 安装方法

建议先进入项目目录，再创建虚拟环境：

```bash
cd xsmart_upper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果 RK3588S 上已经有系统 Python，也可以直接安装：

```bash
pip install -r requirements.txt
```

## 配置文件说明

配置文件位于 `config/config.yaml`，所有关键参数均集中在这里，便于现场调参。

### 1. `camera`

- `mode`: 图像源模式，`camera` 或 `video`
- `device_id`: 摄像头设备号
- `video_path`: 视频文件路径
- `loop_video`: 视频回放是否循环
- `width` / `height` / `fps`: 采集分辨率与目标帧率
- `mirror`: 是否镜像翻转
- `reconnect_interval_sec`: 读取失败后的重连间隔
- `max_reconnect_attempts`: 最大重连次数

### 2. `preprocess`

- `resize`: 缩放开关与目标尺寸
- `roi`: ROI 区域比例，建议重点覆盖赛道中下部
- `gaussian_blur`: 高斯模糊开关与核大小
- `clahe`: 局部对比度增强参数
- `brightness_normalization`: 亮度归一化开关与目标亮度

### 3. `detector`

- `color_space`: 蓝色检测使用 `hsv` 或 `lab`
- `hsv.lower` / `hsv.upper`: HSV 阈值
- `lab.lower` / `lab.upper`: Lab 阈值
- `morphology`: 开闭运算和腐蚀膨胀参数
- `connected_components`: 连通域筛选参数
- `centerline`: 分层扫描、单侧边界推断、拟合相关参数
- `confidence`: 置信度与丢线阈值

### 4. `tracker`

- `ema_alpha`: 常规帧平滑权重
- `recovery_alpha`: 丢线恢复后的加速收敛权重
- `confidence_gate`: 高置信度阈值
- `max_prediction_frames`: 丢线时最多允许使用历史预测的帧数

### 5. `planner`

- `lateral_gain` / `heading_gain` / `curvature_gain`: 高层转向合成权重
- `base_speed` / `max_speed` / `min_speed`: 速度策略范围
- `straight_boost_speed`: 直道附加速度
- `lost_speed`: 丢线时保守速度

### 6. `bridge`

- `type`: `mock` 或 `serial`
- `serial.port`: 串口名，例如 `/dev/ttyS4`
- `serial.baudrate`: 波特率
- `serial.timeout`: 串口超时

### 7. `visualizer`

- `show_window`: 是否显示调试窗口
- `save_video`: 是否保存调试视频
- `save_screenshot`: 是否允许按 `s` 保存截图
- `save_dir`: 调试视频与截图输出目录

### 8. `logger`

- `enable`: 是否记录 CSV
- `output_dir`: 日志输出目录

### 9. `extensions`

- `target_detector` / `ocr` / `traffic_light` / `coin_planner`: 预留扩展节点
- 当前默认均为 `enable: false`
- 后续可在 `main.py` 的 `_collect_future_module_hints()` 中接入这些模块输出
- 扩展模块如果只想影响高层策略，建议统一转换成 `core/planner.py` 中的 `ModuleHints`

## 如何运行实时摄像头模式

默认配置就是摄像头模式，先确认 `config/config.yaml` 中的 `camera.device_id` 和串口参数正确，然后运行：

```bash
python main.py
```

如果想强制使用串口桥接：

```bash
python main.py --bridge serial
```

如果现场只想看串口文本打印，不发送真实串口：

```bash
python main.py --bridge mock
```

## 如何运行视频回放模式

方法一：直接使用主程序切换到视频模式

```bash
python main.py --mode video --video /path/to/demo.mp4 --bridge mock
```

方法二：使用专门的视频测试脚本

```bash
python tests/demo_video_test.py --video /path/to/demo.mp4
```

如果没有显示器或不想弹窗：

```bash
python tests/demo_video_test.py --video /path/to/demo.mp4 --no-gui --save-video
```

## HSV 实时调参工具

如果你现在只有摄像头，还没有正式赛道，最推荐先用 HSV 调参工具把蓝色阈值调准。

运行命令：

```bash
python tools/hsv_tuner.py
```

如果想指定摄像头设备号：

```bash
python tools/hsv_tuner.py --device-id 1
```

如果想对视频离线调参：

```bash
python tools/hsv_tuner.py --mode video --video /path/to/demo.mp4
```

窗口说明：

- 左上：原始画面 + ROI 范围
- 右上：ROI 区域 + 原始 HSV 掩膜
- 左下：筛选后的主航道掩膜
- 右下：当前参数和快捷键说明

滑块建议优先调这些：

- `ROI_top(%)`：先把 ROI 压到只看地面，减少桌面和背景干扰
- `H_low/H_high`：先决定蓝色大致色相范围
- `S_low`：提高后可减少灰蓝、暗蓝误检
- `V_low`：提高后可减少阴影和黑色区域误检
- `MinArea/MinHeight`：去掉零碎小蓝块

快捷键：

- `Q` / `Esc`：退出
- `P`：把当前推荐 YAML 片段打印到终端
- `S`：把当前参数快照保存到 `outputs/hsv_tuner_last_snippet.yaml`

## 调试画面说明

调试窗口默认包含 4 个区域：

- 原图 + ROI 框 + 中心线；
- ROI + 蓝色掩膜 + 中心线；
- 主航道二值掩膜；
- 误差、曲率、置信度、FPS、速度/转向指令等文字信息。

按键说明：

- `q` / `Esc`: 退出程序
- `s`: 保存当前截图

## 默认协议说明

默认协议位于 `core/protocol.py`，使用一行文本形式，便于调试与抓串口日志。

字段包括：

- `ts_ms`
- `mode`
- `target_speed`
- `steer_deg`
- `lateral_error_px`
- `heading_error_deg`
- `curvature`
- `confidence`
- `is_lane_lost`

示例：

```text
ts_ms=1711111111111,mode=NORMAL,target_speed=1.200,steer_deg=-3.500,lateral_error_px=12.300,heading_error_deg=-1.200,curvature=0.004500,confidence=0.860,is_lane_lost=0
```

## 如何对接 TC264

后续与 TC264 联调时，请明确以下边界：

- TC264 已经实现底层速度环、位置环等闭环；
- 本项目只输出高层目标控制量；
- 不要在上位机重复实现底层 PID；
- 真正协议请集中修改 `core/protocol.py` 和 `core/bridge.py`。

推荐对接步骤：

1. 先保持 `planner.py` 输出 `target_speed` 与 `steer_deg` 不变；
2. 在 `core/protocol.py` 中把文本协议改为 TC264 真正需要的定长帧或二进制帧；
3. 在 `core/bridge.py` 中保留 `BaseVehicleBridge` 接口，重写 `SerialBridge.send()` 的发送细节；
4. 下位机根据接收到的高层目标量执行自身闭环控制。

如果后续要插入额外高层模块，推荐这样接：

1. 在 `main.py` 的 `_collect_future_module_hints()` 中汇总 OCR、红绿灯、金币规划等输出；
2. 将这些输出统一整理为 `ModuleHints`；
3. 交给 `planner.py` 做最终高层速度/转向融合；
4. 保持下位机仍只接收高层目标量，不破坏现有闭环边界。

## 巡线算法设计说明

第一版方案完全采用传统视觉，便于在 RK3588S 上稳定部署：

- 颜色空间阈值分割：支持 HSV / Lab；
- 形态学开闭运算：去掉蓝色噪点、填补小孔洞；
- 连通域筛选：抑制误检小蓝块；
- 分层扫描：逐行提取蓝色主航道中心；
- 单侧边界推断：当局部只看到一侧边界时，借助上一帧宽度估计中心；
- 二次曲线拟合：得到更平滑的中心线；
- EMA 时序平滑：减小抖动，并在短时丢线时提供预测补偿；
- 高层速度策略：直道提速、弯道减速、低置信度降速、丢线保守模式。

## 输出日志

CSV 日志默认保存在 `outputs/logs/`，字段包括：

- 时间戳
- 横向误差
- 航向误差
- 曲率
- 置信度
- 目标速度
- 目标转向
- 丢线计数

这些数据适合赛后离线分析与调参。
