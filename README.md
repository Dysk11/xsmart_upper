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

- `mode`: 图像源模式，`camera`、`video` 或 `shared_memory`
- `device_id`: 摄像头设备号
- `video_path`: 视频文件路径
- `shared_memory_name`: AR 系统发布 RGB888 帧的 POSIX 共享内存名称，默认 `shm_ar_video`
- `loop_video`: 视频回放是否循环
- `width` / `height` / `fps`: 采集分辨率与目标帧率
- `mirror`: 是否镜像翻转
- `reconnect_interval_sec`: 读取失败后的重连间隔
- `max_reconnect_attempts`: 最大重连次数

共享内存模式与 AR 系统运行在同一台 Linux 设备上，配置示例：

```yaml
camera:
  mode: shared_memory
  shared_memory_name: shm_ar_video
  mirror: false
  reconnect_interval_sec: 0.5
  max_reconnect_attempts: 5
```

共享内存使用 16 字节原生 `QII` 头部（帧号、宽、高），后接连续 RGB888 图像。接收端会转换为项目下游统一使用的 BGR 图像。

### 2. `lane_geometry`

- `roi`: 直接在相机帧上定义巡线 ROI，应覆盖赛道中下部
- `boundary`: 逐行边界跟踪、梯度限幅和短时丢线补偿参数
- `temporal_filter.weights`: 三帧加权滤波系数，默认 `[0.20, 0.50, 0.30]`
- `centerline`: 中心线采样、有效点数、默认赛道宽度和前视比例
- `confidence`: 车道置信度和丢线判定参数
- `fork`: 左右边界拐点、外移阈值及岔路确认/释放帧数

### 3. `tracker`

- `ema_alpha`: 常规帧平滑权重
- `recovery_alpha`: 丢线恢复后的加速收敛权重
- `confidence_gate`: 高置信度阈值
- `max_prediction_frames`: 丢线时最多允许使用历史预测的帧数

### 4. `planner`

- `lateral_gain` / `heading_gain` / `curvature_gain`: 高层转向合成权重
- `base_speed` / `max_speed` / `min_speed`: 速度策略范围
- `straight_boost_speed`: 直道附加速度
- `lost_speed`: 丢线时保守速度

### 5. `bridge`

- `type`: `mock` 或 `serial`
- `serial.port`: 串口名，例如 `/dev/ttyS4`
- `serial.baudrate`: 波特率
- `serial.timeout`: 串口超时

### 6. `visualizer`

- `show_window`: 是否显示调试窗口
- `save_video`: 是否保存调试视频
- `save_screenshot`: 是否允许按 `s` 保存截图
- `save_dir`: 调试视频与截图输出目录

### 7. `logger`

- `enable`: 是否记录 CSV
- `output_dir`: 日志输出目录

## 如何运行实时摄像头模式

默认配置就是摄像头模式，先确认 `config/config.yaml` 中的 `camera.device_id` 和串口参数正确，然后运行：

```bash
python main.py
```

## 如何运行视频回放模式

方法一：直接使用主程序切换到视频模式

```bash
python main.py --mode video --video /path/to/demo.mp4 --bridge mock
```

## HSV 实时调参工具

如果你现在只有摄像头，还没有正式赛道，最推荐先用 HSV 调参工具把蓝色阈值调准。

运行命令：

```bash
python tools/hsv_tuner.py
```

快捷键：

- `Q` / `Esc`：退出
- `P`：把当前推荐 YAML 片段打印到终端
- `S`：把当前参数快照保存到 `outputs/hsv_tuner_last_snippet.yaml`

---

# RKNN 目标识别、避障、吃 coin 功能

本章节说明当前工程里和 `rknn_7classes.rknn` 模型、测试视频、`car/human` 避障、`coin` 目标追踪相关的实现细节。

## 1. 当前已接入的内容

当前工程已经完成以下接入：

- 模型文件：`rknn_7classes.rknn`
- 测试视频：`outputs/video/cbf977c5bd5978922b972f4f0285c0bd.mp4`
- RKNN 推理模块：`core/rknn_object_detector.py`
- coin 目标规划模块：`core/gold_target_planner.py`
- 避障判断模块：`core/blocking_analyzer.py`
- 避障目标规划模块：`core/avoidance_target_planner.py`
- 主流程入口：`main.py`
- 主要配置文件：`config/config.yaml`

默认配置已经把图像源设置成视频模式：

```yaml
camera:
  mode: video
  video_path: outputs/video/cbf977c5bd5978922b972f4f0285c0bd.mp4
```

模型配置如下：

```yaml
rknn_object_detector:
  enable: true
  model_path: models/rknn_7classes.rknn
  # 按 [width, height] 配置，与 RKNN 固定输入 640x480 一致
  input_size: [640, 480]
  class_names: [car, coin, Go, human, road_sign, speed_limit, Stop]
  # 单核运行入口，可选 NPU_CORE_0、NPU_CORE_1、NPU_CORE_2
  core_mask: NPU_CORE_0
```

## 2. 类别顺序

模型实际输出的是类别编号。当前配置：

- `0 = car`
- `1 = coin`
- `2 = Go`
- `3 = human`
- `4 = road_sign`
- `5 = speed_limit`
- `6 = Stop`

如果训练模型时的类别顺序不同，必须在 `config/config.yaml` 中修改 `class_names`。

## 3. 控制优先级

当前主逻辑优先级是：

```text
car/human 避障 > 吃 coin > 普通巡线
```

注意：避障模块会在巡线中心线上加一个平滑偏移，生成新的目标点。最终仍然只发送一组 `lateral_error_px` 和 `steer_deg` 给下位机。

## 4. coin 目标逻辑

- `class_names: [coin]`：只有识别类别名为 `coin` 的目标才触发吃金币逻辑。
- `approach_speed_limit: 0.85`：朝 coin 走时限制速度。
- `aim_at: bottom_center`：目标点取 coin 框的底部中心。

当且仅当没有障碍物阻挡时，`GOLD` 模式生效。

## 5. car/human 避障逻辑

1. `core/blocking_analyzer.py` 判断识别框是否挡住当前航道危险走廊（`corridor_half_width_px`）。
2. `core/avoidance_target_planner.py` 根据阻挡位置生成偏移后的目标路线。

避障只对 `car` 和 `human` 生效，`coin` 不会被当作障碍物。

## 6. 主流程顺序

1. 读取图像 -> 2. ROI 裁剪/预处理 -> 3. 蓝色航道巡线 -> 4. RKNN 目标识别 -> 5. 避障判断 -> 6. 决策规划（避障 > 金币 > 巡线） -> 7. 生成协议帧发送。

## 7. 如何确认功能正常

1. 终端出现 `RKNN detector loaded`；
2. 调试画面中出现 `coin/car/human` 目标框；
3. 画面上 `G` 为 coin 目标点，`A` 为最终控制目标点；
4. 阻挡时模式显示 `avoid_left/right` 或 `too_close`，吃金币时显示 `GOLD`。

---

## 默认协议说明

默认协议位于 `core/protocol.py`，使用固定 7 字节二进制帧。高层 payload 仍包含
`target_speed` 等规划字段，协议层新增 `motion_flag` 并将其写入帧尾状态字节。

## UART 通信协议说明

本项目通过 CH340 串口 USB 转 TTL 模块将树莓派/RK3588 与下位机（如 Arduino 或 TC264）连接。

- **配置**: 115200 8N1 (115200 波特率, 8 数据位, 无校验位, 1 停止位)
- **帧结构**: 2 字节帧头 + 2 字节误差 (Int16) + 2 字节转向角度 (Int16) + 1 字节运行状态

| 字节偏移 | 长度 | 定义 | 说明 |
| :--- | :--- | :--- | :--- |
| 0 | 1 | 帧头 1 | 固定为 `0xAA` |
| 1 | 1 | 帧头 2 | 固定为 `0x55` |
| 2 | 2 | 横向误差 | `lateral_error_px` 转为 Int16 (大端序) |
| 4 | 2 | 转向角度 | `steer_deg` 转为 Int16 (大端序) |
| 6 | 1 | 运行状态 | bit0：停车为 `0`，不停为 `1`；bit1..bit7 固定为 `0` |

示例代码：
```python
data = bytearray([
    0xAA, 0x55,
    (error >> 8) & 0xFF, error & 0xFF,
    (angle >> 8) & 0xFF, angle & 0xFF,
    motion_flag & 0x01
])
ser.write(data)
```

TC264 必须按固定 7 字节重新解包；继续按旧的 6 字节步长读取会导致后续帧错位。

## RKNN 航道分割部署

当前主巡线链路在 RK3588 上使用单类 `track` 的 YOLOv5n-seg INT8 模型生成 mask。应用层不再执行缩放、增强或颜色预处理；模型内部仍保留必需的 RGB/letterbox 输入转换。mask 在 ROI 中逐行提取左右边界、滤波中线并识别左右岔路。

- 模型：`models/yolov5n_seg_track_480x640_int8_rk3588.rknn`
- SHA-256：`0ffd0f431505fa362b4d1f4a94ae69321b2c77a4081c6a919f758f28712b1dce`
- 输入：RGB uint8 NHWC `[1, 480, 640, 3]`；模型内部完成 `/255` 归一化。
- 输出：三组 box/class、三组 32 维 mask coefficient 和一个 `[1, 32, 120, 160]` prototype。
- 类别：`0 = track`；默认置信度阈值 `0.25`、NMS IoU `0.45`、mask 阈值 `0.5`。
- 转换环境：RKNN-Toolkit2 `2.3.2`，目标平台 `rk3588`，W8A8 per-channel INT8。

板端应安装与 Toolkit2 2.3.2 兼容的 RKNN Toolkit Lite2 和 RKNPU2 Runtime。准备好依赖后运行：

```bash
python3 main.py --mode camera --bridge serial
```

正常启动时终端会输出 `RKNN lane segmenter loaded`。调试窗口在 ROI 内半透明显示 `track` mask，并显示 `track: ok conf=...`。加载或推理失败不会静默回退到 HSV，而会输出一次明确告警并按丢线处理。默认不会保存视频或截图。

### RKNN 性能测试与核分配

RK3588 上不要假设固定核分配一定最优。项目提供独立 benchmark，可在不修改主配置的情况下比较单核、组合核、多实例和目标检测抽帧策略：

```bash
python3 tools/benchmark_rknn.py \
  --video outputs/video/record_20260708_135111.mp4 \
  --output outputs/benchmark/rknn_benchmark.json
```

快速筛选可增加 `--scout`，单模型核缩放测试可增加 `--single-models --scout`。正式测试默认预热 200 帧、每种策略运行 120 秒并重复 3 次，同时记录 FPS、P50/P95/P99 延迟、结果帧龄、NPU 负载、频率和温度。

当前板端正式测试的最佳低延迟策略为三个车道分割实例分别绑定 `NPU_CORE_0/1/2`，目标检测绑定 `NPU_CORE_1` 并设置 `inference_stride: 2`。在固定测试视频上，未优化基线为 `72.558 FPS / 51.324 ms lane P95`，最佳方案为 `92.787 FPS / 45.083 ms lane P95`，吞吐提升约 `27.9%`。该结果应在模型、Runtime 或视频输入变化后重新测试。

Windows 端可使用 `tools/run_orangepi_benchmark.ps1 -Scout` 通过专用 SSH 密钥启动板端测试并拉回 JSON/CSV。benchmark 不读取或修改板端 `config/config.yaml`。

## 如何对接 TC264

1. TC264 已经实现底层闭环，本项目只输出高层目标量；
2. 若需修改协议，请集中修改 `core/protocol.py` 和 `core/bridge.py`；
3. 扩展模块输出建议统一整理为 `ModuleHints` 交给 `planner.py`。

## 巡线算法设计说明

- 从车体附近向远处扫描 `track` mask，每行跟踪与历史中心最连续的前景区间；
- 根据左右边界得到原始中线，依次执行梯度限幅和五点滑动平均；
- 把 120 行赛道权重重采样到当前 ROI，计算加权横向误差并做三帧时域滤波；
- 通过左右边界外移、拐点、丢线统计和连续帧确认独立上报左/右岔路，默认仍沿当前主路行驶；
- 避障、coin 目标和丢线历史恢复仍在原有优先级链中工作。

## RK3588 PP-OCR 路牌识别

目标检测在同一帧识别到 `road_sign` 且原始框至少为 `96x48` 像素时，AI
子进程会按检测框中心扩展 10% 并送入 PP-OCRv4 Det/Rec RKNN 模型。只有整体
置信度达到 `0.60` 的非空文字才写入
`outputs/logs/ocr/ocr_events_YYYYMMDD_HHMMSS.jsonl`；成功后按
`extensions.ocr.cooldown_seconds` 配置全局 OCR 冷却时间，默认 20 秒。
每次 OCR 尝试都会在控制台输出 `[OCR]` 行，并在调试窗口用紫色框标出裁剪区域；
紫框默认显示 1 秒后清除，最新文字、置信度和耗时继续保留。低分候选只显示，
不写 JSONL，也不启动冷却。
模型、阈值、NPU 核和输出目录均在 `config/config.yaml` 的
`extensions.ocr` 中配置。板端只需要 RKNN-Toolkit-Lite2，不使用 ONNX 或
PaddlePaddle；额外 Python 依赖为 `shapely`、`pyclipper` 和 `six`。

## RK3588 千帆岔路决策

高置信度 OCR 事件会由独立进程发送给千帆 `ernie-4.5-turbo-vl`，HTTP 请求不会阻塞
摄像头、目标检测或巡线。模型回答必须严格为 `left` 或 `right`。无有效结果时默认选择
左侧；请求未完成时车辆可以继续接近岔路，但到达岔路后会以 `QIANFAN_WAIT` 模式
停车。最终结果从主进程收到时开始按 `decision_ttl_sec` 计时，默认 20 秒，期间所有
岔路遵循该方向，到期恢复左侧。

启动前在板端设置 API Key：

```bash
export QIANFAN_API_KEY='你的 API Key'
python3 main.py --no-gui
```

`config/config.yaml` 的 `extensions.qianfan_route` 提供 API 地址、模型、API Key 环境
变量名、连接超时、读取超时、最大尝试次数、重试间隔、默认方向、结果有效时间和
API 日志目录。超时与结果有效时间必须大于 0，最大尝试次数至少为 1；缺少 API Key
或鉴权失败时不会把密钥打印到日志。

每次尝试和最终决策都会输出 `[QIANFAN]` 日志，并保存到
`outputs/logs/api/api_events_YYYYMMDD_HHMMSS.jsonl`。记录包括 OCR 事件号、问题、
尝试次数、配置超时、HTTP 状态、耗时、原始回答、解析方向、错误、是否回退和过期
时间，不包含 API Key。有效期内识别到新路牌时会发起新请求，新结果覆盖旧方向并
重新计时。

## 输出日志

CSV 日志保存在 `outputs/logs/`，包含误差、曲率、置信度、目标速度/转向等，适合离线分析。
