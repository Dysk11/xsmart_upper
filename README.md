# X-SmartCar 上位机视觉巡线项目

## 项目用途

这是一个面向“全国大学生智能汽车竞赛 X-SmartCar 人工智能模型组”的 Python 上位机项目，运行平台为 RK3588S Linux + Python 3.10+。

项目职责聚焦在上位机主链路：

- 从摄像头或视频文件读取图像；
- 在 AR 混合赛道中识别蓝色航道；
- 提取航道中心线并计算横向误差、航向误差和置信度；
- 生成目标速度、目标转向等高层控制量；
- 通过桥接层发送给下位机；
- 为后续扩展目标检测 / OCR / 红绿灯 / 金币规划模块预留接口。

注意：

- 下位机 TC264 已经由队友完成位置环、速度环等底层闭环；
- 本项目**不实现**底层 PID、PWM 输出、电机闭环；
- 上位机只输出高层目标量，例如 `target_speed` 和 `steer_deg`；
- 若后续需要切换为 TC264 的真实协议，只需替换 `core/io/protocol.py` 与 `core/io/bridge.py`。

## 目录结构

```text
xsmart_upper/
  main.py
  config.yaml
  core/
    lane/             # lane detection, tracking, and RKNN segmentation
    object/           # object detection and pedestrian safety analysis
    ocr/              # OCR recognition, road-sign handling, and bundled PPOCR runtime
      ppocr/          # RKNN runtime and executor
    planning/         # driving, target, and road-sign planning
    io/               # camera, vehicle bridge, protocol, and logging
    visualization/    # runtime visualization
    runtime/          # processes, shared memory, and app orchestration
  models/
    lane/
    object/
    ocr/              # OCR RKNN models and character dictionary
  utils/
    math_utils.py
    image_utils.py
    fps.py
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

配置文件位于 `config.yaml`，所有关键参数均集中在这里，便于现场调参。

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

共享内存使用 16 字节原生 `QII` 头部（帧号、宽、高），后接连续 RGB888 图像。接收端保留原始 RGB 供 RKNN 推理，同时只转换一次并缓存对应 BGR，供 OCR、可视化和录像直接复用。`video/camera` 则保留 OpenCV 原始 BGR，并只生成一次供两个模型共享的 RGB 推理画面。

### 2. `lane_geometry`

- `roi`: 直接在相机帧上定义巡线 ROI，应覆盖赛道中下部
- `boundary`: 逐行边界跟踪、梯度限幅和短时丢线补偿参数
- `temporal_filter.weights`: 三帧加权滤波系数，默认 `[0.20, 0.50, 0.30]`
- `centerline`: 中心线采样、有效点数、固定透视宽度和前视比例
- `confidence`: 车道置信度和丢线判定参数
- `fork`: 左右边界拐点、双中线分离阈值、候选线平滑度阈值及岔路确认/释放帧数

### 3. `tracker`

- `ema_alpha`: 常规帧平滑权重
- `recovery_alpha`: 丢线恢复后的加速收敛权重
- `confidence_gate`: 高置信度阈值
- `max_prediction_frames`: 丢线时最多允许使用历史预测的帧数

### 4. `planner`

- `lateral_gain` / `heading_gain`: 高层转向合成权重
- `base_speed` / `max_speed` / `min_speed`: 速度策略范围
- `lost_speed`: 丢线时保守速度

### 5. `bridge`

- `type`: `mock` 或 `serial`
- `drive_speed_state`: 正常行驶固定档位，`1`=低速、`2`=中速、`3`=高速
- `serial.port`: 串口名，例如 `/dev/ttyS4`
- `serial.baudrate`: 波特率
- `serial.timeout`: 串口超时

### 6. `visualizer`

- `show_window`: 是否显示调试窗口
- `debug_window_name`: 独立调试信息窗口的名称
- `debug_panel_font_size`: 独立调试窗口的双列面板字号，默认 `18`；长状态和原因会按面板列宽自动换行
- `save_video`: 是否保存调试视频
- `record_without_ui`: 是否只把摄像头原始画面写入视频；不影响窗口继续显示完整 UI 标注，缺省为 `false`
- `save_screenshot`: 是否允许按 `s` 保存截图
- `save_dir`: 调试视频与截图输出目录

巡线调试窗口会额外显示 track 实例数、最终目标点坐标和目标生成原因，用于区分分割实例切换、中心线切换与目标点外推。

### 7. `logger`

- `enable`: 是否记录 CSV
- `output_dir`: 日志输出目录

## 如何运行实时摄像头模式

默认配置就是摄像头模式，先确认 `config.yaml` 中的 `camera.device_id` 和串口参数正确，然后运行：

```bash
python main.py
```

## 如何运行视频回放模式

方法一：直接使用主程序切换到视频模式

```bash
python main.py --mode video --video /path/to/demo.mp4 --bridge mock
```

---

# RKNN 目标识别、行人停车、吃 coin 功能

本章节说明当前工程里和目标检测模型、行人目标点穿越停车、`coin` 目标追踪相关的实现细节。

## 1. 当前已接入的内容

当前工程已经完成以下接入：

- 模型文件：`rknn_7classes.rknn`
- 测试视频：`outputs/video/cbf977c5bd5978922b972f4f0285c0bd.mp4`
- RKNN 推理模块：`core/object/rknn_detector.py`
- coin 目标规划模块：`core/planning/gold_target.py`
- 行人目标点穿越模块：`core/object/pedestrian_safety.py`
- 主流程入口：`main.py`
- 主要配置文件：`config.yaml`

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
  model_path: models/object/rknn_7classes.rknn
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

如果训练模型时的类别顺序不同，必须在 `config.yaml` 中修改 `class_names`。

## 3. 控制优先级

当前主逻辑优先级是：

```text
脱轨停车 > 行人穿越等待 > 路牌分析停车等待 > Go/Stop 路径 > 吃 coin > 普通巡线
```

行人框触发目标点穿越锁存后，运行模式为 `PEDESTRIAN_WAIT`，目标速度和协议
`speed_state` 均为 0。`car` 只显示检测框，不参与车辆控制。

## 4. coin 目标逻辑

- `class_names: [coin]`：只有识别类别名为 `coin` 的目标才触发吃金币逻辑。
- `approach_speed_limit: 0.85`：朝 coin 走时限制速度。
- `aim_at: bottom_center`：目标点取 coin 框的底部中心。

当没有更高优先级停车或路径目标时，`GOLD` 模式生效。

## 5. Go/Stop 断轨连接逻辑

- `Go` 和 `Stop` 都是路径标记，`Stop` 不触发停车。
- 目标点取检测框几何中心，模式显示为 `PATH_TARGET`，不增加额外限速。
- 当标记横向切断赛道时，规划层保留下方赛道，通过检测框中心连接到上方赛道；不会修改分割 mask 或岔路检测输入。
- 连接点在检测框上下各避让 `connection_margin_px`，按 `interpolation_step_px` 生成连续路径；目标短暂漏检时最多保持 `hold_frames` 帧。
- 标记中心到达 ROI 底部 `release_y_ratio` 后释放，防止车辆驶过后继续向身后目标转向。

调试画面中紫线为 Go/Stop 连接路径，`L/U` 为下方和上方锚点，`P` 为检测框中心目标。

## 6. human 目标点穿越停车逻辑

`center_region` 由 ROI 相对横坐标配置并贯穿 ROI 全高；其左侧是 `left`
区域，右侧是 `right` 区域。完整 `human` 框面积达到 `min_box_area_px: 600`
且框中心位于 ROI 时，车辆选择面积最大的合格行人并立即停车。

```yaml
pedestrian_safety:
  enabled: true
  min_box_area_px: 600
  rearm_cooldown_sec: 3.0
  center_region:
    left_ratio: 0.30
    right_ratio: 0.70
```

停车时冻结普通巡线目标点 x 和它所属的 `left/center/right` 区域。后续结果
始终选择距触发行人上一中心最近的 human 框，不设置关联距离上限；漏检时无限
保持停车。center 目标允许任意方向跨越，left 目标只接受右到左，right 目标只
接受左到右。完成穿越后立即恢复并进入 3 秒冷却，冷却期间 human 不再触发。
`car` 不触发停车或避让。

## 7. 主流程顺序

1. 读取图像 -> 2. ROI 裁剪/预处理 -> 3. 蓝色航道巡线 -> 4. RKNN 目标识别 -> 5. 行人目标点穿越判断 -> 6. 决策规划（停车 > Go/Stop 路径 > 金币 > 巡线） -> 7. 生成协议帧发送。

## 8. 如何确认功能正常

1. 终端出现 `RKNN detector loaded`；
2. 调试画面中出现 `coin/Go/Stop/car/human` 目标框；
3. 画面上 `G` 为 coin 目标点，`P` 为 Go/Stop 框中心，两条黄线分隔 `LEFT/CENTER/RIGHT`，停车时红线为冻结目标 x；
4. 行人停车时模式显示 `PEDESTRIAN_WAIT`，Go/Stop 路径显示 `PATH_TARGET`，吃金币时显示 `GOLD`。

---

## 默认协议说明

默认协议位于 `core/io/protocol.py`，使用固定 7 字节二进制帧。高层 payload 仍包含
`target_speed` 等规划字段仍用于内部策略与日志；运行时根据
`bridge.drive_speed_state` 写入帧尾字节的低 2 位，停车命令固定写入 `0x00`。

## UART 通信协议说明

本项目通过 CH340 串口 USB 转 TTL 模块将树莓派/RK3588 与下位机（如 Arduino 或 TC264）连接。

- **配置**: 115200 8N1 (115200 波特率, 8 数据位, 无校验位, 1 停止位)
- **帧结构**: 2 字节帧头 + 2 字节误差 (Int16) + 2 字节转向角度 (Int16) + 1 字节速度状态

| 字节偏移 | 长度 | 定义 | 说明 |
| :--- | :--- | :--- | :--- |
| 0 | 1 | 帧头 1 | 固定为 `0xAA` |
| 1 | 1 | 帧头 2 | 固定为 `0x55` |
| 2 | 2 | 横向误差 | `lateral_error_px` 转为 Int16 (大端序) |
| 4 | 2 | 转向角度 | `steer_deg` 转为 Int16 (大端序) |
| 6 | 1 | 速度状态 | 低 2 位：`0x00` 停止、`0x01` 低速、`0x02` 中速、`0x03` 高速；高 6 位固定为 `0` |

示例代码：
```python
data = bytearray([
    0xAA, 0x55,
    (error >> 8) & 0xFF, error & 0xFF,
    (angle >> 8) & 0xFF, angle & 0xFF,
    speed_state & 0x03
])
ser.write(data)
```

TC264 必须按固定 7 字节重新解包；继续按旧的 6 字节步长读取会导致后续帧错位。

正常行驶档位通过 `bridge.drive_speed_state` 配置，只允许 `1`（低速）、`2`（中速）
或 `3`（高速），默认值为 `2`。无论正常档位为何值，只要规划结果要求停车，第 7
字节都会发送 `0x00`。

## RKNN 航道分割部署

当前主巡线链路在 RK3588 上使用单类 `track` 的 YOLOv5n-seg INT8 模型生成 mask。应用层不再执行缩放、增强或颜色预处理；模型内部仍保留必需的 RGB/letterbox 输入转换。mask 在 ROI 中逐行提取左右边界、滤波中线并识别左右岔路。

- 模型：`models/lane/yolov5n_seg_track_480x640_int8_rk3588.rknn`
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

## 如何对接 TC264

1. TC264 已经实现底层闭环，本项目只输出高层目标量；
2. 若需修改协议，请集中修改 `core/io/protocol.py` 和 `core/io/bridge.py`；
3. 扩展模块输出建议统一整理为 `ModuleHints` 交给 `planner.py`。

## 巡线算法设计说明

- 从车体附近向远处扫描 `track` mask，每行跟踪与历史中心最连续的前景区间；
- 根据左右边界得到原始中线，依次执行梯度限幅和五点滑动平均；
- 固定透视宽度在 ROI 顶部为 `74 px`、底部为 `250 px`，中间按纵坐标线性插值；
- 正常巡线时，若选中 `track` 仅左边界碰到 ROI 左边缘，则使用右边界减去对应行的半个透视赛道宽度重建中心线；仅右边界碰到 ROI 右边缘时对称处理；两侧都碰边或都未碰边时使用左右边界中点；
- 中心线优先级固定为：已确认岔路中心线 > ROI 触边单边巡线 > 左右边界中点；
- 岔路确认后，分别由最左边界向右、最右边界向左内推半个透视宽度，得到左/右候选中线；
- 候选距离相对透视宽度连续 5 行达到 `0.15` 时区分双线，连续 5 行低于 `0.10` 时恢复普通中线；
- 首次自动选路前，用原始左右候选线的平均绝对二阶差分评估粗糙度；超过 `roughness_threshold_px`（默认 `3 px`）的候选会被舍弃。若两侧都粗糙，则舍弃粗糙度更大的一侧；差值不超过 `roughness_tie_margin_px`（默认 `0.1 px`）时回退到画面中心距离规则；
- 把 120 行赛道权重重采样到当前 ROI，计算加权横向误差并做三帧时域滤波；
- 通过左右边界外移、拐点、丢线统计和连续帧确认独立上报左/右岔路；无新方向时，以整幅画面中心线表示车辆朝向，选择平均横向距离短至少 `1 px` 的候选路线并锁定，距离差不超过 `1 px` 时继续普通巡线，离开岔路后释放；
- 行人目标点穿越停车、coin 目标和丢线历史恢复按新的优先级链工作。

## RK3588 PP-OCR 路牌识别

目标检测在同一帧识别到 `road_sign` 且原始框至少为 `96x48` 像素时，AI
子进程会按检测框中心扩展 10% 并送入 PP-OCRv4 Det/Rec RKNN 模型。只有整体
置信度达到 `0.60` 的非空文字才写入
`outputs/logs/ocr/ocr_events_YYYYMMDD_HHMMSS.jsonl`；成功后按
`ocr.cooldown_seconds` 配置全局 OCR 冷却时间，默认 20 秒。
OCR 模型真正开始推理时，主循环立即进入 `ROAD_SIGN_WAIT` 并持续向下位机发送
`speed_state=0x00`。停车覆盖 OCR 重试和后续千帆 API 请求；API 正常返回或产生
fallback 决策后恢复配置档位。`ocr.stop_timeout_sec` 控制 OCR/API 总等待上限，默认
20 秒；超时后取消 pending 请求、忽略迟到结果并保持当前分支。关闭路牌 API 时，
OCR 完成后同样保持当前分支。
每次 OCR 尝试都会在控制台输出 `[OCR]` 行，并在调试窗口用紫色框标出裁剪区域；
紫框默认显示 1 秒后清除，最新文字、置信度和耗时继续保留。低分候选只显示，
不写 JSONL，也不启动冷却。
模型、阈值、NPU 核和输出目录均在 `config.yaml` 的
`ocr` 中配置。板端只需要 RKNN-Toolkit-Lite2，不使用 ONNX 或
PaddlePaddle；额外 Python 依赖为 `shapely`、`pyclipper` 和 `six`。

## RK3588 路牌分析与岔路决策

高置信度 OCR 事件会由独立进程发送给千帆 `ernie-4.5-turbo-vl`，HTTP 请求不会阻塞
摄像头、目标检测或巡线。模型回答必须严格为 `left` 或 `right`。无有效结果时不强制
切换方向，而是保持车辆在本次岔路已经进入的分支；请求未完成时车辆可以继续接近岔路，
但到达岔路后会以 `ROAD_SIGN_WAIT` 模式停车。最终结果从主进程收到时开始按
`decision_ttl_sec` 计时，默认 20 秒，期间所有岔路遵循该方向，到期恢复 `current` 策略。

启动前在板端设置 API Key：

```bash
export QIANFAN_API_KEY='你的 API Key'
python3 main.py --no-gui
```

`config.yaml` 的 `road_sign_analyzer` 提供 API 地址、模型、API Key 环境
变量名、连接超时、读取超时、最大尝试次数、重试间隔、默认方向、结果有效时间和
API 日志目录。超时与结果有效时间必须大于 0，最大尝试次数至少为 1；缺少 API Key
或鉴权失败时不会把密钥打印到日志。`default_direction` 可设为 `left`、`right` 或
`current`，其中 `current` 表示不下发新的左右选择。

每次尝试和最终决策都会输出 `[ROAD_SIGN_ANALYZER]` 日志，并保存到
`outputs/logs/api/road_sign_analysis_events_YYYYMMDD_HHMMSS.jsonl`。记录包括 OCR 事件号、问题、
尝试次数、配置超时、HTTP 状态、耗时、原始回答、解析方向、错误、是否回退和过期
时间，不包含 API Key。有效期内识别到新路牌时会发起新请求，新结果覆盖旧方向并
重新计时。

## 输出日志

CSV 日志保存在 `outputs/logs/`，包含误差、置信度、目标速度/转向等，适合离线分析。
