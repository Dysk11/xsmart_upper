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

- 下位机已完成位置环、速度环等底层闭环（具体实现不属于本仓库）；
- 本项目**不实现**底层 PID、PWM 输出、电机闭环；
- 上位机只输出高层目标量，例如 `target_speed` 和 `steer_deg`；
- 上位机与下位机的全部交互集中在 `core/io/protocol.py`（协议打包）与
  `core/io/bridge.py`（发送通道）两个接口点，其余模块不感知下位机细节。

## RK3588 部署与环境配置

本项目面向 RK3588/RK3588S（示例板卡 Orange Pi 5）的 Linux aarch64 系统部署：
巡线与目标检测使用 RKNN C API 原生后端，OCR 使用 RKNNLite。以下步骤均在板端
终端执行，项目根路径示例为 `/home/orangepi/xsmart_upper`，其他板卡可整体替换。

### 环境要求

- 硬件：RK3588/RK3588S（如 Orange Pi 5），建议 4 GB 以上内存；
- 系统：Linux aarch64（Ubuntu/Debian），Python 3.10+；
- NPU 驱动与运行时：RKNPU2 Runtime（`librknnrt.so`），与 RKNN-Toolkit2 2.3.2
  同版本系，确认 `/dev/rknpu` 存在；
- 构建工具：`g++`（C++17）、`make`、`cmake`；可选 `librga`（缺失时巡线原生
  后端自动走 CPU 预处理路径）；
- Python 依赖：`requirements.txt` 列出的包，另需 `rknn-toolkit-lite2==2.3.2`
  （OCR 的 RKNNLite 执行器依赖，与板端 `librknnrt.so` 保持同版本系）。

### 获取代码

建议在板端直接 `git clone`；也可以从开发机 scp 整个工程（`main.py`、
`config.yaml`、`core/`、`native/`、`models/`、`tools/`、`utils/`）。从 Windows
上传的文件可能带 CRLF 行尾，需先修正：

```bash
sed -i 's/\r$//' config.yaml main.py README.md
find core native tools utils -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) \
  -exec sed -i 's/\r$//' {} +
git diff --check
```

`config.yaml` 中的模型、原生库、视频与输出目录路径均按项目根目录解析，
部署时保持目录结构完整即可。

### 板端环境配置

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-venv fonts-wqy-zenhei
# 如板卡镜像缺少 RGA，请按板卡 SDK 额外安装 librga 及其头文件（可选）

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install rknn-toolkit-lite2==2.3.2
```

确认 NPU 运行时可用：

```bash
ls -l /dev/rknpu
ldconfig -p | grep rknnrt   # 或 find /usr /usr/local -name 'librknnrt.so' 2>/dev/null
```

### 构建原生后端

先按板端实际安装位置设置 RKNN/RGA 头文件与库目录（以下为 Orange Pi 5
实测路径示例）：

```bash
export RKNN_INCLUDE_DIR=/home/orangepi/Downloads/xsmart_upper_native/native/include
export RKNN_LIBRARY_DIR=/usr/lib
export RGA_INCLUDE_DIR=/usr/include/rga
export RGA_LIBRARY_DIR=/usr/lib
```

依次构建三个后端，产物均生成在各自的 `build/` 目录：

```bash
# 1) 巡线 RKNN C API 二进制：native/lane_rknn_backend/build/lane_rknn_backend
bash native/lane_rknn_backend/build_board.sh
# 或使用 cmake：
# cmake -S native/lane_rknn_backend -B native/lane_rknn_backend/build -DCMAKE_BUILD_TYPE=Release
# cmake --build native/lane_rknn_backend/build -j

# 2) 目标检测 C API 共享库：native/object_rknn_backend/build/libxsmart_object_rknn.so
cmake -S native/object_rknn_backend -B native/object_rknn_backend/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/object_rknn_backend/build -j
# 板端无 cmake 时可直接编译：
# mkdir -p native/object_rknn_backend/build
# g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -fPIC -shared \
#   -Inative/object_rknn_backend/include -I"${RKNN_INCLUDE_DIR}" \
#   native/object_rknn_backend/src/object_backend.cpp \
#   -L"${RKNN_LIBRARY_DIR:-/lib}" -lrknnrt -pthread \
#   -o native/object_rknn_backend/build/libxsmart_object_rknn.so

# 3) 巡线几何行游程库：native/lane_geometry_backend/build/libxsmart_lane_geometry.so
cmake -S native/lane_geometry_backend -B native/lane_geometry_backend/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/lane_geometry_backend/build -j
```

构建完成后确认以下产物存在（与 `config.yaml` 中的路径一致）：

```text
native/lane_rknn_backend/build/lane_rknn_backend
native/object_rknn_backend/build/libxsmart_object_rknn.so
native/lane_geometry_backend/build/libxsmart_lane_geometry.so
```

### 模型与配置文件

`models/` 下三个目录必须完整复制，至少包含：

```text
models/lane/yolov5n_seg_track_480x640_int8_rk3588.rknn
models/object/rknn_7classes_0727.rknn
models/ocr/ppocrv4_det.rknn
models/ocr/ppocrv4_rec.rknn
models/ocr/ppocr_keys_v1.txt
```

板端启动前按现场情况检查/修改 `config.yaml`：

- `camera.mode`：`shared_memory`（默认，与 AR 系统共享 `shm_ar_video`）、
  `camera`（需确认 `device_id`）或 `video`（回放调试，需 `video_path`）；
- `bridge.type`：测试用 `mock`；实车用 `serial`，`bridge.serial` 下的端口与
  通信参数按现场实际通道配置；
- `visualizer.show_window`：无显示环境设为 `false` 或启动加 `--no-gui`；
  `visualizer.font_path` 板端可留空，程序会自动尝试 wqy-zenhei / Noto CJK
  等系统字体；
- `road_sign_analyzer`：需要千帆路牌分析时先
  `export QIANFAN_API_KEY='你的 API Key'` 再启动（详见“RK3588 路牌分析与
  岔路决策”）。

### 启动与验证

共享内存模式（与 AR 系统同机，默认配置）：

```bash
python3 main.py
```

摄像头实车模式：

```bash
python3 main.py --mode camera --bridge serial
```

视频回放调试（建议 `--bridge mock`，避免误发串口指令）：

```bash
python3 main.py --mode video --video /path/to/demo.mp4 --bridge mock
```

无显示环境：

```bash
python3 main.py --mode shared_memory --bridge serial --no-gui
```

正常启动时，生产配置（`runtime_backend: c_api`）终端应出现：

```text
[LANE_C_API] native dual-context backend ready
RKNN C API detector loaded on NPU2: ...
```

Lite2 模式（仅回退/基准场景）会输出 `RKNN lane segmenter loaded`。路牌识别时
出现 `[OCR]`、`[ROAD_SIGN_ANALYZER]` 日志；按 Ctrl+C 退出。

板端校验（需要测试视频，做固定帧与原生后端一致性检查）：

```bash
python3 tools/validate_capi_lane.py --video /path/to/test.mp4
python3 tools/validate_capi_object.py --video /path/to/test.mp4
python3 tools/validate_lane_geometry_backend.py
```

注意：`--bridge serial` 会真实向下位机发送指令，实车联调前请确认车辆断电
或架空，避免意外移动。

### 可选：systemd 开机自启

将以下内容保存为 `/etc/systemd/system/xsmart_upper.service`（路径按实际
部署替换）：

```ini
[Unit]
Description=X-SmartCar Upper Machine
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/orangepi/xsmart_upper
ExecStart=/home/orangepi/xsmart_upper/.venv/bin/python3 /home/orangepi/xsmart_upper/main.py --mode shared_memory --bridge serial --no-gui
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now xsmart_upper
sudo systemctl status xsmart_upper
```

启用自启后不要再手动启动同一配置，避免两个实例同时发送串口指令；
临时手动运行前先 `sudo systemctl stop xsmart_upper`。

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
- `line_loss_hold_sec`: 短时丢线保持上一有效控制量的秒数，超时后停车
- `lateral_error_slowdown_threshold_px`: 原始横向误差达到该绝对值时进入弯道档位，默认 `83`
- `curve_speed_state`: 弯道固定档位，`1`=低速、`2`=中速、`3`=高速
- `curve_speed_hold_sec`: 退出弯道阈值后继续保持弯道档位的秒数，默认 `1.0`；设为 `0` 时立即恢复

### 5. `bridge`

- `type`: `mock` 或 `serial`
- `drive_speed_state`: 正常行驶固定档位，`1`=低速、`2`=中速、`3`=高速
- `serial.port`: 串口名，例如 `/dev/ttyS4`
- `serial.baudrate`: 波特率
- `serial.timeout`: 串口超时

### 6. `visualizer`

- `show_window`: 是否显示实时画面窗口；画面左上角固定显示主循环 FPS
- `show_debug_window`: 是否独立显示详细 Debug 信息窗口，默认 `false`
- `debug_refresh_hz`: 独立 Debug 信息面板的重建频率，默认 `10` Hz；窗口仍逐帧显示缓存面板
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

摄像头模式需先确认 `config.yaml` 中的 `camera.device_id` 和串口参数正确，然后运行：

```bash
python3 main.py --mode camera
```

## 如何运行视频回放模式

方法一：直接使用主程序切换到视频模式

```bash
python3 main.py --mode video --video /path/to/demo.mp4 --bridge mock
```

## Camera-to-command latency benchmark

The reusable benchmark records command-aligned source, RKNN, lane geometry,
planning, protocol, and bridge timings. It writes raw CSV samples plus JSON and
Markdown summaries under `outputs/benchmarks`.

```bash
python tools/benchmark_latency.py \
  --mode shared_memory \
  --bridge mock \
  --warmup-sec 10 \
  --duration-sec 60

python tools/benchmark_latency.py \
  --mode video \
  --video outputs/video/record_20260708_135111.mp4 \
  --bridge mock \
  --warmup-sec 10 \
  --duration-sec 60
```

Serial benchmarking is intentionally blocked unless vehicle motion has been
physically disabled and `--serial-safety-confirmed` is supplied. The serial
endpoint is the return of host-side `write()` plus `flush()`; it does not
include lower-machine parsing or mechanical response.

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
  model_path: models/object/rknn_7classes_0727.rknn
  # 按 [width, height] 配置，与 RKNN 固定输入 640x480 一致
  input_size: [640, 480]
  class_names: [car, coin, Go, human, road_sign, speed_limit, Stop]
  runtime_backend: c_api
  c_api_library: native/object_rknn_backend/build/libxsmart_object_rknn.so
  pipeline_depth: 2
  # 目标检测强制使用 NPU2；其他值会导致启动失败
  core_mask: NPU_CORE_2
```

目标检测原生库在板端构建，构建命令与固定帧一致性校验见“RK3588 部署与
环境配置”。构建时使用与板端 `librknnrt.so` 匹配的 RKNN 2.3.2 头文件。

生产配置不回退到 Lite2；`--object-backend lite2` 仅供
`tools/benchmark_latency.py` 做同条件 A/B 基准。

For a paced 60 FPS shared-memory A/B run, use
`tools/replay_video_to_shm.py`; `tools/run_object_ab_benchmark.sh` performs the
required three 10-second-warmup/60-second-sample runs for both backends.

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
脱轨停车 > 行人穿越等待 > 路牌分析停车等待 > 不可行 car 避让停车 >
car 避让 > Go/Stop 路径 > 吃 coin > 普通巡线
```

行人框触发目标点穿越锁存后，运行模式为 `PEDESTRIAN_WAIT`，目标速度和
`speed_state` 均为停车档。`car` 原始检测框无法从指定侧安全绕过时使用
`CAR_AVOID_STOP`，同样输出停车指令。

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

Car 与 pedestrian 共用独立于巡线区域的安全触发范围。默认配置复制巡线 ROI；
旧配置未声明 `avoidance_roi` 时也会回退到当前 `lane_geometry.roi`：

```yaml
avoidance_roi:
  top_ratio: 0.585
  bottom_ratio: 1.0
  left_ratio: 0.05
  right_ratio: 0.95
```

调试画面用青色 `AVOID ROI` 矩形标出该范围，黄色矩形仍表示巡线 ROI。
`center_region` 由 avoidance ROI 相对横坐标配置并贯穿 avoidance ROI 全高；其左侧是 `left`
区域，右侧是 `right` 区域。只处理检测框中心位于 avoidance ROI 内的 `human`：
框面积达到减速入口后使用独立的行人接近档位，严格大于停车面积阈值时选择面积最大的
合格行人并立即停车。

```yaml
pedestrian_safety:
  enabled: true
  slowdown_min_box_area_px: 0
  stop_min_box_area_px: 1600
  approach_speed_state: 2
  rearm_cooldown_sec: 3.0
  target_stability_threshold_px: 20
  target_stability_confirm_frames: 2
  crossing_confirm_frames: 3
  moving_away_min_delta_px: 3
  moving_away_confirm_frames: 2
  center_region:
    left_ratio: 0.30
    right_ratio: 0.70
```

检测到 ROI 内行人后先切换到 `approach_speed_state`；框面积严格大于
`stop_min_box_area_px` 后立即停车，但先不冻结目标线。每个巡线帧比较普通目标点 x 与上一帧
的跳变；连续 `target_stability_confirm_frames` 次严格小于
`target_stability_threshold_px` 后，才冻结当前目标点 x 和它所属的
`left/center/right` 区域。稳定期间继续关联行人但不判断穿越，锁线后的第一份
新行人检测只建立穿越基线，后续检测才参与释放判断，因此锁线前的移动不会被误算
为穿越。调试画面在稳定期间不绘制冻结目标线，并在原因文本中显示稳定计数。

锁线后仍逐帧比较当前普通目标点 x 与冻结线；偏差达到或超过
`target_stability_threshold_px` 时立即作废旧线、暂停穿越判断并保持停车。
触发重锁的当前帧不会锁定新线，下一巡线帧直接冻结当时的有效目标点，不再执行
连续稳定计数；若目标点不是有限值则继续等待。新线锁定后的第一份新行人检测重新
建立穿越基线，旧线失效前后的行人移动不会触发释放。

后续结果只在中心位于 avoidance ROI 内的 human 框中选择距触发行人上一中心最近者，
不设置关联距离上限；漏检或关联行人中心移出 avoidance ROI 时无限保持停车，并清除
穿越基线和远离计数。重新进入 ROI 的第一份新 AI 结果只重建基线，ROI 外的运动不能
用于放行。center 目标允许任意方向跨越，left 目标只接受右到左，right 目标只接受
左到右。首次严格跨线计为一次，行人需要连续 `crossing_confirm_frames` 份新 AI 结果
保持在首次到达的同一侧才会放行；压线、回到原侧、漏检、移出 avoidance ROI 或目标线
重锁都会清零跨线确认，缓存 AI 结果不会推进计数。

锁线并建立行人基线后，还会用连续的新 AI 结果判断行人是否正在远离冻结线。相邻
结果中，行人中心到冻结线的距离至少增加 `moving_away_min_delta_px`，才累计一次
有效远离；连续达到 `moving_away_confirm_frames` 次后放行。center 目标不启用远离
放行，无论行人向哪一侧远离都继续停车，仅在严格穿越冻结线时恢复。left 目标只允许
行人在线左侧继续向左，right 目标只允许行人在线右侧继续向右。停滞、靠近、增量不足、
侧别不符、漏检、移出 avoidance ROI 或目标线重锁都会清零累计；缓存 AI 结果不推进
累计。跨线确认期间不累计远离次数。完成跨线确认或确认远离后立即恢复并进入 3 秒冷却，
冷却期间 human 不再触发停车。
行人逻辑只处理 `human`，不会把 `car` 当作行人停车目标。

## 7. car 原始检测框避让

`car` 的原始检测框只要与 avoidance ROI 接触或重叠就触发避让，不再放大检测框。
一个避让过程开始时，先选择框底边最大（相同时置信度最高）的 car 作为主目标，
在其垂直中心对应的 ROI 行比较原巡线中心线与 car 中心：中心线在左侧或同 x
时锁定 track 左边界，中心线在右侧时锁定 track 右边界。该方向保持到退出平滑
完成，避免检测框或中心线抖动导致左右切换。

```yaml
car_avoidance:
  enabled: true
  entry_duration_s: 1.0
  edge_slow_margin_px: 20
  route_offset_px: 0
  release_duration_s: 1.0
  speed_state: 1
```

`route_offset_px` 为非负像素值：锁定左边界时向左偏移，锁定右边界时向右偏移；
偏移结果会裁剪到巡线 ROI 的水平范围，默认 `0` 保持原路线不变。规划器在相同 ROI
行上对齐正常中心线与偏移后的锁定侧 track 边界，进入时用 1 秒时间 smoothstep
从正常路线过渡到避障路线；新 AI 检测结果确认 avoidance ROI 内已无 car 后，模式切换为
`CAR_AVOID_RECOVERY`，再用 1 秒 smoothstep 回到当前正常中心线。
锁定侧边界每帧只做一次稠密化和短缺口线性插值，所有目标行复用该结果；进入和
恢复阶段对齐相同 y 行后直接逐点混合，不重复排序边界。该方法不进行栅格搜索或
曲线拟合。

最终折线路线只会与 car 框和巡线 ROI 的实际交集做线段相交检查。位于 avoidance ROI
内但尚未进入巡线 ROI 的 car 会提前触发边界避让，不会被投影到巡线 ROI 边缘制造
假碰撞。过渡路线仍碰框、指定侧
边界缺失或无法连续插值时进入 `CAR_AVOID_STOP` 并发送零速度；路线安全后自动恢复
避让。安全路线进入 ROI 左右边缘 20 px 时模式为 `CAR_AVOID_EDGE`，速度限制为
`planner.min_speed`。避让目标点仍固定在 ROI `y=80`。调试画面中的橙/红框为原始
检测框，紫色折线为锁定的 track 边界，黄色折线为最终平滑路线，并显示锁定侧、
切换阶段和进度。

终端“巡线耗时”中的 `planning_ms` 单独统计目标选择、car 避让和高层控制耗时，
用于比较普通巡线、进入避让、稳态避让及恢复阶段。

## 8. 主流程顺序

1. 读取图像 -> 2. ROI 裁剪/预处理 -> 3. 蓝色航道巡线 -> 4. RKNN 目标识别 -> 5. 行人安全判断 -> 6. car 原始框避让 -> 7. 特殊目标/普通巡线决策 -> 8. 生成协议帧发送。

## 9. 如何确认功能正常

1. 终端出现 `RKNN detector loaded`；
2. 调试画面中出现 `coin/Go/Stop/car/human` 目标框；
3. 画面上 `G` 为 coin 目标点，`P` 为 Go/Stop 框中心，两条黄线分隔 `LEFT/CENTER/RIGHT`，停车时红线为冻结目标 x；
4. 行人停车时模式显示 `PEDESTRIAN_WAIT`，car 避让显示 `CAR_AVOID` /
   `CAR_AVOID_EDGE` / `CAR_AVOID_RECOVERY` / `CAR_AVOID_STOP`，Go/Stop 路径显示 `PATH_TARGET`，
   吃金币时显示 `GOLD`。

---

## 各模块与下位机的接口

上位机与下位机之间只有两个接口点，其余模块一律不直接访问串口或构造协议帧：

- `core/io/protocol.py`：协议层，唯一负责把高层控制量打包成下位机可解析的
  帧；调整帧格式时只修改这里。
- `core/io/bridge.py`：桥接层，唯一负责把协议帧发送出去；`type: mock` 为
  模拟输出，`type: serial` 走串口通道，更换物理通道时只修改这里。

各模块对下位机的接口：

| 模块 | 对下位机的接口 | 说明 |
| :--- | :--- | :--- |
| `core/io/protocol.py` | 高层控制量 → 协议帧 | 唯一协议打包入口，内容由规划结果与 `bridge` 配置决定 |
| `core/io/bridge.py` | 协议帧 → 串口/模拟输出 | 唯一发送通道，不参与决策 |
| `core/planning/high_level.py` | `ControlCommand`（`target_speed`、`steer_deg`、`speed_state_override`、`force_mode` 等） | 决策层只产出高层目标量 |
| `core/planning/*` 与 `core/object/pedestrian_safety.py`（目标选择、避让、行人安全、金币、路径标记、路牌分析） | `ModuleHints` 模式/限速提示 | 只影响 `ControlCommand`，不直接访问串口 |
| `core/lane/*`、`core/object/*`、`core/ocr/*` | 无直接输出 | 感知结果只交给规划层，不感知下位机 |

速度档位语义：`bridge.drive_speed_state` 为正常行驶档位（只允许 `1`/`2`/`3`）；
弯道、car 避让、行人/路牌减速等请求与正常档位仲裁时取最低档且不累计降档；
任何停车请求都覆盖为停车档（`speed_state=0`）。具体字节编码、串口参数与下位机
解析规则不在本仓库文档范围，以下位机侧协议文档为准。

## RKNN 航道分割部署

当前主巡线链路在 RK3588 上使用单类 `track` 的 YOLOv5n-seg INT8 模型生成 mask。应用层不再执行缩放、增强或颜色预处理；模型内部仍保留必需的 RGB/letterbox 输入转换。mask 在 ROI 中逐行提取左右边界、滤波中线并识别左右岔路。

- 模型：`models/lane/yolov5n_seg_track_480x640_int8_rk3588.rknn`
- SHA-256：`0ffd0f431505fa362b4d1f4a94ae69321b2c77a4081c6a919f758f28712b1dce`
- 输入：RGB uint8 NHWC `[1, 480, 640, 3]`；模型内部完成 `/255` 归一化。
- 输出：三组 box/class、三组 32 维 mask coefficient 和一个 `[1, 32, 120, 160]` prototype。
- 类别：`0 = track`；默认置信度阈值 `0.25`、NMS IoU `0.45`、mask 阈值 `0.5`。
- 转换环境：RKNN-Toolkit2 `2.3.2`，目标平台 `rk3588`，W8A8 per-channel INT8。

板端环境安装、原生后端构建与启动步骤见“RK3588 部署与环境配置”。生产配置
（`runtime_backend: c_api`）启动时终端输出 `[LANE_C_API] native dual-context
backend ready`；Lite2 模式输出 `RKNN lane segmenter loaded`。调试窗口在 ROI
内半透明显示 `track` mask，并显示 `track: ok conf=...`。加载或推理失败不会
静默回退到 HSV，而会输出一次明确告警并按丢线处理。默认不会保存视频或截图。

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

目标检测在同一帧识别到 `road_sign` 且原始框满足配置的置信度、最小宽度和最小
高度时，AI 子进程立即通知主循环进入 `ROAD_SIGN_WAIT`，并按检测框中心扩展 10%。
系统选择置信度最高、面积最大的候选；相邻原始框面积变化不超过
`bbox_area_stability_threshold_ratio: 0.10`，连续满足
`bbox_area_stability_confirm_deltas: 2` 次且岔路已经确认后，才把裁剪送入
PP-OCRv4 Det/Rec RKNN 模型。候选缺失或面积变化超限会重新累计稳定次数，但停车
会话继续保持到流程完成或总超时。只有整体
置信度达到 `0.60` 的非空文字才写入
`outputs/logs/ocr/ocr_events_YYYYMMDD_HHMMSS.jsonl`；成功后按
`ocr.cooldown_seconds` 配置全局 OCR 冷却时间，当前默认 10 秒。
主循环在整个 `ROAD_SIGN_WAIT` 期间持续向下位机输出停车指令（`speed_state`
为停车档）。
停车覆盖面积稳定等待、岔路确认和后续千帆 API 请求；API 正常返回或产生
fallback 决策后恢复配置档位。`ocr.stop_timeout_sec` 控制 OCR/API 总等待上限，默认
20 秒；超时后取消 pending 请求、忽略迟到结果并保持当前分支。关闭路牌 API 时，
OCR 完成后同样保持当前分支。
每次 OCR 尝试都会在控制台输出 `[OCR]` 行，并在调试窗口用紫色框标出裁剪区域；
紫框默认显示 1 秒后清除，最新文字、置信度和耗时继续保留。空文本、低于
`ocr.accept_score` 的文字或 OCR 推理异常会立即解除本次 OCR 停车并进入全局冷却，
不写 JSONL、不请求路牌 API；同一路牌必须先在新检测结果中消失，之后才能重新触发。
`ocr.retry_interval_sec` 只用于有效 OCR 事件写日志失败后的重试。
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
