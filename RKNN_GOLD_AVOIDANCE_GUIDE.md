# RKNN 目标识别、避障、吃 Gold 功能说明

本文档说明当前工程里和 `rknn_lt.rknn` 模型、测试视频、`Car/Human` 避障、`Gold` 目标追踪相关的实现细节。

## 1. 当前已接入的内容

当前工程已经完成以下接入：

- 模型文件：`rknn_lt.rknn`
- 测试视频：`cbf977c5bd5978922b972f4f0285c0bd.mp4`
- RKNN 推理模块：`core/rknn_object_detector.py`
- Gold 目标规划模块：`core/gold_target_planner.py`
- 避障判断模块：`core/blocking_analyzer.py`
- 避障目标规划模块：`core/avoidance_target_planner.py`
- 主流程入口：`main.py`
- 主要配置文件：`config/config.yaml`

默认配置已经把图像源设置成视频模式：

```yaml
camera:
  mode: video
  video_path: cbf977c5bd5978922b972f4f0285c0bd.mp4
```

模型配置如下：

```yaml
rknn_object_detector:
  enable: true
  model_path: rknn_lt.rknn
  input_size: [640, 640]
  input_layout: nhwc
  input_color: rgb
  input_dtype: uint8
  score_threshold: 0.20
  nms_threshold: 0.45
  max_detections: 30
  class_names: [Gold, Car, Human]
```

## 2. 类别顺序是什么意思

模型实际输出的是类别编号，不会直接输出中文或英文名称。比如模型输出：

```text
0
1
2
```

代码必须知道这些编号分别代表什么。当前配置：

```yaml
class_names: [Gold, Car, Human]
```

表示：

```text
0 = Gold
1 = Car
2 = Human
```

如果训练模型时的类别顺序不是这个，例如：

```text
0 = Car
1 = Human
2 = Gold
```

就必须改成：

```yaml
class_names: [Car, Human, Gold]
```

否则会出现危险情况：模型明明识别到 `Car`，但程序误以为是 `Gold`，车可能会朝障碍物开。

类别顺序一般可以在训练模型用的数据集配置里找到，常见文件名包括：

- `data.yaml`
- `dataset.yaml`
- `classes.txt`

常见格式：

```yaml
names:
  0: Gold
  1: Car
  2: Human
```

## 3. 控制优先级

当前主逻辑优先级是：

```text
巡线基础控制一直运行
        |
        v
识别 Car/Human 是否进入航道危险走廊
        |
        +-- 是：避障优先，忽略 Gold 控制
        |
        +-- 否：如果识别到 Gold，朝 Gold 走去吃
        |
        +-- 否：正常巡线
```

也就是说：

```text
Car/Human 避障 > 吃 Gold > 普通巡线
```

注意：这里的“避障和巡线同时进行”不是两套控制指令同时发给下位机，而是以巡线中心线为基础。如果 `Car/Human` 阻挡了当前航道，避障模块会在巡线中心线上加一个平滑偏移，生成新的目标点。最终仍然只发送一组 `lateral_error_px` 和 `steer_deg` 给下位机。

## 4. RKNN 模型加载与识别

模型加载代码在：

```text
core/rknn_object_detector.py
```

运行到香橙派/RK3588 环境时，会尝试：

```python
from rknnlite.api import RKNNLite
```

然后加载：

```text
rknn_lt.rknn
```

成功时终端会打印类似：

```text
RKNN detector loaded: .../rknn_lt.rknn
```

如果本地电脑没有 `rknnlite`，检测模块会提示缺少运行库，并跳过目标检测。这个是正常的，因为 RKNN 模型主要是在香橙派/RK3588 上跑。

## 5. Gold 目标逻辑

Gold 逻辑在：

```text
core/gold_target_planner.py
```

配置在：

```yaml
gold_target:
  enabled: true
  class_names: [Gold]
  min_confidence: 0.20
  hold_frames: 4
  approach_speed_limit: 0.85
  close_speed_limit: 0.45
  close_y_ratio: 0.82
  max_above_roi_ratio: 0.8
  aim_at: bottom_center
```

含义：

- `class_names: [Gold]`：只有识别类别名为 `Gold` 的目标才触发吃金币逻辑。
- `min_confidence: 0.20`：低于这个置信度的 Gold 不采用。
- `hold_frames: 4`：Gold 短暂丢失时，会继续保持几帧目标，减少检测闪烁。
- `approach_speed_limit: 0.85`：朝 Gold 走时限制速度。
- `close_speed_limit: 0.45`：Gold 很近时进一步降速。
- `close_y_ratio: 0.82`：Gold 框底部超过 ROI 高度的 82% 时认为比较近。
- `aim_at: bottom_center`：目标点取 Gold 框的底部中心，更像是朝物体落地点走。

当 Gold 生效时，主流程会进入：

```text
mode = GOLD
```

但是如果同一帧里 `Car/Human` 需要避障，`GOLD` 模式不会生效。

## 6. Car/Human 避障逻辑

避障分两步：

1. `core/blocking_analyzer.py` 判断识别框是否挡住当前航道。
2. `core/avoidance_target_planner.py` 根据阻挡位置生成偏移后的目标路线。

避障只对非 Gold 目标生效。当前逻辑会把 `Gold` 从避障目标列表里排除，避免把 Gold 当障碍物绕开。

当前模型类别配置里：

```yaml
class_names: [Gold, Car, Human]
```

因此 `Car` 和 `Human` 会进入避障分析，`Gold` 会进入吃 Gold 分析。

避障判断关键配置：

```yaml
blocking_analyzer:
  enabled: true
  corridor_half_width_px: 90
  confidence_threshold: 0.45
  min_box_area: 600
  near_y_ratio: 0.42
  too_close_y_ratio: 0.82
  blocking_score_threshold: 0.25
  side_deadband_px: 25
```

含义：

- `corridor_half_width_px`：以航道中心线为中心，左右多少像素算危险走廊。
- `confidence_threshold`：目标置信度低于此值不参与避障。
- `min_box_area`：目标框太小不参与避障。
- `near_y_ratio`：目标太远时先不避障。
- `too_close_y_ratio`：目标太近时进入保守状态。
- `blocking_score_threshold`：目标框和危险走廊重叠比例超过此值才认为需要避障。
- `side_deadband_px`：判断障碍物在中心线左侧还是右侧的死区。

## 7. 主流程关键位置

主循环在：

```text
main.py
```

核心顺序：

1. 读取视频帧或摄像头帧。
2. 做 ROI 裁剪和图像预处理。
3. 做蓝色航道巡线检测。
4. 做 RKNN 目标识别。
5. 判断 `Car/Human` 是否需要避障。
6. 如果需要避障，优先使用避障目标。
7. 如果不需要避障且识别到 `Gold`，使用 Gold 目标。
8. 否则正常巡线。
9. 打包并通过串口或 mock bridge 发送控制量。

真正的优先级代码在 `main.py` 的 `_build_planning_state()` 中：

```text
if blocking_result.need_avoid or blocking_result.too_close:
    使用避障结果
elif gold_result.active:
    使用 Gold 目标
else:
    正常巡线
```

## 8. 串口发送内容

串口协议在：

```text
core/protocol.py
```

当前协议格式：

```text
0xAA 0x55 error_high error_low angle_high angle_low
```

其中：

- `error` 来自 `lateral_error_px`
- `angle` 来自 `steer_deg`
- 两者都会转成 16 位有符号整数发送

示例：

```text
AA 55 00 10 01 2C
```

表示：

```text
error = 16
angle = 300
```

## 9. 如何运行

在香橙派上进入项目目录：

```bash
cd xsmart_upper
```

安装基础依赖：

```bash
pip3 install -r requirements.txt
```

还需要安装 Rockchip 的 RKNN Lite 运行库，也就是能导入：

```python
from rknnlite.api import RKNNLite
```

运行主程序：

```bash
python3 main.py
```

如果要强制使用 mock，不发真实串口：

```bash
python3 main.py --bridge mock
```

如果要连接真实串口：

```bash
python3 main.py --bridge serial
```

串口参数在：

```yaml
bridge:
  serial:
    port: /dev/ttyUSB0
    baudrate: 115200
```

## 10. 如何确认功能正常

启动后重点看终端和调试画面：

1. 终端出现：

```text
RKNN detector loaded: .../rknn_lt.rknn
```

说明模型成功加载。

2. 调试画面中出现目标框：

```text
Gold 0.xx
Car 0.xx
Human 0.xx
```

说明模型有识别输出。

3. 画面上：

- `G` 表示 Gold 目标点。
- `N` 表示普通巡线目标点。
- `A` 表示最终控制目标点。
- 目标框会直接画在原始画面上。

4. 当出现 `Car/Human` 且挡住航道时，调试信息里应该优先显示避障相关模式，例如：

```text
avoid_left
avoid_right
too_close
```

5. 当前方没有 `Car/Human` 阻挡且识别到 `Gold` 时，模式会变成：

```text
GOLD
```

## 11. 常见问题

### 识别框有，但类别错了

检查：

```yaml
rknn_object_detector:
  class_names: [Gold, Car, Human]
```

把顺序改成训练时的数据集类别顺序。

### 没有识别框

检查：

- `rknnlite` 是否安装成功。
- `rknn_lt.rknn` 是否在项目目录。
- 香橙派是否是 RK3588/RK3588S 对应环境。
- `score_threshold` 是否太高，可以临时降到 `0.10` 测试。

### 识别到 Gold 但不去吃

可能原因：

- 同时有 `Car/Human` 触发避障，避障优先级更高。
- Gold 置信度低于 `gold_target.min_confidence`。
- 类别顺序配置错误，Gold 被识别成其他类别。

### 识别到 Car/Human 但不避障

可能原因：

- 目标没有进入航道危险走廊。
- 置信度低于 `blocking_analyzer.confidence_threshold`。
- 目标框面积低于 `blocking_analyzer.min_box_area`。
- 类别顺序配置错误，Car/Human 被识别成 Gold 或其他类别。

## 12. 现场调参建议

如果误识别太多：

```yaml
rknn_object_detector:
  score_threshold: 0.30
```

如果识别框太少：

```yaml
rknn_object_detector:
  score_threshold: 0.10
```

如果避障太敏感：

```yaml
blocking_analyzer:
  blocking_score_threshold: 0.35
  corridor_half_width_px: 70
```

如果避障太迟：

```yaml
blocking_analyzer:
  near_y_ratio: 0.35
  corridor_half_width_px: 110
```

如果朝 Gold 走太快：

```yaml
gold_target:
  approach_speed_limit: 0.60
  close_speed_limit: 0.30
```

## 13. 当前最终结论

当前工程的行为是：

```text
正常情况下：巡线
识别到 Car/Human 并挡住航道：避障，优先级最高
没有障碍且识别到 Gold：朝 Gold 走去吃
Gold 消失或吃完：回到巡线
```

这满足：

```text
避障优先级大于吃 Gold
```

