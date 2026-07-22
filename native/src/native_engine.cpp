#include "native_engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <numeric>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

namespace xsmart {

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point from, Clock::time_point to) {
  return std::chrono::duration<double, std::milli>(to - from).count();
}

double monotonic_seconds() {
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

float sigmoid(float value) { return 1.0F / (1.0F + std::exp(-value)); }

float probability(float value) {
  return (value >= 0.0F && value <= 1.0F) ? value : sigmoid(value);
}

float box_iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  const float left = std::max(a[0], b[0]);
  const float top = std::max(a[1], b[1]);
  const float right = std::min(a[2], b[2]);
  const float bottom = std::min(a[3], b[3]);
  const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
  const float area_a = std::max(0.0F, a[2] - a[0]) * std::max(0.0F, a[3] - a[1]);
  const float area_b = std::max(0.0F, b[2] - b[0]) * std::max(0.0F, b[3] - b[1]);
  return intersection / std::max(1.0e-6F, area_a + area_b - intersection);
}

std::array<cv::Point2f, 4> order_points(const cv::Point2f points[4]) {
  std::array<cv::Point2f, 4> ordered{};
  std::vector<cv::Point2f> sorted(points, points + 4);
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.x == b.x ? a.y < b.y : a.x < b.x;
  });
  const auto left_top = sorted[0].y < sorted[1].y ? sorted[0] : sorted[1];
  const auto left_bottom = sorted[0].y < sorted[1].y ? sorted[1] : sorted[0];
  const auto right_top = sorted[2].y < sorted[3].y ? sorted[2] : sorted[3];
  const auto right_bottom = sorted[2].y < sorted[3].y ? sorted[3] : sorted[2];
  ordered[0] = left_top;
  ordered[1] = right_top;
  ordered[2] = right_bottom;
  ordered[3] = left_bottom;
  return ordered;
}

cv::Mat perspective_crop(const cv::Mat& image, const std::array<cv::Point2f, 4>& box) {
  const int width = std::max(
      1, static_cast<int>(std::round(std::max(cv::norm(box[0] - box[1]),
                                              cv::norm(box[2] - box[3])))));
  const int height = std::max(
      1, static_cast<int>(std::round(std::max(cv::norm(box[0] - box[3]),
                                              cv::norm(box[1] - box[2])))));
  const std::array<cv::Point2f, 4> target = {
      cv::Point2f(0.0F, 0.0F), cv::Point2f(static_cast<float>(width), 0.0F),
      cv::Point2f(static_cast<float>(width), static_cast<float>(height)),
      cv::Point2f(0.0F, static_cast<float>(height))};
  cv::Mat transform = cv::getPerspectiveTransform(box.data(), target.data());
  cv::Mat crop;
  cv::warpPerspective(image, crop, transform, cv::Size(width, height), cv::INTER_CUBIC,
                      cv::BORDER_REPLICATE);
  if (crop.rows > 0 && crop.cols > 0 && static_cast<double>(crop.rows) / crop.cols >= 1.5) {
    cv::rotate(crop, crop, cv::ROTATE_90_COUNTERCLOCKWISE);
  }
  return crop;
}

std::array<float, 8> flatten_box(const std::array<cv::Point2f, 4>& box) {
  std::array<float, 8> result{};
  for (std::size_t i = 0; i < box.size(); ++i) {
    result[i * 2] = box[i].x;
    result[i * 2 + 1] = box[i].y;
  }
  return result;
}

struct SharedHeader {
  uint64_t frame_id;
  uint32_t width;
  uint32_t height;
};

static_assert(sizeof(SharedHeader) == 16, "shared-memory header must match @QII");

struct LaneCandidate {
  std::array<float, 4> box{};
  std::array<float, 32> coeff{};
  float score = 0.0F;
};

struct ObjectCandidate {
  std::array<float, 4> box{};
  float score = 0.0F;
  int class_id = 0;
};

}  // namespace

NativeEngine::NativeEngine(EngineConfig config) : config_(std::move(config)) {
  config_.ring_buffer_size = std::max<std::size_t>(3, config_.ring_buffer_size);
}

NativeEngine::~NativeEngine() { close(); }

void NativeEngine::open() {
  if (opened_) {
    return;
  }
  try {
    open_capture();
    lane_model_.open(config_.lane.model_path, RKNN_NPU_CORE_0, RKNN_TENSOR_UINT8,
                     RKNN_TENSOR_NHWC);
    object_model_.open(config_.object.model_path, RKNN_NPU_CORE_1, RKNN_TENSOR_UINT8,
                       RKNN_TENSOR_NHWC);
    ocr_det_model_.open(config_.ocr.det_model_path, RKNN_NPU_CORE_2, RKNN_TENSOR_UINT8,
                        RKNN_TENSOR_NHWC);
    ocr_rec_model_.open(config_.ocr.rec_model_path, RKNN_NPU_CORE_2, RKNN_TENSOR_FLOAT32,
                        RKNN_TENSOR_NHWC);
    load_characters();
    ocr_stop_ = false;
    ocr_thread_ = std::thread(&NativeEngine::ocr_worker_loop, this);
    opened_ = true;
  } catch (...) {
    close();
    throw;
  }
}

void NativeEngine::open_capture() {
  close_capture();
  if (config_.camera.mode == "shared_memory") {
    std::string name = config_.camera.shared_memory_name;
    if (name.empty() || name.front() != '/') {
      name.insert(name.begin(), '/');
    }
    for (int attempt = 0; attempt < std::max(1, config_.camera.reconnect_attempts); ++attempt) {
      shm_fd_ = shm_open(name.c_str(), O_RDONLY, 0);
      if (shm_fd_ >= 0) {
        break;
      }
      std::this_thread::sleep_for(
          std::chrono::duration<double>(config_.camera.reconnect_interval_sec));
    }
    if (shm_fd_ < 0) {
      throw std::runtime_error("failed to open shared memory: " + name);
    }
    struct stat info {};
    if (fstat(shm_fd_, &info) != 0 || info.st_size < static_cast<off_t>(sizeof(SharedHeader))) {
      throw std::runtime_error("invalid shared-memory size");
    }
    shm_size_ = static_cast<std::size_t>(info.st_size);
    shm_addr_ = mmap(nullptr, shm_size_, PROT_READ, MAP_SHARED, shm_fd_, 0);
    if (shm_addr_ == MAP_FAILED) {
      shm_addr_ = nullptr;
      throw std::runtime_error("mmap(shared_memory) failed");
    }
    return;
  }

  const bool opened = config_.camera.mode == "video"
                          ? capture_.open(config_.camera.video_path)
                          : capture_.open(config_.camera.device_id, cv::CAP_V4L2);
  if (!opened) {
    throw std::runtime_error("failed to open native image source");
  }
  if (config_.camera.mode == "camera") {
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, config_.camera.width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.camera.height);
    capture_.set(cv::CAP_PROP_FPS, config_.camera.fps);
    capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);
  }
}

void NativeEngine::close_capture() noexcept {
  capture_.release();
  if (shm_addr_ != nullptr) {
    munmap(shm_addr_, shm_size_);
  }
  shm_addr_ = nullptr;
  shm_size_ = 0;
  if (shm_fd_ >= 0) {
    ::close(shm_fd_);
  }
  shm_fd_ = -1;
  shared_last_frame_id_ = 0;
}

bool NativeEngine::read_shared_memory(cv::Mat& rgb, uint64_t& source_frame_id) {
  if (shm_addr_ == nullptr) {
    return false;
  }
  for (int poll = 0; poll < 20; ++poll) {
    SharedHeader before{};
    std::memcpy(&before, shm_addr_, sizeof(before));
    std::atomic_thread_fence(std::memory_order_acquire);
    if (before.frame_id == 0 || before.frame_id == shared_last_frame_id_ || before.width == 0 ||
        before.height == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    const std::size_t bytes = static_cast<std::size_t>(before.width) * before.height * 3;
    if (sizeof(SharedHeader) + bytes > shm_size_) {
      throw std::runtime_error("shared-memory frame exceeds mapped size");
    }
    rgb.create(static_cast<int>(before.height), static_cast<int>(before.width), CV_8UC3);
    std::memcpy(rgb.data, static_cast<const uint8_t*>(shm_addr_) + sizeof(SharedHeader), bytes);
    std::atomic_thread_fence(std::memory_order_acquire);
    SharedHeader after{};
    std::memcpy(&after, shm_addr_, sizeof(after));
    if (std::memcmp(&before, &after, sizeof(before)) != 0) {
      continue;
    }
    shared_last_frame_id_ = before.frame_id;
    source_frame_id = before.frame_id;
    return true;
  }
  return false;
}

bool NativeEngine::read_rgb(cv::Mat& rgb, uint64_t& source_frame_id) {
  if (config_.camera.mode == "shared_memory") {
    return read_shared_memory(rgb, source_frame_id);
  }
  cv::Mat bgr;
  if (!capture_.read(bgr) || bgr.empty()) {
    if (config_.camera.mode == "video" && config_.camera.loop_video) {
      capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
      if (!capture_.read(bgr) || bgr.empty()) {
        return false;
      }
    } else {
      return false;
    }
  }
  if (config_.camera.mirror) {
    cv::flip(bgr, bgr, 1);
  }
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  source_frame_id = ++frame_counter_;
  return true;
}

cv::Mat NativeEngine::letterbox_rgb(const cv::Mat& rgb, int width, int height,
                                    Letterbox& info) {
  info.source_width = rgb.cols;
  info.source_height = rgb.rows;
  info.scale = std::min(static_cast<float>(width) / rgb.cols,
                        static_cast<float>(height) / rgb.rows);
  info.resized_width = static_cast<int>(std::round(rgb.cols * info.scale));
  info.resized_height = static_cast<int>(std::round(rgb.rows * info.scale));
  info.pad_x = (width - info.resized_width) / 2;
  info.pad_y = (height - info.resized_height) / 2;
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(info.resized_width, info.resized_height), 0.0, 0.0,
             cv::INTER_LINEAR);
  cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(info.pad_x, info.pad_y, info.resized_width,
                                info.resized_height)));
  return canvas;
}

LaneResult NativeEngine::run_lane(const cv::Mat& rgb) {
  const auto started = Clock::now();
  Letterbox info;
  cv::Mat input = letterbox_rgb(rgb, config_.lane.input_width, config_.lane.input_height, info);
  const auto preprocessed = Clock::now();
  const auto outputs = lane_model_.run(input.data, input.total() * input.elemSize());
  const auto inferred = Clock::now();

  static const int grid_heights[3] = {60, 30, 15};
  static const int grid_widths[3] = {80, 40, 20};
  static const float strides[3] = {8.0F, 16.0F, 32.0F};
  static const float anchors[3][3][2] = {
      {{10.0F, 13.0F}, {16.0F, 30.0F}, {33.0F, 23.0F}},
      {{30.0F, 61.0F}, {62.0F, 45.0F}, {59.0F, 119.0F}},
      {{116.0F, 90.0F}, {156.0F, 198.0F}, {373.0F, 326.0F}}};
  if (outputs.size() != 7 || outputs[6].size != 32U * 120U * 160U) {
    throw std::runtime_error("unexpected lane output tensor count or prototype size");
  }

  std::vector<LaneCandidate> candidates;
  for (int level = 0; level < 3; ++level) {
    const int height = grid_heights[level];
    const int width = grid_widths[level];
    const int plane = height * width;
    const auto& box_cls = outputs[level * 2];
    const auto& coeff = outputs[level * 2 + 1];
    if (box_cls.size != static_cast<std::size_t>(18 * plane) ||
        coeff.size != static_cast<std::size_t>(96 * plane)) {
      throw std::runtime_error("unexpected lane split-head shape");
    }
    for (int anchor = 0; anchor < 3; ++anchor) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const int offset = y * width + x;
          const int channel = anchor * 6;
          const float objectness = box_cls.data[(channel + 4) * plane + offset];
          const float class_score = box_cls.data[(channel + 5) * plane + offset];
          const float score = objectness * class_score;
          if (score < config_.lane.score_threshold) {
            continue;
          }
          const float cx = (box_cls.data[(channel + 0) * plane + offset] * 2.0F +
                            static_cast<float>(x) - 0.5F) *
                           strides[level];
          const float cy = (box_cls.data[(channel + 1) * plane + offset] * 2.0F +
                            static_cast<float>(y) - 0.5F) *
                           strides[level];
          const float w = std::pow(box_cls.data[(channel + 2) * plane + offset] * 2.0F,
                                   2.0F) *
                          anchors[level][anchor][0];
          const float h = std::pow(box_cls.data[(channel + 3) * plane + offset] * 2.0F,
                                   2.0F) *
                          anchors[level][anchor][1];
          LaneCandidate candidate;
          candidate.box = {cx - w * 0.5F, cy - h * 0.5F, cx + w * 0.5F, cy + h * 0.5F};
          candidate.score = score;
          for (int index = 0; index < 32; ++index) {
            candidate.coeff[index] = coeff.data[(anchor * 32 + index) * plane + offset];
          }
          candidates.push_back(candidate);
        }
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.score > b.score; });
  std::vector<LaneCandidate> selected;
  for (const auto& candidate : candidates) {
    bool suppressed = false;
    for (const auto& kept : selected) {
      if (box_iou(candidate.box, kept.box) > config_.lane.nms_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      selected.push_back(candidate);
      if (static_cast<int>(selected.size()) >= config_.lane.max_instances) {
        break;
      }
    }
  }

  LaneResult result;
  result.mask = cv::Mat::zeros(info.source_height, info.source_width, CV_8UC1);
  if (!selected.empty()) {
    cv::Mat union_input = cv::Mat::zeros(config_.lane.input_height, config_.lane.input_width,
                                         CV_8UC1);
    const float threshold = std::clamp(config_.lane.mask_threshold, 1.0e-6F, 1.0F - 1.0e-6F);
    const float logit_threshold = std::log(threshold / (1.0F - threshold));
    for (const auto& candidate : selected) {
      cv::Mat logits(120, 160, CV_32FC1);
      for (int y = 0; y < 120; ++y) {
        float* row = logits.ptr<float>(y);
        for (int x = 0; x < 160; ++x) {
          const int offset = y * 160 + x;
          float value = 0.0F;
          for (int channel = 0; channel < 32; ++channel) {
            value += candidate.coeff[channel] * outputs[6].data[channel * 120 * 160 + offset];
          }
          row[x] = value;
        }
      }
      cv::Mat resized_logits;
      cv::resize(logits, resized_logits,
                 cv::Size(config_.lane.input_width, config_.lane.input_height), 0.0, 0.0,
                 cv::INTER_LINEAR);
      cv::Mat binary = resized_logits >= logit_threshold;
      const int x1 = std::clamp(static_cast<int>(std::floor(candidate.box[0])), 0,
                                config_.lane.input_width);
      const int y1 = std::clamp(static_cast<int>(std::floor(candidate.box[1])), 0,
                                config_.lane.input_height);
      const int x2 = std::clamp(static_cast<int>(std::ceil(candidate.box[2])), 0,
                                config_.lane.input_width);
      const int y2 = std::clamp(static_cast<int>(std::ceil(candidate.box[3])), 0,
                                config_.lane.input_height);
      if (x2 > x1 && y2 > y1) {
        binary(cv::Rect(x1, y1, x2 - x1, y2 - y1))
            .copyTo(union_input(cv::Rect(x1, y1, x2 - x1, y2 - y1)));
      }
      LaneInstance instance;
      instance.bbox = {
          std::clamp(static_cast<int>(std::round((candidate.box[0] - info.pad_x) / info.scale)),
                     0, info.source_width),
          std::clamp(static_cast<int>(std::round((candidate.box[1] - info.pad_y) / info.scale)),
                     0, info.source_height),
          std::clamp(static_cast<int>(std::round((candidate.box[2] - info.pad_x) / info.scale)),
                     0, info.source_width),
          std::clamp(static_cast<int>(std::round((candidate.box[3] - info.pad_y) / info.scale)),
                     0, info.source_height)};
      instance.confidence = candidate.score;
      result.instances.push_back(instance);
    }
    const cv::Rect content(info.pad_x, info.pad_y, info.resized_width, info.resized_height);
    cv::resize(union_input(content), result.mask, cv::Size(info.source_width, info.source_height),
               0.0, 0.0, cv::INTER_NEAREST);
    result.confidence = selected.front().score;
    result.status = "ok";
  }

  const auto finished = Clock::now();
  result.timing.preprocess_ms = milliseconds(started, preprocessed);
  result.timing.inference_ms = milliseconds(preprocessed, inferred);
  result.timing.postprocess_ms = milliseconds(inferred, finished);
  result.timing.total_ms = milliseconds(started, finished);
  return result;
}

std::pair<std::vector<Detection>, Timing> NativeEngine::run_object(const cv::Mat& rgb) {
  const auto started = Clock::now();
  Letterbox info;
  cv::Mat input = letterbox_rgb(rgb, config_.object.input_width, config_.object.input_height, info);
  const auto preprocessed = Clock::now();
  const auto outputs = object_model_.run(input.data, input.total() * input.elemSize());
  const auto inferred = Clock::now();
  if (outputs.size() != 2 || config_.object.class_names.empty()) {
    throw std::runtime_error("native object detector expects two PP-YOLOE outputs");
  }

  const std::size_t class_count = config_.object.class_names.size();
  const TensorView* boxes = nullptr;
  const TensorView* scores = nullptr;
  for (const auto& output : outputs) {
    if (output.size % 4 == 0) {
      const std::size_t count = output.size / 4;
      const auto& other = (&output == &outputs[0]) ? outputs[1] : outputs[0];
      if (other.size == count * class_count) {
        boxes = &output;
        scores = &other;
        break;
      }
    }
  }
  if (boxes == nullptr || scores == nullptr) {
    throw std::runtime_error("unexpected PP-YOLOE output sizes");
  }

  const std::size_t count = boxes->size / 4;
  std::vector<ObjectCandidate> candidates;
  for (std::size_t index = 0; index < count; ++index) {
    int class_id = 0;
    float best = -1.0F;
    for (std::size_t cls = 0; cls < class_count; ++cls) {
      const float value = probability(scores->data[cls * count + index]);
      if (value > best) {
        best = value;
        class_id = static_cast<int>(cls);
      }
    }
    if (best < config_.object.score_threshold) {
      continue;
    }
    const float* box = boxes->data + index * 4;
    candidates.push_back(ObjectCandidate{{box[0], box[1], box[2], box[3]}, best, class_id});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.score > b.score; });
  std::vector<ObjectCandidate> selected;
  for (const auto& candidate : candidates) {
    bool suppressed = false;
    for (const auto& kept : selected) {
      if ((config_.object.class_agnostic_nms || candidate.class_id == kept.class_id) &&
          box_iou(candidate.box, kept.box) > config_.object.nms_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      selected.push_back(candidate);
      if (static_cast<int>(selected.size()) >= config_.object.max_detections) {
        break;
      }
    }
  }

  std::vector<Detection> detections;
  for (const auto& candidate : selected) {
    const float x1 = std::clamp((candidate.box[0] - info.pad_x) / info.scale, 0.0F,
                                static_cast<float>(info.source_width - 1));
    const float y1 = std::clamp((candidate.box[1] - info.pad_y) / info.scale, 0.0F,
                                static_cast<float>(info.source_height - 1));
    const float x2 = std::clamp((candidate.box[2] - info.pad_x) / info.scale, 0.0F,
                                static_cast<float>(info.source_width - 1));
    const float y2 = std::clamp((candidate.box[3] - info.pad_y) / info.scale, 0.0F,
                                static_cast<float>(info.source_height - 1));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    detections.push_back(Detection{config_.object.class_names[candidate.class_id],
                                   candidate.score,
                                   {static_cast<int>(std::round(x1)),
                                    static_cast<int>(std::round(y1)),
                                    static_cast<int>(std::round(x2)),
                                    static_cast<int>(std::round(y2))}});
  }

  const auto finished = Clock::now();
  Timing timing;
  timing.preprocess_ms = milliseconds(started, preprocessed);
  timing.inference_ms = milliseconds(preprocessed, inferred);
  timing.postprocess_ms = milliseconds(inferred, finished);
  timing.total_ms = milliseconds(started, finished);
  return {std::move(detections), timing};
}

FramePacket NativeEngine::next_frame(bool want_bgr) {
  if (!opened_) {
    throw std::runtime_error("native perception engine is not open");
  }
  const auto started = Clock::now();
  cv::Mat rgb;
  uint64_t source_frame_id = 0;
  if (!read_rgb(rgb, source_frame_id)) {
    return FramePacket{};
  }
  const double captured_at = monotonic_seconds();
  {
    std::lock_guard<std::mutex> lock(ring_mutex_);
    ring_.push_back(RingFrame{source_frame_id, rgb.clone()});
    while (ring_.size() > config_.ring_buffer_size) {
      ring_.pop_front();
    }
  }

  auto lane_future = std::async(std::launch::async, [this, &rgb]() { return run_lane(rgb); });
  auto object_future =
      std::async(std::launch::async, [this, &rgb]() { return run_object(rgb); });
  FramePacket packet;
  packet.ok = true;
  packet.frame_id = source_frame_id;
  packet.captured_at = captured_at;
  packet.width = rgb.cols;
  packet.height = rgb.rows;
  packet.lane = lane_future.get();
  auto object_result = object_future.get();
  packet.detections = std::move(object_result.first);
  packet.object_timing = object_result.second;
  if (want_bgr) {
    cv::cvtColor(rgb, packet.bgr, cv::COLOR_RGB2BGR);
  }
  packet.total_ms = milliseconds(started, Clock::now());
  return packet;
}

void NativeEngine::load_characters() {
  std::ifstream file(config_.ocr.character_dict_path);
  if (!file) {
    throw std::runtime_error("failed to open OCR character dictionary");
  }
  ocr_characters_.clear();
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    ocr_characters_.push_back(line);
  }
  ocr_characters_.push_back(" ");
}

bool NativeEngine::submit_ocr(uint64_t trigger_id, uint64_t frame_id,
                              const std::array<int, 4>& bbox) {
  if (!opened_ || trigger_id == 0 || frame_id == 0) {
    return false;
  }
  cv::Mat frame;
  {
    std::lock_guard<std::mutex> lock(ring_mutex_);
    const auto found = std::find_if(ring_.begin(), ring_.end(),
                                    [frame_id](const auto& item) {
                                      return item.frame_id == frame_id;
                                    });
    if (found == ring_.end()) {
      return false;
    }
    frame = found->rgb.clone();
  }
  {
    std::lock_guard<std::mutex> lock(ocr_mutex_);
    if (ocr_busy_ || !ocr_jobs_.empty()) {
      return false;
    }
    ocr_jobs_.push_back(OcrJob{trigger_id, frame_id, bbox, std::move(frame)});
  }
  ocr_cv_.notify_one();
  return true;
}

OcrResult NativeEngine::run_ocr(const OcrJob& job) {
  const auto started = Clock::now();
  OcrResult result;
  result.trigger_id = job.trigger_id;
  result.frame_id = job.frame_id;
  result.source_bbox = job.bbox;
  try {
    const int left = std::clamp(job.bbox[0], 0, job.rgb.cols);
    const int top = std::clamp(job.bbox[1], 0, job.rgb.rows);
    const int right = std::clamp(job.bbox[2], 0, job.rgb.cols);
    const int bottom = std::clamp(job.bbox[3], 0, job.rgb.rows);
    if (right <= left || bottom <= top) {
      throw std::runtime_error("invalid OCR source bbox");
    }
    cv::Mat crop_rgb = job.rgb(cv::Rect(left, top, right - left, bottom - top));
    cv::Mat crop_bgr;
    cv::cvtColor(crop_rgb, crop_bgr, cv::COLOR_RGB2BGR);
    const int side = std::max(crop_bgr.rows, crop_bgr.cols);
    cv::Mat square(side, side, CV_8UC3, cv::Scalar(114, 114, 114));
    const int x = (side - crop_bgr.cols) / 2;
    const int y = (side - crop_bgr.rows) / 2;
    crop_bgr.copyTo(square(cv::Rect(x, y, crop_bgr.cols, crop_bgr.rows)));
    cv::resize(square, square, cv::Size(config_.ocr.det_width, config_.ocr.det_height), 0.0,
               0.0, cv::INTER_LINEAR);

    const auto det_outputs =
        ocr_det_model_.run(square.data, square.total() * square.elemSize());
    if (det_outputs.empty() || det_outputs[0].size !=
                                   static_cast<std::size_t>(config_.ocr.det_width *
                                                            config_.ocr.det_height)) {
      throw std::runtime_error("unexpected OCR detector output size");
    }
    cv::Mat probability_map(config_.ocr.det_height, config_.ocr.det_width, CV_32FC1,
                            const_cast<float*>(det_outputs[0].data));
    cv::Mat bitmap = probability_map > config_.ocr.det_threshold;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::array<cv::Point2f, 4>> boxes;
    const std::size_t contour_limit = std::min<std::size_t>(1000, contours.size());
    for (std::size_t index = 0; index < contour_limit; ++index) {
      cv::RotatedRect rect = cv::minAreaRect(contours[index]);
      if (std::min(rect.size.width, rect.size.height) < 3.0F) {
        continue;
      }
      cv::Point2f raw_points[4];
      rect.points(raw_points);
      auto ordered = order_points(raw_points);
      cv::Mat score_mask = cv::Mat::zeros(probability_map.size(), CV_8UC1);
      std::vector<cv::Point> polygon;
      for (const auto& point : ordered) {
        polygon.emplace_back(static_cast<int>(std::round(point.x)),
                             static_cast<int>(std::round(point.y)));
      }
      cv::fillPoly(score_mask, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar(1));
      const float score = static_cast<float>(cv::mean(probability_map, score_mask)[0]);
      if (score < config_.ocr.box_threshold) {
        continue;
      }
      const float perimeter = 2.0F * (rect.size.width + rect.size.height);
      const float distance = rect.size.area() * config_.ocr.unclip_ratio /
                             std::max(1.0F, perimeter);
      rect.size.width += 2.0F * distance;
      rect.size.height += 2.0F * distance;
      if (std::min(rect.size.width, rect.size.height) < 5.0F) {
        continue;
      }
      rect.points(raw_points);
      ordered = order_points(raw_points);
      for (auto& point : ordered) {
        point.x = std::clamp(point.x, 0.0F, static_cast<float>(square.cols - 1));
        point.y = std::clamp(point.y, 0.0F, static_cast<float>(square.rows - 1));
      }
      if (cv::norm(ordered[0] - ordered[1]) <= 3.0 ||
          cv::norm(ordered[0] - ordered[3]) <= 3.0) {
        continue;
      }
      boxes.push_back(ordered);
    }
    std::sort(boxes.begin(), boxes.end(), [](const auto& a, const auto& b) {
      if (std::abs(a[0].y - b[0].y) < 10.0F) {
        return a[0].x < b[0].x;
      }
      return a[0].y < b[0].y;
    });

    double weighted_score = 0.0;
    std::size_t total_chars = 0;
    for (const auto& box : boxes) {
      cv::Mat text_crop = perspective_crop(square, box);
      if (text_crop.empty()) {
        continue;
      }
      cv::Mat resized;
      cv::resize(text_crop, resized, cv::Size(config_.ocr.rec_width, config_.ocr.rec_height),
                 0.0, 0.0, cv::INTER_LINEAR);
      cv::Mat normalized;
      resized.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
      const auto rec_outputs =
          ocr_rec_model_.run(normalized.data, normalized.total() * normalized.elemSize());
      if (rec_outputs.empty()) {
        continue;
      }
      const std::size_t vocabulary = ocr_characters_.size() + 1;
      if (vocabulary <= 1 || rec_outputs[0].size % vocabulary != 0) {
        throw std::runtime_error("unexpected OCR recognizer output size");
      }
      const std::size_t steps = rec_outputs[0].size / vocabulary;
      int previous = -1;
      std::string text;
      double score_sum = 0.0;
      std::size_t score_count = 0;
      for (std::size_t step = 0; step < steps; ++step) {
        const float* row = rec_outputs[0].data + step * vocabulary;
        const auto best = std::max_element(row, row + vocabulary);
        const int token = static_cast<int>(best - row);
        if (token != 0 && token != previous &&
            static_cast<std::size_t>(token - 1) < ocr_characters_.size()) {
          text += ocr_characters_[token - 1];
          score_sum += *best;
          ++score_count;
        }
        previous = token;
      }
      const float score = score_count == 0 ? 0.0F : static_cast<float>(score_sum / score_count);
      OcrTextItem item{text, score, flatten_box(box)};
      result.items.push_back(item);
      if (!text.empty() && score >= config_.ocr.min_score) {
        result.text += text;
        weighted_score += static_cast<double>(score) * text.size();
        total_chars += text.size();
      }
    }
    result.confidence =
        total_chars == 0 ? 0.0F : static_cast<float>(weighted_score / total_chars);
  } catch (const std::exception& error) {
    result.error = error.what();
  }
  result.inference_ms = milliseconds(started, Clock::now());
  return result;
}

void NativeEngine::ocr_worker_loop() {
  while (true) {
    OcrJob job;
    {
      std::unique_lock<std::mutex> lock(ocr_mutex_);
      ocr_cv_.wait(lock, [this]() { return ocr_stop_ || !ocr_jobs_.empty(); });
      if (ocr_stop_ && ocr_jobs_.empty()) {
        return;
      }
      job = std::move(ocr_jobs_.front());
      ocr_jobs_.pop_front();
      ocr_busy_ = true;
    }
    OcrResult result = run_ocr(job);
    {
      std::lock_guard<std::mutex> lock(ocr_mutex_);
      ocr_busy_ = false;
      ocr_results_.push_back(std::move(result));
      while (ocr_results_.size() > 4) {
        ocr_results_.pop_front();
      }
    }
  }
}

std::optional<OcrResult> NativeEngine::poll_ocr() {
  std::lock_guard<std::mutex> lock(ocr_mutex_);
  if (ocr_results_.empty()) {
    return std::nullopt;
  }
  OcrResult result = std::move(ocr_results_.front());
  ocr_results_.pop_front();
  return result;
}

void NativeEngine::close() noexcept {
  {
    std::lock_guard<std::mutex> lock(ocr_mutex_);
    ocr_stop_ = true;
  }
  ocr_cv_.notify_all();
  if (ocr_thread_.joinable()) {
    ocr_thread_.join();
  }
  {
    std::lock_guard<std::mutex> lock(ocr_mutex_);
    ocr_jobs_.clear();
    ocr_results_.clear();
    ocr_busy_ = false;
  }
  {
    std::lock_guard<std::mutex> lock(ring_mutex_);
    ring_.clear();
  }
  ocr_rec_model_.close();
  ocr_det_model_.close();
  object_model_.close();
  lane_model_.close();
  close_capture();
  opened_ = false;
}

}  // namespace xsmart
