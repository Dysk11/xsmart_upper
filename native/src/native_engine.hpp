#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "rknn_model.hpp"

namespace xsmart {

struct CameraConfig {
  std::string mode = "shared_memory";
  int device_id = 0;
  std::string video_path;
  std::string shared_memory_name = "shm_ar_video";
  bool loop_video = false;
  bool mirror = false;
  int width = 640;
  int height = 480;
  int fps = 60;
  int reconnect_attempts = 5;
  double reconnect_interval_sec = 0.5;
};

struct LaneConfig {
  std::string model_path;
  int input_width = 640;
  int input_height = 480;
  float score_threshold = 0.30F;
  float nms_threshold = 0.45F;
  float mask_threshold = 0.50F;
  int max_instances = 3;
};

struct ObjectConfig {
  std::string model_path;
  int input_width = 640;
  int input_height = 480;
  float score_threshold = 0.50F;
  float nms_threshold = 0.45F;
  int max_detections = 30;
  bool class_agnostic_nms = false;
  std::vector<std::string> class_names;
};

struct OcrConfig {
  std::string det_model_path;
  std::string rec_model_path;
  std::string character_dict_path;
  int det_width = 480;
  int det_height = 480;
  int rec_width = 320;
  int rec_height = 48;
  float det_threshold = 0.30F;
  float box_threshold = 0.60F;
  float unclip_ratio = 1.50F;
  float min_score = 0.50F;
};

struct EngineConfig {
  CameraConfig camera;
  LaneConfig lane;
  ObjectConfig object;
  OcrConfig ocr;
  std::size_t ring_buffer_size = 3;
};

struct Detection {
  std::string class_name;
  float confidence = 0.0F;
  std::array<int, 4> bbox{};
};

struct LaneInstance {
  std::array<int, 4> bbox{};
  float confidence = 0.0F;
};

struct Timing {
  double preprocess_ms = 0.0;
  double inference_ms = 0.0;
  double postprocess_ms = 0.0;
  double total_ms = 0.0;
};

struct LaneResult {
  cv::Mat mask;
  std::vector<LaneInstance> instances;
  float confidence = 0.0F;
  std::string status = "no_detection";
  Timing timing;
};

struct FramePacket {
  bool ok = false;
  uint64_t frame_id = 0;
  double captured_at = 0.0;
  int width = 0;
  int height = 0;
  cv::Mat bgr;
  LaneResult lane;
  std::vector<Detection> detections;
  Timing object_timing;
  double total_ms = 0.0;
};

struct OcrTextItem {
  std::string text;
  float score = 0.0F;
  std::array<float, 8> box{};
};

struct OcrResult {
  uint64_t trigger_id = 0;
  uint64_t frame_id = 0;
  std::array<int, 4> source_bbox{};
  std::string text;
  float confidence = 0.0F;
  double inference_ms = 0.0;
  std::string error;
  std::vector<OcrTextItem> items;
};

class NativeEngine {
 public:
  explicit NativeEngine(EngineConfig config);
  NativeEngine(const NativeEngine&) = delete;
  NativeEngine& operator=(const NativeEngine&) = delete;
  ~NativeEngine();

  void open();
  FramePacket next_frame(bool want_bgr);
  bool submit_ocr(uint64_t trigger_id, uint64_t frame_id, const std::array<int, 4>& bbox);
  std::optional<OcrResult> poll_ocr();
  void close() noexcept;
  bool is_open() const { return opened_; }

 private:
  struct Letterbox {
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
    int resized_width = 0;
    int resized_height = 0;
    int source_width = 0;
    int source_height = 0;
  };

  struct RingFrame {
    uint64_t frame_id = 0;
    cv::Mat rgb;
  };

  struct OcrJob {
    uint64_t trigger_id = 0;
    uint64_t frame_id = 0;
    std::array<int, 4> bbox{};
    cv::Mat rgb;
  };

  bool read_rgb(cv::Mat& rgb, uint64_t& source_frame_id);
  bool read_shared_memory(cv::Mat& rgb, uint64_t& source_frame_id);
  void open_capture();
  void close_capture() noexcept;
  static cv::Mat letterbox_rgb(const cv::Mat& rgb, int width, int height, Letterbox& info);
  LaneResult run_lane(const cv::Mat& rgb);
  std::pair<std::vector<Detection>, Timing> run_object(const cv::Mat& rgb);
  OcrResult run_ocr(const OcrJob& job);
  void ocr_worker_loop();
  void load_characters();

  EngineConfig config_;
  bool opened_ = false;
  uint64_t frame_counter_ = 0;
  uint64_t shared_last_frame_id_ = 0;
  cv::VideoCapture capture_;
  int shm_fd_ = -1;
  void* shm_addr_ = nullptr;
  std::size_t shm_size_ = 0;

  RknnModel lane_model_;
  RknnModel object_model_;
  RknnModel ocr_det_model_;
  RknnModel ocr_rec_model_;
  std::vector<std::string> ocr_characters_;

  std::mutex ring_mutex_;
  std::deque<RingFrame> ring_;

  std::mutex ocr_mutex_;
  std::condition_variable ocr_cv_;
  std::deque<OcrJob> ocr_jobs_;
  std::deque<OcrResult> ocr_results_;
  bool ocr_stop_ = false;
  bool ocr_busy_ = false;
  std::thread ocr_thread_;
};

}  // namespace xsmart
