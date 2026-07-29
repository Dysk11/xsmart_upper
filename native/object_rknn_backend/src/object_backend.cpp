#include "object_backend.h"

#include <rknn_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void set_error(char* output, const std::uint32_t capacity,
               const std::string& message) {
  if (output == nullptr || capacity == 0) {
    return;
  }
  const std::size_t count =
      std::min<std::size_t>(capacity - 1U, message.size());
  std::memcpy(output, message.data(), count);
  output[count] = '\0';
}

std::vector<std::uint8_t> read_model(const char* path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("failed to open RKNN model");
  }
  const auto size = stream.tellg();
  if (size <= 0) {
    throw std::runtime_error("RKNN model is empty");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char*>(bytes.data()), size)) {
    throw std::runtime_error("failed to read RKNN model");
  }
  return bytes;
}

float half_to_float(const std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1FU;
  std::uint32_t mantissa = value & 0x03FFU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127U - 15U + 1U;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        --exponent;
      }
      mantissa &= 0x03FFU;
      bits = sign | (exponent << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 31U) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U);
  }
  float output = 0.0F;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

std::size_t element_size(const rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_INT8:
    case RKNN_TENSOR_UINT8:
      return 1U;
    case RKNN_TENSOR_INT16:
    case RKNN_TENSOR_FLOAT16:
      return 2U;
    case RKNN_TENSOR_INT32:
    case RKNN_TENSOR_FLOAT32:
      return 4U;
    default:
      throw std::runtime_error("unsupported RKNN tensor type");
  }
}

float read_scalar(const rknn_tensor_attr& attr, const void* raw,
                  const std::size_t index) {
  const auto* bytes = static_cast<const std::uint8_t*>(raw);
  float value = 0.0F;
  switch (attr.type) {
    case RKNN_TENSOR_INT8:
      value = static_cast<float>(
          reinterpret_cast<const std::int8_t*>(bytes)[index]);
      break;
    case RKNN_TENSOR_UINT8:
      value = static_cast<float>(bytes[index]);
      break;
    case RKNN_TENSOR_INT16:
      value = static_cast<float>(
          reinterpret_cast<const std::int16_t*>(bytes)[index]);
      break;
    case RKNN_TENSOR_INT32:
      value = static_cast<float>(
          reinterpret_cast<const std::int32_t*>(bytes)[index]);
      break;
    case RKNN_TENSOR_FLOAT16:
      return half_to_float(
          reinterpret_cast<const std::uint16_t*>(bytes)[index]);
    case RKNN_TENSOR_FLOAT32:
      return reinterpret_cast<const float*>(bytes)[index];
    default:
      throw std::runtime_error("unsupported native RKNN output type");
  }
  if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
    return (value - static_cast<float>(attr.zp)) * attr.scale;
  }
  if (attr.qnt_type == RKNN_TENSOR_QNT_DFP) {
    return std::ldexp(value, -attr.fl);
  }
  return value;
}

class TensorReader {
 public:
  TensorReader(const rknn_tensor_attr& native_attr,
               const rknn_tensor_attr& logical_attr, const void* data)
      : native_(native_attr), logical_(logical_attr), data_(data) {}

  std::vector<std::uint32_t> squeezed_dims() const {
    std::vector<std::uint32_t> output;
    for (std::uint32_t index = 0; index < logical_.n_dims; ++index) {
      if (logical_.dims[index] != 1U) {
        output.push_back(logical_.dims[index]);
      }
    }
    return output;
  }

  float flat(const std::size_t index) const {
    if (index >= logical_.n_elems) {
      throw std::out_of_range("RKNN tensor index out of range");
    }
    // PP-YOLOE exports use UNDEFINED/contiguous 3-D outputs. For ordinary
    // NCHW/NHWC tensors without padding, the logical row-major index is also
    // the native index. Reject padded layouts instead of silently decoding
    // incorrect boxes.
    if ((native_.w_stride > 0 || native_.h_stride > 0) &&
        native_.size_with_stride >
            native_.n_elems * element_size(native_.type)) {
      throw std::runtime_error(
          "padded PP-YOLOE output layout is unsupported");
    }
    return read_scalar(native_, data_, index);
  }

 private:
  rknn_tensor_attr native_{};
  rknn_tensor_attr logical_{};
  const void* data_ = nullptr;
};

struct Letterbox {
  float scale = 1.0F;
  std::uint32_t pad_x = 0;
  std::uint32_t pad_y = 0;
  std::uint32_t resized_width = 0;
  std::uint32_t resized_height = 0;
  std::uint32_t source_width = 0;
  std::uint32_t source_height = 0;
};

struct Candidate {
  float score = 0.0F;
  std::int32_t class_id = 0;
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
};

float intersection_over_union(const Candidate& left, const Candidate& right) {
  const float x1 = std::max(left.x1, right.x1);
  const float y1 = std::max(left.y1, right.y1);
  const float x2 = std::min(left.x2, right.x2);
  const float y2 = std::min(left.y2, right.y2);
  const float intersection =
      std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  const float left_area =
      std::max(0.0F, left.x2 - left.x1) *
      std::max(0.0F, left.y2 - left.y1);
  const float right_area =
      std::max(0.0F, right.x2 - right.x1) *
      std::max(0.0F, right.y2 - right.y1);
  return intersection /
         std::max(1.0e-6F, left_area + right_area - intersection);
}

class Runtime {
 public:
  Runtime(const std::vector<std::uint8_t>& model,
          const xsmart_object_config& config, const std::uint32_t index,
          std::mutex* npu_mutex)
      : config_(config), index_(index), npu_mutex_(npu_mutex) {
    if (rknn_init(&context_, const_cast<std::uint8_t*>(model.data()),
                  static_cast<std::uint32_t>(model.size()), 0, nullptr) !=
        RKNN_SUCC) {
      throw std::runtime_error("rknn_init failed");
    }
    try {
      if (rknn_set_core_mask(context_, RKNN_NPU_CORE_2) != RKNN_SUCC) {
        throw std::runtime_error("failed to bind object detector to NPU2");
      }
      initialize_io();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  ~Runtime() { cleanup(); }

  void cleanup() noexcept {
    for (auto* memory : output_memories_) {
      if (memory != nullptr) {
        rknn_destroy_mem(context_, memory);
      }
    }
    output_memories_.clear();
    if (input_memory_ != nullptr) {
      rknn_destroy_mem(context_, input_memory_);
      input_memory_ = nullptr;
    }
    if (context_ != 0) {
      rknn_destroy(context_);
      context_ = 0;
    }
  }

  void detect(const std::uint8_t* rgb, const std::uint32_t width,
              const std::uint32_t height, const std::uint32_t row_stride,
              const std::uint64_t frame_id, xsmart_object_result* result) {
    std::lock_guard<std::mutex> guard(mutex_);
    std::memset(result, 0, sizeof(*result));
    result->abi_version = XSMART_OBJECT_ABI_VERSION;
    result->struct_size = sizeof(*result);
    result->frame_id = frame_id;
    result->context_index = index_;
    result->core_mask = XSMART_OBJECT_NPU_CORE_2_MASK;
    const auto started = Clock::now();

    Letterbox letterbox;
    const auto preprocess_started = Clock::now();
    preprocess(rgb, width, height, row_stride, &letterbox);
    const auto preprocess_finished = Clock::now();
    result->preprocess_ms =
        static_cast<float>(elapsed_ms(preprocess_started, preprocess_finished));

    {
      // Both contexts are pinned to the same physical NPU core. Submitting two
      // rknn_run calls concurrently makes the driver queue both graphs and
      // doubles per-frame latency. Serialize only the device stage so CPU
      // preprocessing/postprocessing can still overlap across contexts.
      std::lock_guard<std::mutex> npu_guard(*npu_mutex_);
      const auto input_sync_started = Clock::now();
      if (rknn_mem_sync(context_, input_memory_, RKNN_MEMORY_SYNC_TO_DEVICE) !=
          RKNN_SUCC) {
        throw std::runtime_error("RKNN input memory sync failed");
      }
      const auto input_sync_finished = Clock::now();
      result->input_sync_ms = static_cast<float>(
          elapsed_ms(input_sync_started, input_sync_finished));

      rknn_run_extend extend{};
      extend.non_block = 0;
      const auto inference_started = Clock::now();
      if (rknn_run(context_, &extend) != RKNN_SUCC) {
        throw std::runtime_error("rknn_run failed");
      }
      const auto inference_finished = Clock::now();
      result->inference_ms =
          static_cast<float>(elapsed_ms(inference_started, inference_finished));
      rknn_perf_run perf{};
      if (rknn_query(context_, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf)) ==
          RKNN_SUCC) {
        result->npu_run_ms = static_cast<float>(perf.run_duration / 1000.0);
      }

      const auto output_sync_started = Clock::now();
      for (auto* memory : output_memories_) {
        if (rknn_mem_sync(context_, memory, RKNN_MEMORY_SYNC_FROM_DEVICE) !=
            RKNN_SUCC) {
          throw std::runtime_error("RKNN output memory sync failed");
        }
      }
      const auto output_sync_finished = Clock::now();
      result->output_sync_ms = static_cast<float>(
          elapsed_ms(output_sync_started, output_sync_finished));
    }

    const auto postprocess_started = Clock::now();
    postprocess(letterbox, result);
    const auto finished = Clock::now();
    result->postprocess_ms =
        static_cast<float>(elapsed_ms(postprocess_started, finished));
    result->total_ms = static_cast<float>(elapsed_ms(started, finished));
  }

 private:
  void initialize_io() {
    rknn_input_output_num count{};
    if (rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)) !=
            RKNN_SUCC ||
        count.n_input != 1 || count.n_output != 2) {
      throw std::runtime_error(
          "object RKNN model must expose one input and two outputs");
    }
    std::memset(&input_attr_, 0, sizeof(input_attr_));
    input_attr_.index = 0;
    if (rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr_,
                   sizeof(input_attr_)) != RKNN_SUCC ||
        input_attr_.n_dims != 4) {
      throw std::runtime_error("failed to query object input attributes");
    }
    if (input_attr_.fmt == RKNN_TENSOR_NHWC) {
      input_height_ = input_attr_.dims[1];
      input_width_ = input_attr_.dims[2];
      input_channels_ = input_attr_.dims[3];
    } else if (input_attr_.fmt == RKNN_TENSOR_NCHW) {
      input_channels_ = input_attr_.dims[1];
      input_height_ = input_attr_.dims[2];
      input_width_ = input_attr_.dims[3];
    } else {
      throw std::runtime_error("unsupported object RKNN input layout");
    }
    if (input_width_ != 640 || input_height_ != 480 ||
        input_channels_ != 3 || input_attr_.n_elems != 640U * 480U * 3U) {
      throw std::runtime_error(
          "object RKNN input must be RGB 640x480");
    }
    // rknn_set_io_mem describes the external buffer contract. The graph may
    // use an internal NCHW layout; the runtime performs that conversion.
    input_attr_.type = RKNN_TENSOR_UINT8;
    input_attr_.fmt = RKNN_TENSOR_NHWC;
    input_attr_.pass_through = 0;
    const std::uint32_t input_bytes =
        input_attr_.size_with_stride > 0
            ? input_attr_.size_with_stride
            : input_width_ * input_height_ * input_channels_;
    input_memory_ = rknn_create_mem(context_, input_bytes);
    if (input_memory_ == nullptr ||
        rknn_set_io_mem(context_, input_memory_, &input_attr_) != RKNN_SUCC) {
      throw std::runtime_error("failed to bind object input memory");
    }

    native_output_attrs_.resize(2);
    logical_output_attrs_.resize(2);
    output_memories_.resize(2, nullptr);
    for (std::uint32_t index = 0; index < 2; ++index) {
      auto& native = native_output_attrs_[index];
      auto& logical = logical_output_attrs_[index];
      std::memset(&native, 0, sizeof(native));
      std::memset(&logical, 0, sizeof(logical));
      native.index = index;
      logical.index = index;
      if (rknn_query(context_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &native,
                     sizeof(native)) != RKNN_SUCC ||
          rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &logical,
                     sizeof(logical)) != RKNN_SUCC) {
        throw std::runtime_error("failed to query object output attributes");
      }
      const std::size_t bytes_per_element = element_size(native.type);
      if (native.size < logical.n_elems * bytes_per_element ||
          (native.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
           (!std::isfinite(native.scale) || native.scale == 0.0F))) {
        throw std::runtime_error(
            "invalid native PP-YOLOE output dtype or quantization");
      }
      const std::uint32_t bytes =
          native.size_with_stride > 0 ? native.size_with_stride : native.size;
      output_memories_[index] = rknn_create_mem(context_, bytes);
      if (output_memories_[index] == nullptr ||
          rknn_set_io_mem(context_, output_memories_[index], &native) !=
              RKNN_SUCC) {
        throw std::runtime_error("failed to bind object output memory");
      }
    }
    const TensorReader boxes(native_output_attrs_[0],
                             logical_output_attrs_[0],
                             output_memories_[0]->virt_addr);
    const TensorReader scores(native_output_attrs_[1],
                              logical_output_attrs_[1],
                              output_memories_[1]->virt_addr);
    const auto box_dims = boxes.squeezed_dims();
    const auto score_dims = scores.squeezed_dims();
    if (box_dims.size() != 2 || score_dims.size() != 2 ||
        std::find(box_dims.begin(), box_dims.end(), 4U) == box_dims.end() ||
        std::find(score_dims.begin(), score_dims.end(), config_.class_count) ==
            score_dims.end()) {
      throw std::runtime_error(
          "unexpected PP-YOLOE boxes/scores tensor contract");
    }
    box_count_ = logical_output_attrs_[0].n_elems / 4U;
    if (logical_output_attrs_[1].n_elems !=
        box_count_ * config_.class_count) {
      throw std::runtime_error("PP-YOLOE box/score counts do not match");
    }
    boxes_coordinate_last_ = box_dims.back() == 4U;
    scores_class_first_ = score_dims.front() == config_.class_count;
    candidates_.reserve(box_count_);
  }

  void preprocess(const std::uint8_t* rgb, const std::uint32_t width,
                  const std::uint32_t height, const std::uint32_t row_stride,
                  Letterbox* info) {
    if (rgb == nullptr || width == 0 || height == 0 ||
        row_stride < width * 3U) {
      throw std::invalid_argument("invalid RGB input");
    }
    info->source_width = width;
    info->source_height = height;
    info->scale =
        std::min(static_cast<float>(input_width_) / width,
                 static_cast<float>(input_height_) / height);
    info->resized_width =
        static_cast<std::uint32_t>(std::lround(width * info->scale));
    info->resized_height =
        static_cast<std::uint32_t>(std::lround(height * info->scale));
    info->pad_x = (input_width_ - info->resized_width) / 2U;
    info->pad_y = (input_height_ - info->resized_height) / 2U;
    auto* destination =
        static_cast<std::uint8_t*>(input_memory_->virt_addr);
    const std::uint32_t destination_stride =
        (input_attr_.w_stride > 0 ? input_attr_.w_stride : input_width_) * 3U;
    if (width == input_width_ && height == input_height_) {
      for (std::uint32_t row = 0; row < input_height_; ++row) {
        std::memcpy(destination + static_cast<std::size_t>(row) *
                                      destination_stride,
                    rgb + static_cast<std::size_t>(row) * row_stride,
                    input_width_ * 3U);
      }
      return;
    }
    std::memset(destination, 114,
                static_cast<std::size_t>(input_height_) * destination_stride);
    for (std::uint32_t dy = 0; dy < info->resized_height; ++dy) {
      const std::uint32_t sy = std::min<std::uint32_t>(
          height - 1U,
          static_cast<std::uint32_t>(dy / info->scale));
      for (std::uint32_t dx = 0; dx < info->resized_width; ++dx) {
        const std::uint32_t sx = std::min<std::uint32_t>(
            width - 1U,
            static_cast<std::uint32_t>(dx / info->scale));
        std::memcpy(
            destination +
                static_cast<std::size_t>(dy + info->pad_y) *
                    destination_stride +
                static_cast<std::size_t>(dx + info->pad_x) * 3U,
            rgb + static_cast<std::size_t>(sy) * row_stride +
                static_cast<std::size_t>(sx) * 3U,
            3U);
      }
    }
  }

  float box_value(const TensorReader& boxes, const std::uint32_t box,
                  const std::uint32_t coordinate) const {
    return boxes.flat(boxes_coordinate_last_
                          ? static_cast<std::size_t>(box) * 4U + coordinate
                          : static_cast<std::size_t>(coordinate) * box_count_ +
                                box);
  }

  float score_value(const TensorReader& scores, const std::uint32_t box,
                    const std::uint32_t class_id) const {
    return scores.flat(scores_class_first_
                           ? static_cast<std::size_t>(class_id) * box_count_ +
                                 box
                           : static_cast<std::size_t>(box) *
                                     config_.class_count +
                                 class_id);
  }

  void postprocess(const Letterbox& info, xsmart_object_result* result) {
    const TensorReader boxes(native_output_attrs_[0], logical_output_attrs_[0],
                             output_memories_[0]->virt_addr);
    const TensorReader scores(native_output_attrs_[1],
                              logical_output_attrs_[1],
                              output_memories_[1]->virt_addr);
    candidates_.clear();
    for (std::uint32_t box = 0; box < box_count_; ++box) {
      std::uint32_t best_class = 0;
      float best_score = -std::numeric_limits<float>::infinity();
      for (std::uint32_t class_id = 0; class_id < config_.class_count;
           ++class_id) {
        const float score = score_value(scores, box, class_id);
        if (score > best_score) {
          best_score = score;
          best_class = class_id;
        }
      }
      if (best_score < config_.score_threshold) {
        continue;
      }
      Candidate candidate;
      candidate.score = best_score;
      candidate.class_id = static_cast<std::int32_t>(best_class);
      candidate.x1 = box_value(boxes, box, 0);
      candidate.y1 = box_value(boxes, box, 1);
      candidate.x2 = box_value(boxes, box, 2);
      candidate.y2 = box_value(boxes, box, 3);
      candidates_.push_back(candidate);
    }
    std::sort(candidates_.begin(), candidates_.end(),
              [](const Candidate& left, const Candidate& right) {
                return left.score > right.score;
              });
    std::vector<Candidate> selected;
    selected.reserve(config_.max_detections);
    for (const auto& candidate : candidates_) {
      bool suppressed = false;
      for (const auto& existing : selected) {
        if ((config_.class_agnostic_nms != 0U ||
             existing.class_id == candidate.class_id) &&
            intersection_over_union(existing, candidate) >
                config_.nms_threshold) {
          suppressed = true;
          break;
        }
      }
      if (suppressed) {
        continue;
      }
      selected.push_back(candidate);
      if (selected.size() >= config_.max_detections) {
        break;
      }
    }
    for (const auto& candidate : selected) {
      auto& detection = result->detections[result->detection_count++];
      detection.class_id = candidate.class_id;
      detection.confidence = candidate.score;
      detection.x1 = std::clamp<int>(
          std::lround((candidate.x1 - info.pad_x) / info.scale), 0,
          static_cast<int>(info.source_width - 1U));
      detection.y1 = std::clamp<int>(
          std::lround((candidate.y1 - info.pad_y) / info.scale), 0,
          static_cast<int>(info.source_height - 1U));
      detection.x2 = std::clamp<int>(
          std::lround((candidate.x2 - info.pad_x) / info.scale), 0,
          static_cast<int>(info.source_width - 1U));
      detection.y2 = std::clamp<int>(
          std::lround((candidate.y2 - info.pad_y) / info.scale), 0,
          static_cast<int>(info.source_height - 1U));
    }
  }

  xsmart_object_config config_{};
  std::uint32_t index_ = 0;
  rknn_context context_ = 0;
  rknn_tensor_attr input_attr_{};
  rknn_tensor_mem* input_memory_ = nullptr;
  std::vector<rknn_tensor_attr> native_output_attrs_;
  std::vector<rknn_tensor_attr> logical_output_attrs_;
  std::vector<rknn_tensor_mem*> output_memories_;
  std::uint32_t input_width_ = 0;
  std::uint32_t input_height_ = 0;
  std::uint32_t input_channels_ = 0;
  std::uint32_t box_count_ = 0;
  bool boxes_coordinate_last_ = true;
  bool scores_class_first_ = true;
  std::vector<Candidate> candidates_;
  std::mutex mutex_;
  std::mutex* npu_mutex_ = nullptr;
};

class Detector {
 public:
  Detector(const char* model_path, const xsmart_object_config& config)
      : config_(config) {
    if (config_.abi_version != XSMART_OBJECT_ABI_VERSION ||
        config_.struct_size != sizeof(xsmart_object_config)) {
      throw std::invalid_argument("object C ABI version mismatch");
    }
    if (config_.core_mask != XSMART_OBJECT_NPU_CORE_2_MASK) {
      throw std::invalid_argument(
          "object detector core_mask must be NPU_CORE_2");
    }
    if (config_.pipeline_depth != 2U) {
      throw std::invalid_argument(
          "object detector pipeline_depth must be 2");
    }
    if (config_.class_count == 0U ||
        config_.max_detections == 0U ||
        config_.max_detections > XSMART_OBJECT_MAX_DETECTIONS) {
      throw std::invalid_argument("invalid object detector limits");
    }
    const auto model = read_model(model_path);
    runtimes_.reserve(config_.pipeline_depth);
    for (std::uint32_t index = 0; index < config_.pipeline_depth; ++index) {
      runtimes_.push_back(
          std::make_unique<Runtime>(model, config_, index, &npu_mutex_));
    }
  }

  void detect(const std::uint8_t* rgb, const std::uint32_t width,
              const std::uint32_t height, const std::uint32_t row_stride,
              const std::uint64_t frame_id, xsmart_object_result* result) {
    const std::uint32_t index =
        next_context_.fetch_add(1U, std::memory_order_relaxed) %
        static_cast<std::uint32_t>(runtimes_.size());
    runtimes_[index]->detect(rgb, width, height, row_stride, frame_id, result);
  }

 private:
  xsmart_object_config config_{};
  std::vector<std::unique_ptr<Runtime>> runtimes_;
  std::atomic<std::uint32_t> next_context_{0};
  std::mutex npu_mutex_;
};

}  // namespace

extern "C" int xsmart_object_create(
    const char* model_path, const xsmart_object_config* config,
    xsmart_object_handle* output, char* error_message,
    const std::uint32_t error_capacity) {
  if (model_path == nullptr || config == nullptr || output == nullptr) {
    set_error(error_message, error_capacity, "invalid create arguments");
    return -1;
  }
  *output = nullptr;
  try {
    *output = new Detector(model_path, *config);
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_capacity, error.what());
    return -1;
  }
}

extern "C" int xsmart_object_detect(
    xsmart_object_handle handle, const std::uint8_t* rgb,
    const std::uint32_t width, const std::uint32_t height,
    const std::uint32_t row_stride, const std::uint64_t frame_id,
    xsmart_object_result* output, char* error_message,
    const std::uint32_t error_capacity) {
  if (handle == nullptr || output == nullptr) {
    set_error(error_message, error_capacity, "invalid detect arguments");
    return -1;
  }
  try {
    static_cast<Detector*>(handle)->detect(
        rgb, width, height, row_stride, frame_id, output);
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_capacity, error.what());
    return -1;
  }
}

extern "C" void xsmart_object_destroy(xsmart_object_handle handle) {
  delete static_cast<Detector*>(handle);
}

extern "C" const char* xsmart_object_version(void) {
  return "xsmart-object-rknn-capi/1";
}
