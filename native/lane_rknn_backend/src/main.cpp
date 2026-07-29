#include "shared_protocol.hpp"

#include <rknn_api.h>

#if defined(XSMART_HAVE_RGA)
#include <im2d.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using xsmart::BackendState;
using xsmart::GlobalHeader;
using xsmart::InputSlotHeader;
using xsmart::ResultInstance;
using xsmart::ResultSlotHeader;
using xsmart::ResultStatus;

constexpr std::array<float, 3> kStrides = {8.0F, 16.0F, 32.0F};
constexpr float kAnchors[3][3][2] = {
    {{10.0F, 13.0F}, {16.0F, 30.0F}, {33.0F, 23.0F}},
    {{30.0F, 61.0F}, {62.0F, 45.0F}, {59.0F, 119.0F}},
    {{116.0F, 90.0F}, {156.0F, 198.0F}, {373.0F, 326.0F}},
};

std::atomic<bool> g_stop{false};

std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch())
          .count());
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void signal_handler(int) { g_stop.store(true, std::memory_order_relaxed); }

template <typename T>
T atomic_load(const T* pointer) {
  return __atomic_load_n(pointer, __ATOMIC_ACQUIRE);
}

template <typename T>
void atomic_store(T* pointer, T value) {
  __atomic_store_n(pointer, value, __ATOMIC_RELEASE);
}

class SharedMapping {
 public:
  SharedMapping(const std::string& raw_name, const bool writable) {
    name_ = raw_name.empty() || raw_name.front() == '/' ? raw_name : "/" + raw_name;
    fd_ = shm_open(name_.c_str(), writable ? O_RDWR : O_RDONLY, 0);
    if (fd_ < 0) {
      throw std::runtime_error("shm_open failed for " + name_ + ": " +
                               std::strerror(errno));
    }
    struct stat info {};
    if (fstat(fd_, &info) != 0 || info.st_size <= 0) {
      const std::string message = "fstat failed for " + name_;
      close(fd_);
      fd_ = -1;
      throw std::runtime_error(message);
    }
    size_ = static_cast<std::size_t>(info.st_size);
    address_ = mmap(nullptr, size_, writable ? (PROT_READ | PROT_WRITE) : PROT_READ,
                    MAP_SHARED, fd_, 0);
    if (address_ == MAP_FAILED) {
      address_ = nullptr;
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("mmap failed for " + name_ + ": " +
                               std::strerror(errno));
    }
  }

  SharedMapping(const SharedMapping&) = delete;
  SharedMapping& operator=(const SharedMapping&) = delete;

  ~SharedMapping() {
    if (address_ != nullptr) {
      munmap(address_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  std::uint8_t* bytes() { return static_cast<std::uint8_t*>(address_); }
  const std::uint8_t* bytes() const {
    return static_cast<const std::uint8_t*>(address_);
  }
  std::size_t size() const { return size_; }

 private:
  std::string name_;
  int fd_ = -1;
  void* address_ = nullptr;
  std::size_t size_ = 0;
};

struct Options {
  std::string model_path;
  std::string input_shm;
  std::string result_shm;
  int result_event_fd = -1;
  std::array<std::uint32_t, 2> core_masks = {1U, 2U};
  std::array<int, 2> cpu_cores = {6, 7};
  std::string preprocess = "auto";
  std::string output_mode = "float";
  float score_threshold = 0.30F;
  float nms_threshold = 0.45F;
  float mask_threshold = 0.50F;
  std::uint32_t max_instances = 3;
};

Options parse_options(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + key);
    }
    const std::string value = argv[++index];
    if (key == "--model") {
      options.model_path = value;
    } else if (key == "--input-shm") {
      options.input_shm = value;
    } else if (key == "--result-shm") {
      options.result_shm = value;
    } else if (key == "--result-event-fd") {
      options.result_event_fd = std::stoi(value);
    } else if (key == "--core-masks") {
      const auto comma = value.find(',');
      if (comma == std::string::npos) {
        throw std::invalid_argument("--core-masks requires two comma-separated values");
      }
      options.core_masks[0] = static_cast<std::uint32_t>(
          std::stoul(value.substr(0, comma)));
      options.core_masks[1] = static_cast<std::uint32_t>(
          std::stoul(value.substr(comma + 1)));
    } else if (key == "--cpu-cores") {
      const auto comma = value.find(',');
      if (comma == std::string::npos) {
        throw std::invalid_argument("--cpu-cores requires two comma-separated values");
      }
      options.cpu_cores[0] = std::stoi(value.substr(0, comma));
      options.cpu_cores[1] = std::stoi(value.substr(comma + 1));
    } else if (key == "--preprocess") {
      options.preprocess = value;
    } else if (key == "--output-mode") {
      options.output_mode = value;
    } else if (key == "--score-threshold") {
      options.score_threshold = std::stof(value);
    } else if (key == "--nms-threshold") {
      options.nms_threshold = std::stof(value);
    } else if (key == "--mask-threshold") {
      options.mask_threshold = std::stof(value);
    } else if (key == "--max-instances") {
      options.max_instances =
          std::min<std::uint32_t>(3U, std::max<std::uint32_t>(1U, std::stoul(value)));
    } else {
      throw std::invalid_argument("unknown argument: " + key);
    }
  }
  if (options.model_path.empty() || options.input_shm.empty() ||
      options.result_shm.empty() || options.result_event_fd < 0) {
    throw std::invalid_argument(
        "--model, --input-shm, --result-shm and --result-event-fd are required");
  }
  if (options.preprocess != "auto" && options.preprocess != "direct" &&
      options.preprocess != "rga") {
    throw std::invalid_argument("--preprocess must be auto, direct or rga");
  }
  if (options.output_mode != "float" && options.output_mode != "native") {
    throw std::invalid_argument("--output-mode must be float or native");
  }
  return options;
}

void validate_global_header(const GlobalHeader& header, const char expected_magic[8],
                            const std::size_t mapping_size,
                            const std::size_t slot_header_size) {
  if (std::memcmp(header.magic, expected_magic, 8) != 0 ||
      header.version != xsmart::kProtocolVersion ||
      header.header_size != xsmart::kGlobalHeaderSize ||
      header.slot_size < slot_header_size ||
      mapping_size < header.header_size + xsmart::kSlotCount * header.slot_size) {
    throw std::runtime_error("incompatible shared-memory protocol");
  }
}

struct FrameJob {
  std::uint64_t publish_sequence = 0;
  std::uint64_t frame_id = 0;
  std::uint64_t source_frame_id = 0;
  std::uint64_t captured_ns = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> rgb;
};

class InputReader {
 public:
  explicit InputReader(SharedMapping& mapping)
      : mapping_(mapping), header_(reinterpret_cast<GlobalHeader*>(mapping.bytes())) {
    validate_global_header(*header_, xsmart::kInputMagic, mapping.size(),
                           xsmart::kInputSlotHeaderSize);
  }

  std::uint64_t published_sequence() const {
    return atomic_load(&header_->publish_sequence);
  }

  bool read_latest(FrameJob* output) const {
    const std::uint64_t publish_before = atomic_load(&header_->publish_sequence);
    if (publish_before == 0) {
      return false;
    }
    const std::uint32_t slot = atomic_load(&header_->published_slot);
    if (slot >= xsmart::kSlotCount) {
      return false;
    }
    const auto* slot_bytes =
        mapping_.bytes() + header_->header_size + slot * header_->slot_size;
    const auto* slot_header = reinterpret_cast<const InputSlotHeader*>(slot_bytes);
    const std::uint64_t sequence_before = atomic_load(&slot_header->sequence);
    if (sequence_before == 0 || (sequence_before & 1U) != 0U) {
      return false;
    }
    InputSlotHeader snapshot {};
    std::memcpy(&snapshot, slot_header, sizeof(snapshot));
    if (snapshot.channels != 3 || snapshot.width == 0 || snapshot.height == 0 ||
        snapshot.row_stride < snapshot.width * 3U ||
        snapshot.payload_bytes > header_->payload_capacity ||
        snapshot.payload_bytes < snapshot.row_stride * snapshot.height) {
      return false;
    }
    output->rgb.resize(static_cast<std::size_t>(snapshot.width) * snapshot.height * 3U);
    const auto* source = slot_bytes + xsmart::kInputSlotHeaderSize;
    for (std::uint32_t row = 0; row < snapshot.height; ++row) {
      std::memcpy(output->rgb.data() +
                      static_cast<std::size_t>(row) * snapshot.width * 3U,
                  source + static_cast<std::size_t>(row) * snapshot.row_stride,
                  static_cast<std::size_t>(snapshot.width) * 3U);
    }
    const std::uint64_t sequence_after = atomic_load(&slot_header->sequence);
    const std::uint64_t publish_after = atomic_load(&header_->publish_sequence);
    const std::uint32_t slot_after = atomic_load(&header_->published_slot);
    if (sequence_after != sequence_before || (sequence_after & 1U) != 0U ||
        publish_after != publish_before || slot_after != slot) {
      return false;
    }
    output->publish_sequence = publish_before;
    output->frame_id = snapshot.frame_id;
    output->source_frame_id = snapshot.source_frame_id;
    output->captured_ns = snapshot.captured_ns;
    output->width = snapshot.width;
    output->height = snapshot.height;
    return true;
  }

 private:
  SharedMapping& mapping_;
  GlobalHeader* header_;
};

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

struct TensorShape {
  std::uint32_t n = 1;
  std::uint32_t c = 0;
  std::uint32_t h = 0;
  std::uint32_t w = 0;
};

TensorShape logical_shape(const rknn_tensor_attr& attr) {
  if (attr.n_dims != 4) {
    throw std::runtime_error("expected a four-dimensional logical RKNN tensor");
  }
  if (attr.fmt == RKNN_TENSOR_NHWC) {
    return {attr.dims[0], attr.dims[3], attr.dims[1], attr.dims[2]};
  }
  return {attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3]};
}

class TensorView {
 public:
  TensorView(const rknn_tensor_attr& native_attr,
             const rknn_tensor_attr& logical_attr, const void* data)
      : native_(native_attr), logical_(logical_shape(logical_attr)),
        data_(static_cast<const std::uint8_t*>(data)) {}

  const TensorShape& shape() const { return logical_; }

  float value(const std::uint32_t c, const std::uint32_t y,
              const std::uint32_t x) const {
    const std::size_t index = native_index(c, y, x);
    float raw = 0.0F;
    switch (native_.type) {
      case RKNN_TENSOR_INT8:
        raw = static_cast<float>(reinterpret_cast<const std::int8_t*>(data_)[index]);
        break;
      case RKNN_TENSOR_UINT8:
        raw = static_cast<float>(data_[index]);
        break;
      case RKNN_TENSOR_INT16:
        raw = static_cast<float>(reinterpret_cast<const std::int16_t*>(data_)[index]);
        break;
      case RKNN_TENSOR_FLOAT16:
        raw = half_to_float(reinterpret_cast<const std::uint16_t*>(data_)[index]);
        return raw;
      case RKNN_TENSOR_FLOAT32:
        return reinterpret_cast<const float*>(data_)[index];
      default:
        throw std::runtime_error("unsupported native RKNN output tensor type");
    }
    if (native_.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
      return (raw - static_cast<float>(native_.zp)) * native_.scale;
    }
    if (native_.qnt_type == RKNN_TENSOR_QNT_DFP) {
      return std::ldexp(raw, -native_.fl);
    }
    return raw;
  }

 private:
  std::size_t native_index(const std::uint32_t c, const std::uint32_t y,
                           const std::uint32_t x) const {
    if (native_.fmt == RKNN_TENSOR_NC1HWC2 && native_.n_dims == 5) {
      const std::uint32_t h = native_.dims[2];
      const std::uint32_t w = native_.w_stride > 0 ? native_.w_stride : native_.dims[3];
      const std::uint32_t c2 = native_.dims[4];
      return (((static_cast<std::size_t>(c / c2) * h + y) * w + x) * c2) +
             (c % c2);
    }
    const std::uint32_t width =
        native_.w_stride > 0 ? native_.w_stride : logical_.w;
    if (native_.fmt == RKNN_TENSOR_NHWC) {
      return (static_cast<std::size_t>(y) * width + x) * logical_.c + c;
    }
    return (static_cast<std::size_t>(c) * logical_.h + y) * width + x;
  }

  rknn_tensor_attr native_ {};
  TensorShape logical_ {};
  const std::uint8_t* data_ = nullptr;
};

struct Candidate {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
  float score = 0.0F;
  std::uint32_t source_index = 0;
  std::array<float, 32> coefficients {};
};

float intersection_over_union(const Candidate& a, const Candidate& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float intersection =
      std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  const float area_a =
      std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  const float area_b =
      std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  return intersection / std::max(1.0e-6F, area_a + area_b - intersection);
}

struct Letterbox {
  float scale = 1.0F;
  std::uint32_t pad_x = 0;
  std::uint32_t pad_y = 0;
  std::uint32_t resized_width = 0;
  std::uint32_t resized_height = 0;
  std::uint32_t source_width = 0;
  std::uint32_t source_height = 0;
};

struct InferenceOutput {
  ResultStatus status = ResultStatus::kOk;
  std::uint64_t rknn_frame_id = 0;
  std::vector<std::uint8_t> packed_mask;
  std::uint32_t mask_width = 0;
  std::uint32_t mask_height = 0;
  std::vector<ResultInstance> instances;
  double preprocess_ms = 0.0;
  double input_sync_ms = 0.0;
  double inference_ms = 0.0;
  double output_sync_ms = 0.0;
  double postprocess_ms = 0.0;
  double total_ms = 0.0;
};

class RknnWorkerRuntime {
 public:
  RknnWorkerRuntime(const rknn_context context, std::string preprocess_mode,
                    std::string output_mode,
                    const float score_threshold, const float nms_threshold,
                    const float mask_threshold, const std::uint32_t max_instances)
      : context_(context), preprocess_mode_(std::move(preprocess_mode)),
        output_mode_(std::move(output_mode)),
        score_threshold_(score_threshold), nms_threshold_(nms_threshold),
        mask_threshold_(std::clamp(mask_threshold, 1.0e-6F, 1.0F - 1.0e-6F)),
        max_instances_(max_instances) {
    initialize_io();
    candidates_.reserve(1200);
    selected_.reserve(max_instances_);
    low_resolution_.resize(
        static_cast<std::size_t>(max_instances_) * 120U * 160U);
    union_input_.resize(
        static_cast<std::size_t>(input_width_) * input_height_);
  }

  RknnWorkerRuntime(const RknnWorkerRuntime&) = delete;
  RknnWorkerRuntime& operator=(const RknnWorkerRuntime&) = delete;

  ~RknnWorkerRuntime() { release_io(); }

  InferenceOutput run(const FrameJob& job) {
    InferenceOutput output;
    output.mask_width = job.width;
    output.mask_height = job.height;
    const auto started = Clock::now();
    Letterbox letterbox;
    const auto preprocess_started = Clock::now();
    if (!preprocess(job, &letterbox)) {
      output.status = ResultStatus::kPreprocessError;
      output.total_ms = elapsed_ms(started, Clock::now());
      return output;
    }
    const auto preprocess_finished = Clock::now();
    output.preprocess_ms = elapsed_ms(preprocess_started, preprocess_finished);

    const auto input_sync_started = Clock::now();
    if (rknn_mem_sync(context_, input_mem_, RKNN_MEMORY_SYNC_TO_DEVICE) !=
        RKNN_SUCC) {
      output.status = ResultStatus::kInferenceError;
      output.total_ms = elapsed_ms(started, Clock::now());
      return output;
    }
    const auto input_sync_finished = Clock::now();
    output.input_sync_ms = elapsed_ms(input_sync_started, input_sync_finished);

    rknn_run_extend run_extend {};
    run_extend.non_block = 0;
    const auto inference_started = Clock::now();
    if (rknn_run(context_, &run_extend) != RKNN_SUCC) {
      output.status = ResultStatus::kInferenceError;
      output.total_ms = elapsed_ms(started, Clock::now());
      return output;
    }
    const auto inference_finished = Clock::now();
    output.rknn_frame_id = run_extend.frame_id;
    output.inference_ms = elapsed_ms(inference_started, inference_finished);

    const auto output_sync_started = Clock::now();
    if (output_mode_ == "float") {
      if (rknn_outputs_get(
              context_, static_cast<std::uint32_t>(float_outputs_.size()),
              float_outputs_.data(), nullptr) != RKNN_SUCC) {
        output.status = ResultStatus::kInferenceError;
        output.total_ms = elapsed_ms(started, Clock::now());
        return output;
      }
    } else {
      for (auto* memory : output_memories_) {
        if (rknn_mem_sync(context_, memory, RKNN_MEMORY_SYNC_FROM_DEVICE) !=
            RKNN_SUCC) {
          output.status = ResultStatus::kInferenceError;
          output.total_ms = elapsed_ms(started, Clock::now());
          return output;
        }
      }
    }
    const auto output_sync_finished = Clock::now();
    output.output_sync_ms = elapsed_ms(output_sync_started, output_sync_finished);

    const auto postprocess_started = Clock::now();
    try {
      postprocess(letterbox, &output);
    } catch (const std::exception& error) {
      std::cerr << "[lane-rknn] postprocess failed: " << error.what() << '\n';
      output.status = ResultStatus::kPostprocessError;
      output.packed_mask.assign(
          (static_cast<std::size_t>(job.width) * job.height + 7U) / 8U, 0U);
    }
    if (output_mode_ == "float") {
      rknn_outputs_release(
          context_, static_cast<std::uint32_t>(float_outputs_.size()),
          float_outputs_.data());
    }
    const auto finished = Clock::now();
    output.postprocess_ms = elapsed_ms(postprocess_started, finished);
    output.total_ms = elapsed_ms(started, finished);
    return output;
  }

 private:
  void initialize_io() {
    rknn_input_output_num count {};
    if (rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)) !=
            RKNN_SUCC ||
        count.n_input != 1 || count.n_output != 7) {
      throw std::runtime_error("lane RKNN model must expose one input and seven outputs");
    }

    std::memset(&input_attr_, 0, sizeof(input_attr_));
    input_attr_.index = 0;
    if (rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr_,
                   sizeof(input_attr_)) != RKNN_SUCC) {
      throw std::runtime_error("failed to query RKNN input attributes");
    }
    const TensorShape input_shape = logical_shape(input_attr_);
    input_width_ = input_shape.w;
    input_height_ = input_shape.h;
    if (input_shape.c != 3) {
      throw std::runtime_error("lane RKNN input must contain three channels");
    }
    input_attr_.type = RKNN_TENSOR_UINT8;
    input_attr_.fmt = RKNN_TENSOR_NHWC;
    input_attr_.pass_through = 0;
    const std::uint32_t input_bytes =
        input_attr_.size_with_stride > 0 ? input_attr_.size_with_stride
                                         : input_width_ * input_height_ * 3U;
    input_mem_ = rknn_create_mem(context_, input_bytes);
    if (input_mem_ == nullptr ||
        rknn_set_io_mem(context_, input_mem_, &input_attr_) != RKNN_SUCC) {
      throw std::runtime_error("failed to allocate/bind RKNN input memory");
    }

    native_output_attrs_.resize(count.n_output);
    logical_output_attrs_.resize(count.n_output);
    float_output_attrs_.resize(count.n_output);
    float_output_buffers_.resize(count.n_output);
    float_outputs_.resize(count.n_output);
    output_memories_.reserve(count.n_output);
    for (std::uint32_t index = 0; index < count.n_output; ++index) {
      auto& native_attr = native_output_attrs_[index];
      auto& logical_attr = logical_output_attrs_[index];
      std::memset(&native_attr, 0, sizeof(native_attr));
      std::memset(&logical_attr, 0, sizeof(logical_attr));
      native_attr.index = index;
      logical_attr.index = index;
      if (rknn_query(context_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &native_attr,
                     sizeof(native_attr)) != RKNN_SUCC ||
          rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &logical_attr,
                     sizeof(logical_attr)) != RKNN_SUCC) {
        throw std::runtime_error("failed to query RKNN output attributes");
      }
      if (output_mode_ == "float") {
        auto& float_attr = float_output_attrs_[index];
        float_attr = logical_attr;
        float_attr.type = RKNN_TENSOR_FLOAT32;
        float_attr.qnt_type = RKNN_TENSOR_QNT_NONE;
        float_attr.zp = 0;
        float_attr.scale = 1.0F;
        float_attr.w_stride = 0;
        float_output_buffers_[index].resize(logical_attr.n_elems);
        auto& float_output = float_outputs_[index];
        std::memset(&float_output, 0, sizeof(float_output));
        float_output.index = index;
        float_output.want_float = 1;
        float_output.is_prealloc = 1;
        float_output.buf = float_output_buffers_[index].data();
        float_output.size = static_cast<std::uint32_t>(
            float_output_buffers_[index].size() * sizeof(float));
        continue;
      }
      const std::uint32_t bytes =
          native_attr.size_with_stride > 0 ? native_attr.size_with_stride
                                           : native_attr.size;
      auto* memory = rknn_create_mem(context_, bytes);
      if (memory == nullptr ||
          rknn_set_io_mem(context_, memory, &native_attr) != RKNN_SUCC) {
        if (memory != nullptr) {
          rknn_destroy_mem(context_, memory);
        }
        throw std::runtime_error("failed to allocate/bind RKNN output memory");
      }
      output_memories_.push_back(memory);
    }
  }

  void release_io() {
    for (auto* memory : output_memories_) {
      rknn_destroy_mem(context_, memory);
    }
    output_memories_.clear();
    float_outputs_.clear();
    float_output_buffers_.clear();
    float_output_attrs_.clear();
    if (input_mem_ != nullptr) {
      rknn_destroy_mem(context_, input_mem_);
      input_mem_ = nullptr;
    }
  }

  bool preprocess(const FrameJob& job, Letterbox* info) {
    info->source_width = job.width;
    info->source_height = job.height;
    info->scale = std::min(static_cast<float>(input_width_) / job.width,
                           static_cast<float>(input_height_) / job.height);
    info->resized_width =
        static_cast<std::uint32_t>(std::lround(job.width * info->scale));
    info->resized_height =
        static_cast<std::uint32_t>(std::lround(job.height * info->scale));
    info->pad_x = (input_width_ - info->resized_width) / 2U;
    info->pad_y = (input_height_ - info->resized_height) / 2U;
    auto* destination = static_cast<std::uint8_t*>(input_mem_->virt_addr);
    const std::uint32_t destination_width_stride =
        input_attr_.w_stride > 0 ? input_attr_.w_stride : input_width_;
    const std::size_t source_row_bytes =
        static_cast<std::size_t>(input_width_) * 3U;
    const std::size_t destination_row_bytes =
        static_cast<std::size_t>(destination_width_stride) * 3U;

    const bool exact =
        job.width == input_width_ && job.height == input_height_;
#if defined(XSMART_HAVE_RGA)
    if (exact && destination_width_stride == input_width_ &&
        preprocess_mode_ != "direct") {
      const rga_buffer_t source_buffer =
          wrapbuffer_virtualaddr(const_cast<std::uint8_t*>(job.rgb.data()),
                                 static_cast<int>(job.width),
                                 static_cast<int>(job.height), RK_FORMAT_RGB_888,
                                 static_cast<int>(job.width),
                                 static_cast<int>(job.height));
      const rga_buffer_t destination_buffer =
          wrapbuffer_virtualaddr(destination, static_cast<int>(input_width_),
                                 static_cast<int>(input_height_),
                                 RK_FORMAT_RGB_888,
                                 static_cast<int>(input_width_),
                                 static_cast<int>(input_height_));
      const IM_STATUS status = imcopy(source_buffer, destination_buffer);
      if (status == IM_STATUS_SUCCESS) {
        return true;
      }
      if (preprocess_mode_ == "rga") {
        std::cerr << "[lane-rknn] RGA imcopy failed: " << imStrError(status)
                  << "; using direct copy\n";
      }
    }
#else
    if (preprocess_mode_ == "rga") {
      std::cerr << "[lane-rknn] RGA requested but backend was built without librga\n";
    }
#endif
    if (exact) {
      for (std::uint32_t row = 0; row < input_height_; ++row) {
        std::memcpy(
            destination + static_cast<std::size_t>(row) * destination_row_bytes,
            job.rgb.data() + static_cast<std::size_t>(row) * source_row_bytes,
            source_row_bytes);
      }
      return true;
    }

    std::memset(destination, 114,
                static_cast<std::size_t>(input_height_) * destination_row_bytes);
    for (std::uint32_t dy = 0; dy < info->resized_height; ++dy) {
      const float sy =
          (static_cast<float>(dy) + 0.5F) / info->scale - 0.5F;
      const int y0 = std::max(0, std::min<int>(job.height - 1, std::floor(sy)));
      const int y1 = std::max(0, std::min<int>(job.height - 1, y0 + 1));
      const float wy = std::clamp(sy - std::floor(sy), 0.0F, 1.0F);
      for (std::uint32_t dx = 0; dx < info->resized_width; ++dx) {
        const float sx =
            (static_cast<float>(dx) + 0.5F) / info->scale - 0.5F;
        const int x0 = std::max(0, std::min<int>(job.width - 1, std::floor(sx)));
        const int x1 = std::max(0, std::min<int>(job.width - 1, x0 + 1));
        const float wx = std::clamp(sx - std::floor(sx), 0.0F, 1.0F);
        auto* pixel =
            destination +
            static_cast<std::size_t>(dy + info->pad_y) *
                destination_row_bytes +
            static_cast<std::size_t>(dx + info->pad_x) * 3U;
        for (std::uint32_t channel = 0; channel < 3; ++channel) {
          const float top =
              job.rgb[(static_cast<std::size_t>(y0) * job.width + x0) * 3U +
                      channel] *
                  (1.0F - wx) +
              job.rgb[(static_cast<std::size_t>(y0) * job.width + x1) * 3U +
                      channel] *
                  wx;
          const float bottom =
              job.rgb[(static_cast<std::size_t>(y1) * job.width + x0) * 3U +
                      channel] *
                  (1.0F - wx) +
              job.rgb[(static_cast<std::size_t>(y1) * job.width + x1) * 3U +
                      channel] *
                  wx;
          pixel[channel] = static_cast<std::uint8_t>(
              std::clamp(std::lround(top * (1.0F - wy) + bottom * wy), 0L, 255L));
        }
      }
    }
    return true;
  }

  void postprocess(const Letterbox& info, InferenceOutput* output) {
    std::array<std::unique_ptr<TensorView>, 7> views;
    for (std::size_t index = 0; index < views.size(); ++index) {
      if (output_mode_ == "float") {
        views[index] = std::make_unique<TensorView>(
            float_output_attrs_[index], logical_output_attrs_[index],
            float_output_buffers_[index].data());
      } else {
        views[index] = std::make_unique<TensorView>(
            native_output_attrs_[index], logical_output_attrs_[index],
            output_memories_[index]->virt_addr);
      }
    }
    const std::array<TensorShape, 7> expected = {
        TensorShape{1, 18, 60, 80}, TensorShape{1, 96, 60, 80},
        TensorShape{1, 18, 30, 40}, TensorShape{1, 96, 30, 40},
        TensorShape{1, 18, 15, 20}, TensorShape{1, 96, 15, 20},
        TensorShape{1, 32, 120, 160},
    };
    for (std::size_t index = 0; index < views.size(); ++index) {
      const auto& actual = views[index]->shape();
      const auto& wanted = expected[index];
      if (actual.n != wanted.n || actual.c != wanted.c ||
          actual.h != wanted.h || actual.w != wanted.w) {
        throw std::runtime_error("unexpected RKNN lane output shape");
      }
    }

    candidates_.clear();
    std::uint32_t source_index = 0;
    for (std::uint32_t level = 0; level < 3; ++level) {
      const TensorView& box = *views[level * 2U];
      const TensorView& coefficients = *views[level * 2U + 1U];
      const auto shape = box.shape();
      for (std::uint32_t anchor = 0; anchor < 3; ++anchor) {
        for (std::uint32_t y = 0; y < shape.h; ++y) {
          for (std::uint32_t x = 0; x < shape.w; ++x) {
            const std::uint32_t base = anchor * 6U;
            const float score =
                box.value(base + 4U, y, x) * box.value(base + 5U, y, x);
            if (score < score_threshold_) {
              ++source_index;
              continue;
            }
            const float center_x =
                (box.value(base, y, x) * 2.0F + static_cast<float>(x) - 0.5F) *
                kStrides[level];
            const float center_y =
                (box.value(base + 1U, y, x) * 2.0F + static_cast<float>(y) -
                 0.5F) *
                kStrides[level];
            const float width =
                std::pow(box.value(base + 2U, y, x) * 2.0F, 2.0F) *
                kAnchors[level][anchor][0];
            const float height =
                std::pow(box.value(base + 3U, y, x) * 2.0F, 2.0F) *
                kAnchors[level][anchor][1];
            Candidate candidate;
            candidate.x1 = center_x - width * 0.5F;
            candidate.y1 = center_y - height * 0.5F;
            candidate.x2 = center_x + width * 0.5F;
            candidate.y2 = center_y + height * 0.5F;
            candidate.score = score;
            candidate.source_index = source_index;
            for (std::uint32_t channel = 0; channel < 32; ++channel) {
              candidate.coefficients[channel] =
                  coefficients.value(anchor * 32U + channel, y, x);
            }
            candidates_.push_back(candidate);
            ++source_index;
          }
        }
      }
    }
    std::sort(candidates_.begin(), candidates_.end(),
              [](const Candidate& left, const Candidate& right) {
                if (left.score != right.score) {
                  return left.score > right.score;
                }
                return left.source_index > right.source_index;
              });
    selected_.clear();
    for (const auto& candidate : candidates_) {
      bool suppressed = false;
      for (const auto& existing : selected_) {
        if (intersection_over_union(candidate, existing) > nms_threshold_) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        selected_.push_back(candidate);
        if (selected_.size() >= max_instances_) {
          break;
        }
      }
    }

    output->packed_mask.assign(
        (static_cast<std::size_t>(info.source_width) * info.source_height + 7U) /
            8U,
        0U);
    if (selected_.empty()) {
      output->status = ResultStatus::kNoDetection;
      return;
    }

    const TensorView& prototype = *views[6];
    const float logit_threshold =
        std::log(mask_threshold_ / (1.0F - mask_threshold_));
    std::fill(union_input_.begin(), union_input_.end(), 0U);
    const std::size_t prototype_plane = 120U * 160U;
    for (std::size_t selected_index = 0;
         selected_index < selected_.size(); ++selected_index) {
      const auto& candidate = selected_[selected_index];
      float* low_resolution =
          low_resolution_.data() + selected_index * prototype_plane;
      for (std::uint32_t y = 0; y < 120; ++y) {
        for (std::uint32_t x = 0; x < 160; ++x) {
          float value = 0.0F;
          for (std::uint32_t channel = 0; channel < 32; ++channel) {
            value += candidate.coefficients[channel] *
                     prototype.value(channel, y, x);
          }
          low_resolution[static_cast<std::size_t>(y) * 160U + x] = value;
        }
      }
      const int x1 = std::clamp<int>(std::floor(candidate.x1), 0, input_width_);
      const int y1 = std::clamp<int>(std::floor(candidate.y1), 0, input_height_);
      const int x2 = std::clamp<int>(std::ceil(candidate.x2), 0, input_width_);
      const int y2 = std::clamp<int>(std::ceil(candidate.y2), 0, input_height_);
      for (int y = y1; y < y2; ++y) {
        const float source_y =
            (static_cast<float>(y) + 0.5F) * 120.0F / input_height_ - 0.5F;
        const int py0 = std::clamp<int>(std::floor(source_y), 0, 119);
        const int py1 = std::min(119, py0 + 1);
        const float wy = std::clamp(source_y - std::floor(source_y), 0.0F, 1.0F);
        for (int x = x1; x < x2; ++x) {
          const float source_x =
              (static_cast<float>(x) + 0.5F) * 160.0F / input_width_ - 0.5F;
          const int px0 = std::clamp<int>(std::floor(source_x), 0, 159);
          const int px1 = std::min(159, px0 + 1);
          const float wx =
              std::clamp(source_x - std::floor(source_x), 0.0F, 1.0F);
          const float top =
              low_resolution[static_cast<std::size_t>(py0) * 160U + px0] *
                  (1.0F - wx) +
              low_resolution[static_cast<std::size_t>(py0) * 160U + px1] * wx;
          const float bottom =
              low_resolution[static_cast<std::size_t>(py1) * 160U + px0] *
                  (1.0F - wx) +
              low_resolution[static_cast<std::size_t>(py1) * 160U + px1] * wx;
          if (top * (1.0F - wy) + bottom * wy >= logit_threshold) {
            union_input_[static_cast<std::size_t>(y) * input_width_ + x] = 1U;
          }
        }
      }
      ResultInstance instance {};
      instance.x1 = std::clamp<int>(
          std::lround((candidate.x1 - info.pad_x) / info.scale), 0,
          info.source_width);
      instance.y1 = std::clamp<int>(
          std::lround((candidate.y1 - info.pad_y) / info.scale), 0,
          info.source_height);
      instance.x2 = std::clamp<int>(
          std::lround((candidate.x2 - info.pad_x) / info.scale), 0,
          info.source_width);
      instance.y2 = std::clamp<int>(
          std::lround((candidate.y2 - info.pad_y) / info.scale), 0,
          info.source_height);
      instance.confidence = candidate.score;
      output->instances.push_back(instance);
    }

    for (std::uint32_t source_y = 0; source_y < info.source_height; ++source_y) {
      const std::uint32_t resized_y =
          std::min(info.resized_height - 1U,
                   static_cast<std::uint32_t>(
                       static_cast<std::uint64_t>(source_y) *
                       info.resized_height / info.source_height));
      const std::uint32_t input_y = info.pad_y + resized_y;
      for (std::uint32_t source_x = 0; source_x < info.source_width; ++source_x) {
        const std::uint32_t resized_x =
            std::min(info.resized_width - 1U,
                     static_cast<std::uint32_t>(
                         static_cast<std::uint64_t>(source_x) *
                         info.resized_width / info.source_width));
        const std::uint32_t input_x = info.pad_x + resized_x;
        if (union_input_[static_cast<std::size_t>(input_y) * input_width_ +
                         input_x] != 0U) {
          const std::size_t bit =
              static_cast<std::size_t>(source_y) * info.source_width + source_x;
          output->packed_mask[bit / 8U] |=
              static_cast<std::uint8_t>(1U << (bit % 8U));
        }
      }
    }
    output->status = ResultStatus::kOk;
  }

  rknn_context context_ = 0;
  std::string preprocess_mode_;
  std::string output_mode_;
  float score_threshold_ = 0.30F;
  float nms_threshold_ = 0.45F;
  float mask_threshold_ = 0.50F;
  std::uint32_t max_instances_ = 3;
  std::uint32_t input_width_ = 0;
  std::uint32_t input_height_ = 0;
  rknn_tensor_attr input_attr_ {};
  rknn_tensor_mem* input_mem_ = nullptr;
  std::vector<rknn_tensor_attr> native_output_attrs_;
  std::vector<rknn_tensor_attr> logical_output_attrs_;
  std::vector<rknn_tensor_attr> float_output_attrs_;
  std::vector<std::vector<float>> float_output_buffers_;
  std::vector<rknn_output> float_outputs_;
  std::vector<rknn_tensor_mem*> output_memories_;
  std::vector<Candidate> candidates_;
  std::vector<Candidate> selected_;
  std::vector<float> low_resolution_;
  std::vector<std::uint8_t> union_input_;
};

struct Counters {
  std::atomic<std::uint64_t> claimed{0};
  std::atomic<std::uint64_t> completed{0};
  std::atomic<std::uint64_t> overwritten{0};
  std::atomic<std::uint64_t> out_of_order{0};
  std::atomic<std::uint64_t> errors{0};
  std::atomic<std::uint64_t> notifications{0};
  std::atomic<std::uint64_t> notification_errors{0};
};

class ResultPublisher {
 public:
  ResultPublisher(SharedMapping& mapping, Counters& counters,
                  const int result_event_fd)
      : mapping_(mapping), counters_(counters),
        header_(reinterpret_cast<GlobalHeader*>(mapping.bytes())),
        result_event_fd_(result_event_fd) {
    validate_global_header(*header_, xsmart::kResultMagic, mapping.size(),
                           xsmart::kResultSlotHeaderSize);
  }

  void set_state(const BackendState state, const std::uint32_t error_code = 0) {
    atomic_store(&header_->error_code, error_code);
    atomic_store(&header_->state, static_cast<std::uint32_t>(state));
  }

  bool publish(const FrameJob& job, const std::uint32_t worker_index,
               const std::uint32_t core_mask, const InferenceOutput& output) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (job.frame_id <= latest_frame_id_) {
      counters_.out_of_order.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    if (output.packed_mask.size() > header_->payload_capacity) {
      counters_.errors.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    const auto publish_started = Clock::now();
    const std::uint32_t slot = 1U - atomic_load(&header_->published_slot);
    auto* slot_bytes =
        mapping_.bytes() + header_->header_size + slot * header_->slot_size;
    auto* slot_header = reinterpret_cast<ResultSlotHeader*>(slot_bytes);
    std::uint64_t sequence = atomic_load(&slot_header->sequence);
    sequence = (sequence & 1U) == 0U ? sequence + 1U : sequence + 2U;
    atomic_store(&slot_header->sequence, sequence);

    ResultSlotHeader snapshot {};
    snapshot.sequence = sequence;
    snapshot.frame_id = job.frame_id;
    snapshot.source_frame_id = job.source_frame_id;
    snapshot.captured_ns = job.captured_ns;
    snapshot.completed_ns = now_ns();
    snapshot.rknn_frame_id = output.rknn_frame_id;
    snapshot.worker_index = worker_index;
    snapshot.core_mask = core_mask;
    snapshot.status = static_cast<std::uint32_t>(output.status);
    snapshot.instance_count =
        static_cast<std::uint32_t>(std::min<std::size_t>(3, output.instances.size()));
    snapshot.width = output.mask_width;
    snapshot.height = output.mask_height;
    snapshot.mask_bytes = static_cast<std::uint32_t>(output.packed_mask.size());
    snapshot.preprocess_ms = output.preprocess_ms;
    snapshot.input_sync_ms = output.input_sync_ms;
    snapshot.inference_ms = output.inference_ms;
    snapshot.output_sync_ms = output.output_sync_ms;
    snapshot.postprocess_ms = output.postprocess_ms;
    snapshot.total_ms = output.total_ms;
    snapshot.claimed_count = counters_.claimed.load(std::memory_order_relaxed);
    snapshot.completed_count =
        counters_.completed.load(std::memory_order_relaxed);
    snapshot.overwritten_count =
        counters_.overwritten.load(std::memory_order_relaxed);
    snapshot.out_of_order_count =
        counters_.out_of_order.load(std::memory_order_relaxed);
    snapshot.error_count = counters_.errors.load(std::memory_order_relaxed);
    snapshot.notification_count =
        counters_.notifications.load(std::memory_order_relaxed);
    snapshot.notification_error_count =
        counters_.notification_errors.load(std::memory_order_relaxed);
    for (std::size_t index = 0; index < snapshot.instance_count; ++index) {
      snapshot.instances[index] = output.instances[index];
    }
    std::memcpy(slot_header, &snapshot, sizeof(snapshot));
    if (!output.packed_mask.empty()) {
      std::memcpy(slot_bytes + xsmart::kResultSlotHeaderSize,
                  output.packed_mask.data(), output.packed_mask.size());
    }
    slot_header->publish_ms = elapsed_ms(publish_started, Clock::now());
    atomic_store(&slot_header->sequence, sequence + 1U);
    const std::uint64_t publish_sequence =
        atomic_load(&header_->publish_sequence) + 1U;
    atomic_store(&header_->published_slot, slot);
    atomic_store(&header_->publish_sequence, publish_sequence);
    latest_frame_id_ = job.frame_id;
    const std::uint64_t notification = 1;
    const ssize_t written =
        ::write(result_event_fd_, &notification, sizeof(notification));
    if (written == static_cast<ssize_t>(sizeof(notification))) {
      counters_.notifications.fetch_add(1, std::memory_order_relaxed);
    } else if (written < 0 && errno == EAGAIN) {
      counters_.notifications.fetch_add(1, std::memory_order_relaxed);
    } else {
      counters_.notification_errors.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
  }

 private:
  SharedMapping& mapping_;
  Counters& counters_;
  GlobalHeader* header_;
  std::mutex mutex_;
  std::uint64_t latest_frame_id_ = 0;
  int result_event_fd_ = -1;
};

class Worker {
 public:
  Worker(const std::uint32_t index, const std::uint32_t core_mask,
         const int cpu_core,
         std::unique_ptr<RknnWorkerRuntime> runtime, ResultPublisher& publisher,
         Counters& counters)
      : index_(index), core_mask_(core_mask), runtime_(std::move(runtime)),
        publisher_(publisher), counters_(counters),
        thread_(&Worker::loop, this) {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpu_core, &affinity);
    const int status = pthread_setaffinity_np(
        thread_.native_handle(), sizeof(affinity), &affinity);
    if (status != 0) {
      std::cerr << "[lane-rknn] warning: failed to bind worker " << index_
                << " to CPU " << cpu_core << ": " << std::strerror(status)
                << '\n';
    }
  }

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  ~Worker() { stop(); }

  bool is_idle() const { return !busy_.load(std::memory_order_acquire); }

  bool assign(FrameJob job) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (busy_.load(std::memory_order_relaxed) || job_.has_value()) {
      return false;
    }
    busy_.store(true, std::memory_order_release);
    job_ = std::move(job);
    condition_.notify_one();
    return true;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      condition_.notify_one();
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  void loop() {
    while (true) {
      std::optional<FrameJob> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return stopping_ || job_.has_value(); });
        if (stopping_ && !job_.has_value()) {
          return;
        }
        job = std::move(job_);
        job_.reset();
      }
      try {
        InferenceOutput output = runtime_->run(*job);
        if (output.status == ResultStatus::kInferenceError ||
            output.status == ResultStatus::kPreprocessError ||
            output.status == ResultStatus::kPostprocessError) {
          counters_.errors.fetch_add(1, std::memory_order_relaxed);
        }
        counters_.completed.fetch_add(1, std::memory_order_relaxed);
        publisher_.publish(*job, index_, core_mask_, output);
      } catch (const std::exception& error) {
        counters_.errors.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "[lane-rknn] worker " << index_ << " failed: "
                  << error.what() << '\n';
      }
      busy_.store(false, std::memory_order_release);
    }
  }

  std::uint32_t index_;
  std::uint32_t core_mask_;
  std::unique_ptr<RknnWorkerRuntime> runtime_;
  ResultPublisher& publisher_;
  Counters& counters_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<FrameJob> job_;
  bool stopping_ = false;
  std::atomic<bool> busy_{false};
  std::thread thread_;
};

rknn_core_mask to_core_mask(const std::uint32_t value) {
  switch (value) {
    case 1:
      return RKNN_NPU_CORE_0;
    case 2:
      return RKNN_NPU_CORE_1;
    case 4:
      return RKNN_NPU_CORE_2;
    default:
      throw std::invalid_argument("only individual RK3588 NPU core masks are supported");
  }
}

int run_backend(const Options& options) {
  SharedMapping input_mapping(options.input_shm, false);
  SharedMapping result_mapping(options.result_shm, true);
  InputReader input_reader(input_mapping);
  Counters counters;
  ResultPublisher publisher(result_mapping, counters, options.result_event_fd);

  rknn_context contexts[2] = {0, 0};
  std::unique_ptr<Worker> workers[2];
  try {
    int status = rknn_init(&contexts[0],
                           const_cast<char*>(options.model_path.c_str()), 0, 0,
                           nullptr);
    if (status != RKNN_SUCC) {
      throw std::runtime_error("rknn_init failed: " + std::to_string(status));
    }
    status = rknn_dup_context(&contexts[0], &contexts[1]);
    if (status != RKNN_SUCC) {
      throw std::runtime_error("rknn_dup_context failed: " +
                               std::to_string(status));
    }
    for (std::size_t index = 0; index < 2; ++index) {
      status =
          rknn_set_core_mask(contexts[index], to_core_mask(options.core_masks[index]));
      if (status != RKNN_SUCC) {
        throw std::runtime_error("rknn_set_core_mask failed: " +
                                 std::to_string(status));
      }
    }
    for (std::size_t index = 0; index < 2; ++index) {
      auto runtime = std::make_unique<RknnWorkerRuntime>(
          contexts[index], options.preprocess, options.output_mode,
          options.score_threshold,
          options.nms_threshold, options.mask_threshold, options.max_instances);
      workers[index] = std::make_unique<Worker>(
          static_cast<std::uint32_t>(index), options.core_masks[index],
          options.cpu_cores[index],
          std::move(runtime), publisher, counters);
    }
    publisher.set_state(BackendState::kReady);
    std::cerr << "[lane-rknn] ready: cores=" << options.core_masks[0] << ','
              << options.core_masks[1] << " cpu_cores=" << options.cpu_cores[0]
              << ',' << options.cpu_cores[1]
              << " preprocess=" << options.preprocess
              << " output_mode=" << options.output_mode
              << '\n';

    std::uint64_t last_claimed_publish_sequence = 0;
    std::size_t next_worker = 0;
    while (!g_stop.load(std::memory_order_relaxed)) {
      const std::uint64_t available_sequence = input_reader.published_sequence();
      if (available_sequence <= last_claimed_publish_sequence) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        continue;
      }
      std::optional<std::size_t> idle_worker;
      for (std::size_t offset = 0; offset < 2; ++offset) {
        const std::size_t index = (next_worker + offset) % 2;
        if (workers[index]->is_idle()) {
          idle_worker = index;
          break;
        }
      }
      if (!idle_worker.has_value()) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        continue;
      }
      FrameJob job;
      if (!input_reader.read_latest(&job) ||
          job.publish_sequence <= last_claimed_publish_sequence) {
        std::this_thread::yield();
        continue;
      }
      if (job.publish_sequence > last_claimed_publish_sequence + 1U) {
        counters.overwritten.fetch_add(
            job.publish_sequence - last_claimed_publish_sequence - 1U,
            std::memory_order_relaxed);
      }
      if (workers[*idle_worker]->assign(std::move(job))) {
        last_claimed_publish_sequence = available_sequence;
        counters.claimed.fetch_add(1, std::memory_order_relaxed);
        next_worker = (*idle_worker + 1U) % 2U;
      }
    }
    workers[0]->stop();
    workers[1]->stop();
    workers[0].reset();
    workers[1].reset();
    rknn_destroy(contexts[1]);
    rknn_destroy(contexts[0]);
    return 0;
  } catch (...) {
    publisher.set_state(BackendState::kError, 1);
    for (auto& worker : workers) {
      if (worker) {
        worker->stop();
        worker.reset();
      }
    }
    if (contexts[1] != 0) {
      rknn_destroy(contexts[1]);
    }
    if (contexts[0] != 0) {
      rknn_destroy(contexts[0]);
    }
    throw;
  }
}

}  // namespace

int main(const int argc, char** argv) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  try {
    return run_backend(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "[lane-rknn] fatal: " << error.what() << '\n';
    return 1;
  }
}
