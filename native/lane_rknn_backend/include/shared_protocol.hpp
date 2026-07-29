#pragma once

#include <cstddef>
#include <cstdint>

namespace xsmart {

constexpr std::uint32_t kProtocolVersion = 1;
constexpr std::size_t kGlobalHeaderSize = 64;
constexpr std::size_t kInputSlotHeaderSize = 64;
constexpr std::size_t kResultSlotHeaderSize = 256;
constexpr std::size_t kSlotCount = 2;

constexpr char kInputMagic[8] = {'X', 'S', 'L', 'N', 'I', 'N', '1', '\0'};
constexpr char kResultMagic[8] = {'X', 'S', 'L', 'N', 'O', 'T', '1', '\0'};

enum class BackendState : std::uint32_t {
  kInitializing = 0,
  kReady = 1,
  kError = 2,
};

enum class ResultStatus : std::uint32_t {
  kOk = 0,
  kNoDetection = 1,
  kPreprocessError = 2,
  kInferenceError = 3,
  kPostprocessError = 4,
  kInvalidFrame = 5,
};

#pragma pack(push, 1)

struct GlobalHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_size;
  std::uint32_t slot_size;
  std::uint32_t payload_capacity;
  std::uint64_t publish_sequence;
  std::uint32_t published_slot;
  std::uint32_t state;
  std::uint32_t error_code;
  std::uint32_t writer_pid;
  std::uint8_t reserved[16];
};

struct InputSlotHeader {
  std::uint64_t sequence;
  std::uint64_t frame_id;
  std::uint64_t source_frame_id;
  std::uint64_t captured_ns;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
  std::uint32_t row_stride;
  std::uint32_t payload_bytes;
  std::uint8_t reserved[12];
};

struct ResultInstance {
  std::int32_t x1;
  std::int32_t y1;
  std::int32_t x2;
  std::int32_t y2;
  float confidence;
};

struct ResultSlotHeader {
  std::uint64_t sequence;
  std::uint64_t frame_id;
  std::uint64_t source_frame_id;
  std::uint64_t captured_ns;
  std::uint64_t completed_ns;
  std::uint64_t rknn_frame_id;
  std::uint32_t worker_index;
  std::uint32_t core_mask;
  std::uint32_t status;
  std::uint32_t instance_count;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t mask_bytes;
  double preprocess_ms;
  double input_sync_ms;
  double inference_ms;
  double output_sync_ms;
  double postprocess_ms;
  double total_ms;
  double publish_ms;
  std::uint64_t claimed_count;
  std::uint64_t completed_count;
  std::uint64_t overwritten_count;
  std::uint64_t out_of_order_count;
  std::uint64_t error_count;
  ResultInstance instances[3];
  std::uint8_t reserved[24];
};

#pragma pack(pop)

static_assert(sizeof(GlobalHeader) == kGlobalHeaderSize);
static_assert(sizeof(InputSlotHeader) == kInputSlotHeaderSize);
static_assert(sizeof(ResultSlotHeader) == kResultSlotHeaderSize);

}  // namespace xsmart
