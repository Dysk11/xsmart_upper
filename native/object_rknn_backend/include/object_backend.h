#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define XSMART_OBJECT_ABI_VERSION 1U
#define XSMART_OBJECT_MAX_DETECTIONS 64U
#define XSMART_OBJECT_NPU_CORE_2_MASK 4U

typedef struct xsmart_object_config {
  uint32_t abi_version;
  uint32_t struct_size;
  uint32_t class_count;
  uint32_t max_detections;
  uint32_t pipeline_depth;
  uint32_t core_mask;
  float score_threshold;
  float nms_threshold;
  uint32_t class_agnostic_nms;
  uint32_t reserved[7];
} xsmart_object_config;

typedef struct xsmart_object_detection {
  int32_t class_id;
  float confidence;
  int32_t x1;
  int32_t y1;
  int32_t x2;
  int32_t y2;
} xsmart_object_detection;

typedef struct xsmart_object_result {
  uint32_t abi_version;
  uint32_t struct_size;
  uint64_t frame_id;
  uint32_t context_index;
  uint32_t core_mask;
  int32_t status_code;
  uint32_t detection_count;
  float preprocess_ms;
  float input_sync_ms;
  float inference_ms;
  float npu_run_ms;
  float output_sync_ms;
  float postprocess_ms;
  float total_ms;
  xsmart_object_detection detections[XSMART_OBJECT_MAX_DETECTIONS];
} xsmart_object_result;

typedef void* xsmart_object_handle;

int xsmart_object_create(const char* model_path,
                         const xsmart_object_config* config,
                         xsmart_object_handle* output,
                         char* error_message,
                         uint32_t error_capacity);

int xsmart_object_detect(xsmart_object_handle handle,
                         const uint8_t* rgb,
                         uint32_t width,
                         uint32_t height,
                         uint32_t row_stride,
                         uint64_t frame_id,
                         xsmart_object_result* output,
                         char* error_message,
                         uint32_t error_capacity);

void xsmart_object_destroy(xsmart_object_handle handle);

const char* xsmart_object_version(void);

#if defined(__cplusplus)
}
#endif
