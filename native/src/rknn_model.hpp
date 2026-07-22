#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rknn_api.h"

namespace xsmart {

struct TensorView {
  const float* data = nullptr;
  std::size_t size = 0;
  std::vector<uint32_t> dims;
};

class RknnModel {
 public:
  RknnModel() = default;
  RknnModel(const RknnModel&) = delete;
  RknnModel& operator=(const RknnModel&) = delete;
  ~RknnModel();

  void open(const std::string& model_path, rknn_core_mask core_mask,
            rknn_tensor_type input_type, rknn_tensor_format input_format);
  std::vector<TensorView> run(const void* input, std::size_t bytes);
  void close() noexcept;

  const rknn_tensor_attr& input_attr() const { return input_attr_; }
  const std::vector<rknn_tensor_attr>& output_attrs() const { return output_attrs_; }
  const std::string& model_path() const { return model_path_; }
  bool is_open() const { return context_ != 0; }

 private:
  static void check(int code, const std::string& operation);

  rknn_context context_ = 0;
  rknn_tensor_mem* input_mem_ = nullptr;
  rknn_tensor_attr input_attr_{};
  rknn_tensor_attr input_binding_attr_{};
  std::vector<rknn_tensor_attr> output_attrs_;
  std::vector<std::vector<float>> output_buffers_;
  std::vector<rknn_output> outputs_;
  std::string model_path_;
};

}  // namespace xsmart
