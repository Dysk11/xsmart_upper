#include "rknn_model.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace xsmart {

namespace {

std::size_t tensor_type_size(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32:
    case RKNN_TENSOR_INT32:
    case RKNN_TENSOR_UINT32:
      return 4;
    case RKNN_TENSOR_FLOAT16:
    case RKNN_TENSOR_INT16:
    case RKNN_TENSOR_UINT16:
    case RKNN_TENSOR_BFLOAT16:
      return 2;
    case RKNN_TENSOR_INT64:
      return 8;
    default:
      return 1;
  }
}

std::vector<uint32_t> tensor_dims(const rknn_tensor_attr& attr) {
  return std::vector<uint32_t>(attr.dims, attr.dims + attr.n_dims);
}

}  // namespace

RknnModel::~RknnModel() { close(); }

void RknnModel::check(int code, const std::string& operation) {
  if (code == RKNN_SUCC) {
    return;
  }
  std::ostringstream message;
  message << operation << " failed, ret=" << code;
  throw std::runtime_error(message.str());
}

void RknnModel::open(const std::string& model_path, rknn_core_mask core_mask,
                     rknn_tensor_type input_type, rknn_tensor_format input_format) {
  close();
  model_path_ = model_path;
  try {
    check(rknn_init(&context_, const_cast<char*>(model_path.c_str()), 0, 0, nullptr),
          "rknn_init(" + model_path + ")");
    check(rknn_set_core_mask(context_, core_mask), "rknn_set_core_mask");

    rknn_input_output_num counts{};
    check(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts)),
          "rknn_query(RKNN_QUERY_IN_OUT_NUM)");
    if (counts.n_input != 1 || counts.n_output == 0) {
      throw std::runtime_error("native runtime requires one input and at least one output");
    }

    input_attr_ = {};
    input_attr_.index = 0;
    check(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_)),
          "rknn_query(RKNN_QUERY_INPUT_ATTR)");

    output_attrs_.resize(counts.n_output);
    output_buffers_.resize(counts.n_output);
    outputs_.resize(counts.n_output);
    for (uint32_t index = 0; index < counts.n_output; ++index) {
      auto& attr = output_attrs_[index];
      attr = {};
      attr.index = index;
      check(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)),
            "rknn_query(RKNN_QUERY_OUTPUT_ATTR)");
      output_buffers_[index].resize(attr.n_elems);
      auto& output = outputs_[index];
      output = {};
      output.index = index;
      output.want_float = 1;
      output.is_prealloc = 1;
      output.buf = output_buffers_[index].data();
      output.size = static_cast<uint32_t>(output_buffers_[index].size() * sizeof(float));
    }

    input_binding_attr_ = input_attr_;
    input_binding_attr_.type = input_type;
    input_binding_attr_.fmt = input_format;
    input_binding_attr_.pass_through = 0;
    input_binding_attr_.size = static_cast<uint32_t>(
        static_cast<std::size_t>(input_attr_.n_elems) * tensor_type_size(input_type));
    input_binding_attr_.size_with_stride = input_binding_attr_.size;
    input_binding_attr_.w_stride = 0;
    input_binding_attr_.h_stride = 0;
    input_mem_ = rknn_create_mem(context_, input_binding_attr_.size);
    if (input_mem_ == nullptr || input_mem_->virt_addr == nullptr) {
      throw std::runtime_error("rknn_create_mem(input) failed");
    }
    check(rknn_set_io_mem(context_, input_mem_, &input_binding_attr_),
          "rknn_set_io_mem(input)");
  } catch (...) {
    close();
    throw;
  }
}

std::vector<TensorView> RknnModel::run(const void* input, std::size_t bytes) {
  if (!is_open() || input_mem_ == nullptr) {
    throw std::runtime_error("RKNN model is not open");
  }
  if (input == nullptr || bytes != input_binding_attr_.size) {
    std::ostringstream message;
    message << "input byte count mismatch for " << model_path_ << ": got " << bytes
            << ", expected " << input_binding_attr_.size;
    throw std::runtime_error(message.str());
  }

  std::memcpy(input_mem_->virt_addr, input, bytes);
  check(rknn_mem_sync(context_, input_mem_, RKNN_MEMORY_SYNC_TO_DEVICE),
        "rknn_mem_sync(input)");
  check(rknn_run(context_, nullptr), "rknn_run");
  check(rknn_outputs_get(context_, static_cast<uint32_t>(outputs_.size()), outputs_.data(),
                         nullptr),
        "rknn_outputs_get");

  std::vector<TensorView> views;
  views.reserve(outputs_.size());
  for (std::size_t index = 0; index < outputs_.size(); ++index) {
    views.push_back(
        TensorView{output_buffers_[index].data(), output_buffers_[index].size(),
                   tensor_dims(output_attrs_[index])});
  }
  check(rknn_outputs_release(context_, static_cast<uint32_t>(outputs_.size()), outputs_.data()),
        "rknn_outputs_release");
  return views;
}

void RknnModel::close() noexcept {
  if (input_mem_ != nullptr && context_ != 0) {
    rknn_destroy_mem(context_, input_mem_);
  }
  input_mem_ = nullptr;
  if (context_ != 0) {
    rknn_destroy(context_);
  }
  context_ = 0;
  input_attr_ = {};
  input_binding_attr_ = {};
  output_attrs_.clear();
  output_buffers_.clear();
  outputs_.clear();
  model_path_.clear();
}

}  // namespace xsmart
