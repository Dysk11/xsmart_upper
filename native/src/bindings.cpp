#include <cstring>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "native_engine.hpp"

namespace py = pybind11;

namespace xsmart {
namespace {

template <typename T>
T value_or(const py::dict& config, const char* key, T fallback) {
  py::str name(key);
  if (!config.contains(name)) {
    return fallback;
  }
  return py::cast<T>(config[name]);
}

EngineConfig parse_config(const py::dict& values) {
  EngineConfig config;
  config.camera.mode = value_or(values, "camera_mode", config.camera.mode);
  config.camera.device_id = value_or(values, "camera_device_id", config.camera.device_id);
  config.camera.video_path = value_or(values, "video_path", config.camera.video_path);
  config.camera.shared_memory_name =
      value_or(values, "shared_memory_name", config.camera.shared_memory_name);
  config.camera.loop_video = value_or(values, "loop_video", config.camera.loop_video);
  config.camera.mirror = value_or(values, "mirror", config.camera.mirror);
  config.camera.width = value_or(values, "camera_width", config.camera.width);
  config.camera.height = value_or(values, "camera_height", config.camera.height);
  config.camera.fps = value_or(values, "camera_fps", config.camera.fps);
  config.camera.reconnect_attempts =
      value_or(values, "reconnect_attempts", config.camera.reconnect_attempts);
  config.camera.reconnect_interval_sec =
      value_or(values, "reconnect_interval_sec", config.camera.reconnect_interval_sec);

  config.lane.model_path = value_or(values, "lane_model_path", std::string{});
  config.lane.input_width = value_or(values, "lane_input_width", config.lane.input_width);
  config.lane.input_height = value_or(values, "lane_input_height", config.lane.input_height);
  config.lane.score_threshold =
      value_or(values, "lane_score_threshold", config.lane.score_threshold);
  config.lane.nms_threshold = value_or(values, "lane_nms_threshold", config.lane.nms_threshold);
  config.lane.mask_threshold =
      value_or(values, "lane_mask_threshold", config.lane.mask_threshold);
  config.lane.max_instances = value_or(values, "lane_max_instances", config.lane.max_instances);

  config.object.model_path = value_or(values, "object_model_path", std::string{});
  config.object.input_width =
      value_or(values, "object_input_width", config.object.input_width);
  config.object.input_height =
      value_or(values, "object_input_height", config.object.input_height);
  config.object.score_threshold =
      value_or(values, "object_score_threshold", config.object.score_threshold);
  config.object.nms_threshold =
      value_or(values, "object_nms_threshold", config.object.nms_threshold);
  config.object.max_detections =
      value_or(values, "object_max_detections", config.object.max_detections);
  config.object.class_agnostic_nms =
      value_or(values, "object_class_agnostic_nms", config.object.class_agnostic_nms);
  config.object.class_names =
      value_or(values, "object_class_names", std::vector<std::string>{});

  config.ocr.det_model_path = value_or(values, "ocr_det_model_path", std::string{});
  config.ocr.rec_model_path = value_or(values, "ocr_rec_model_path", std::string{});
  config.ocr.character_dict_path = value_or(values, "ocr_character_dict_path", std::string{});
  config.ocr.det_width = value_or(values, "ocr_det_width", config.ocr.det_width);
  config.ocr.det_height = value_or(values, "ocr_det_height", config.ocr.det_height);
  config.ocr.rec_width = value_or(values, "ocr_rec_width", config.ocr.rec_width);
  config.ocr.rec_height = value_or(values, "ocr_rec_height", config.ocr.rec_height);
  config.ocr.det_threshold = value_or(values, "ocr_det_threshold", config.ocr.det_threshold);
  config.ocr.box_threshold = value_or(values, "ocr_box_threshold", config.ocr.box_threshold);
  config.ocr.unclip_ratio = value_or(values, "ocr_unclip_ratio", config.ocr.unclip_ratio);
  config.ocr.min_score = value_or(values, "ocr_min_score", config.ocr.min_score);
  config.ring_buffer_size = value_or(values, "ring_buffer_size", config.ring_buffer_size);
  return config;
}

py::dict timing_dict(const Timing& timing) {
  py::dict result;
  result["preprocess_ms"] = timing.preprocess_ms;
  result["inference_ms"] = timing.inference_ms;
  result["postprocess_ms"] = timing.postprocess_ms;
  result["total_ms"] = timing.total_ms;
  return result;
}

py::array_t<uint8_t> mat_u8(const cv::Mat& mat) {
  if (mat.channels() == 1) {
    py::array_t<uint8_t> output({mat.rows, mat.cols});
    std::memcpy(output.mutable_data(), mat.data, mat.total());
    return output;
  }
  py::array_t<uint8_t> output({mat.rows, mat.cols, mat.channels()});
  std::memcpy(output.mutable_data(), mat.data, mat.total() * mat.elemSize());
  return output;
}

py::dict frame_packet_dict(const FramePacket& packet) {
  py::dict result;
  result["ok"] = packet.ok;
  if (!packet.ok) {
    return result;
  }
  result["frame_id"] = packet.frame_id;
  result["captured_at"] = packet.captured_at;
  result["width"] = packet.width;
  result["height"] = packet.height;
  if (packet.bgr.empty()) {
    result["frame_bgr"] = py::none();
  } else {
    result["frame_bgr"] = mat_u8(packet.bgr);
  }

  py::dict lane;
  lane["mask"] = mat_u8(packet.lane.mask);
  lane["confidence"] = packet.lane.confidence;
  lane["status"] = packet.lane.status;
  lane["timing"] = timing_dict(packet.lane.timing);
  py::list instances;
  for (const auto& instance : packet.lane.instances) {
    py::dict item;
    item["bbox"] = instance.bbox;
    item["confidence"] = instance.confidence;
    instances.append(std::move(item));
  }
  lane["instances"] = std::move(instances);
  result["lane"] = std::move(lane);

  py::list detections;
  for (const auto& detection : packet.detections) {
    py::dict item;
    item["class_name"] = detection.class_name;
    item["confidence"] = detection.confidence;
    item["bbox"] = detection.bbox;
    detections.append(std::move(item));
  }
  result["detections"] = std::move(detections);
  result["object_timing"] = timing_dict(packet.object_timing);
  result["total_ms"] = packet.total_ms;
  return result;
}

py::dict ocr_result_dict(const OcrResult& result) {
  py::dict output;
  output["trigger_id"] = result.trigger_id;
  output["frame_id"] = result.frame_id;
  output["source_bbox"] = result.source_bbox;
  output["text"] = result.text;
  output["confidence"] = result.confidence;
  output["inference_ms"] = result.inference_ms;
  if (result.error.empty()) {
    output["error"] = py::none();
  } else {
    output["error"] = result.error;
  }
  py::list items;
  for (const auto& item : result.items) {
    py::dict encoded;
    encoded["text"] = item.text;
    encoded["score"] = item.score;
    encoded["box"] = item.box;
    items.append(std::move(encoded));
  }
  output["items"] = std::move(items);
  return output;
}

}  // namespace
}  // namespace xsmart

PYBIND11_MODULE(xsmart_rknn_native, module) {
  module.doc() = "RKNNRT C API perception runtime for RK3588";
  py::class_<xsmart::NativeEngine>(module, "NativePerception")
      .def(py::init([](const py::dict& config) {
        return std::make_unique<xsmart::NativeEngine>(xsmart::parse_config(config));
      }))
      .def("open", [](xsmart::NativeEngine& engine) {
        py::gil_scoped_release release;
        engine.open();
      })
      .def("next_frame", [](xsmart::NativeEngine& engine, bool want_bgr) {
        xsmart::FramePacket packet;
        {
          py::gil_scoped_release release;
          packet = engine.next_frame(want_bgr);
        }
        return xsmart::frame_packet_dict(packet);
      }, py::arg("want_bgr") = true)
      .def("submit_ocr", [](xsmart::NativeEngine& engine, uint64_t trigger_id,
                             uint64_t frame_id, const std::array<int, 4>& bbox) {
        py::gil_scoped_release release;
        return engine.submit_ocr(trigger_id, frame_id, bbox);
      })
      .def("poll_ocr", [](xsmart::NativeEngine& engine) -> py::object {
        std::optional<xsmart::OcrResult> result;
        {
          py::gil_scoped_release release;
          result = engine.poll_ocr();
        }
        if (result.has_value()) {
          return xsmart::ocr_result_dict(*result);
        }
        return py::none();
      })
      .def("close", [](xsmart::NativeEngine& engine) {
        py::gil_scoped_release release;
        engine.close();
      })
      .def_property_readonly("is_open", &xsmart::NativeEngine::is_open);
}
