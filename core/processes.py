"""Multiprocessing workers for inference and UI rendering."""

from __future__ import annotations

from typing import Any, Dict

import queue

from core.rknn_object_detector import RknnObjectDetector
from core.visualizer import Visualizer


def run_inference_worker(
    detector_config: Dict[str, Any],
    input_queue: "queue.Queue[Dict[str, Any]]",
    output_queue: "queue.Queue[Dict[str, Any]]",
    stop_event: Any,
) -> None:
    """Run AI inference in a dedicated process."""

    detector = RknnObjectDetector(detector_config)
    try:
        while not stop_event.is_set():
            try:
                item = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break

            frame = item.get("frame")
            frame_id = item.get("frame_id")
            if frame is None:
                continue

            detections = detector.detect(frame)
            _offer_queue(output_queue, {"frame_id": frame_id, "detected_objects": detections})
    finally:
        detector.close()


def run_ui_worker(
    visualizer_config: Dict[str, Any],
    input_queue: "queue.Queue[Dict[str, Any]]",
    stop_event: Any,
) -> None:
    """Run UI rendering in a dedicated process."""

    visualizer = Visualizer(visualizer_config)
    try:
        while not stop_event.is_set():
            try:
                payload = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None:
                break

            should_continue = visualizer.render(**payload)
            if not should_continue:
                stop_event.set()
                break
    finally:
        visualizer.close()


def _offer_queue(target_queue: "queue.Queue[Dict[str, Any]]", payload: Dict[str, Any]) -> None:
    try:
        target_queue.put_nowait(payload)
        return
    except queue.Full:
        pass

    try:
        _ = target_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        target_queue.put_nowait(payload)
    except Exception:
        pass
