"""上下位机通信协议打包与解析模块。"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping


PROTOCOL_DESCRIPTION = """
固定 7 字节二进制协议:
1. 帧头为 AA 55。
2. 后接横向误差 Int16、转向角 Int16，均为大端序。
3. 最后一个字节的低 2 位为速度状态：0=停止、1=低速、2=中速、3=高速；
   bit2..bit7 保留且固定为 0。
""".strip()

PACKET_FIELDS = [
    "ts_ms",
    "mode",
    "target_speed",
    "speed_state",
    "steer_deg",
    "lateral_error_px",
    "heading_error_deg",
    "curvature",
    "confidence",
    "is_lane_lost",
]


def target_speed_to_speed_state(target_speed: float) -> int:
    """Convert the continuous planner speed to the 2-bit vehicle speed state."""

    speed = float(target_speed)
    if not math.isfinite(speed) or speed <= 0.0:
        return 0x00
    if speed <= 1.0:
        return 0x01
    if speed <= 2.0:
        return 0x02
    return 0x03


def validate_drive_speed_state(value: Any) -> int:
    """Validate the configured non-stop speed state sent to the vehicle."""

    state = int(value)
    if state not in (0x01, 0x02, 0x03):
        raise ValueError("bridge.drive_speed_state must be 1, 2, or 3")
    return state


def resolve_configured_speed_state(target_speed: float, drive_speed_state: Any) -> int:
    """Return stop for non-positive targets, otherwise the configured drive state."""

    speed = float(target_speed)
    if not math.isfinite(speed) or speed <= 0.0:
        return 0x00
    return validate_drive_speed_state(drive_speed_state)


def normalize_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """对通信负载做默认值补全与类型归一化。

    输入:
        payload: 上位机待发送的字段字典。

    输出:
        返回归一化后的字段字典，字段名与 PACKET_FIELDS 保持一致。
    """

    target_speed = float(payload.get("target_speed", 0.0))
    speed_state = (
        int(payload["speed_state"]) & 0x03
        if "speed_state" in payload
        else target_speed_to_speed_state(target_speed)
    )
    return {
        "ts_ms": int(payload.get("ts_ms", 0)),
        "mode": str(payload.get("mode", "NORMAL")),
        "target_speed": target_speed,
        "speed_state": speed_state,
        "steer_deg": float(payload.get("steer_deg", 0.0)),
        "lateral_error_px": float(payload.get("lateral_error_px", 0.0)),
        "heading_error_deg": float(payload.get("heading_error_deg", 0.0)),
        "curvature": float(payload.get("curvature", 0.0)),
        "confidence": float(payload.get("confidence", 0.0)),
        "is_lane_lost": int(bool(payload.get("is_lane_lost", False))),
    }


def build_packet(payload: Mapping[str, Any]) -> bytes:
    """将字段字典打包成二进制通信协议。

    输入:
        payload: 包含控制量与巡线状态的字段字典。

    输出:
        返回可直接通过串口发送的字节串。
    """

    normalized = normalize_payload(payload)
    
    error = int(normalized['lateral_error_px'])
    angle = int(normalized['steer_deg'])
    
    # 限制在16位有符号整数范围内
    error = max(-32768, min(32767, error))
    angle = max(-32768, min(32767, angle))

    data = bytearray([
        0xAA,
        0x55,
        (error >> 8) & 0xFF,
        error & 0xFF,
        (angle >> 8) & 0xFF,
        angle & 0xFF,
        normalized["speed_state"] & 0x03,
    ])
    
    return bytes(data)


def parse_packet(packet: bytes | str) -> Dict[str, Any]:
    """将二进制协议解析回字段字典，方便本地调试和联调。

    输入:
        packet: 字节串或字符串形式的一帧协议内容。

    输出:
        返回解析后的字段字典。
    """

    result: Dict[str, Any] = {}
    if not isinstance(packet, bytes) or len(packet) != 7:
        return result

    if packet[0] == 0xAA and packet[1] == 0x55:
        error = (packet[2] << 8) | packet[3]
        if error >= 32768:
            error -= 65536
            
        angle = (packet[4] << 8) | packet[5]
        if angle >= 32768:
            angle -= 65536
            
        result["lateral_error_px"] = float(error)
        result["steer_deg"] = float(angle)
        result["speed_state"] = int(packet[6] & 0x03)

    return result
