"""上下位机通信协议打包与解析模块。"""

from __future__ import annotations

from typing import Any, Dict, Mapping


PROTOCOL_DESCRIPTION = """
固定 7 字节二进制协议:
1. 帧头为 AA 55。
2. 后接横向误差 Int16、转向角 Int16，均为大端序。
3. 最后一个字节为运行状态，bit0=0 表示停车，bit0=1 表示允许运行；
   bit1..bit7 保留且固定为 0。
""".strip()

PACKET_FIELDS = [
    "ts_ms",
    "mode",
    "target_speed",
    "motion_flag",
    "steer_deg",
    "lateral_error_px",
    "heading_error_deg",
    "curvature",
    "confidence",
    "is_lane_lost",
]


def normalize_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """对通信负载做默认值补全与类型归一化。

    输入:
        payload: 上位机待发送的字段字典。

    输出:
        返回归一化后的字段字典，字段名与 PACKET_FIELDS 保持一致。
    """

    target_speed = float(payload.get("target_speed", 0.0))
    motion_flag = (
        int(payload["motion_flag"]) & 0x01
        if "motion_flag" in payload
        else int(target_speed > 0.0)
    )
    return {
        "ts_ms": int(payload.get("ts_ms", 0)),
        "mode": str(payload.get("mode", "NORMAL")),
        "target_speed": target_speed,
        "motion_flag": motion_flag,
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
        normalized["motion_flag"] & 0x01,
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
        result["motion_flag"] = int(packet[6] & 0x01)

    return result
