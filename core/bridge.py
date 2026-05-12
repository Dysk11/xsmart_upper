"""上下位机桥接层。"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None

from core.protocol import build_packet


class BaseVehicleBridge(ABC):
    """定义上位机向车辆发送高层指令的统一接口。"""

    @abstractmethod
    def connect(self) -> None:
        """建立通信连接。

        输入:
            无。

        输出:
            无返回值；连接失败时可抛出异常。
        """

    @abstractmethod
    def send(self, payload: Mapping[str, Any]) -> None:
        """发送一帧高层控制负载。

        输入:
            payload: 已经整理好的协议字段字典。

        输出:
            无返回值，内部完成发送动作。
        """

    @abstractmethod
    def close(self) -> None:
        """关闭通信连接并释放资源。

        输入:
            无。

        输出:
            无返回值。
        """


class MockBridge(BaseVehicleBridge):
    """本地调试桥接层，仅打印协议内容。"""

    def connect(self) -> None:
        """初始化 Mock 模式。

        输入:
            无。

        输出:
            无返回值。
        """

        print("MockBridge 已启用，当前只打印控制指令，不发送真实串口数据。")

    def send(self, payload: Mapping[str, Any]) -> None:
        """将协议内容打印到终端，方便调试。

        输入:
            payload: 已经整理好的协议字段字典。

        输出:
            无返回值。
        """

        # Mock 模式不接触真实硬件，只把将要发送的数据打印出来。
        packet_bytes = build_packet(payload)
        packet_hex = " ".join(f"{b:02X}" for b in packet_bytes)
        print(f"[MockBridge] {packet_hex} | payload: {payload}")

    def close(self) -> None:
        """关闭 Mock 桥接层。

        输入:
            无。

        输出:
            无返回值。
        """


class SerialBridge(BaseVehicleBridge):
    """基于 pyserial 的串口桥接层。"""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """保存串口参数并准备延迟连接。

        输入:
            config: 串口配置字典，包含端口号、波特率、超时等参数。

        输出:
            无返回值。
        """

        self.port = str(config.get("port", "/dev/ttyS4"))
        self.baudrate = int(config.get("baudrate", 115200))
        self.timeout = float(config.get("timeout", 0.02))
        self.write_timeout = float(config.get("write_timeout", self.timeout))
        self.reconnect_interval_sec = float(config.get("reconnect_interval_sec", 0.5))
        self.max_reconnect_attempts = int(config.get("max_reconnect_attempts", 3))
        self.serial_port: Optional[serial.Serial] = None

    def connect(self) -> None:
        """打开串口并建立连接。

        输入:
            无。

        输出:
            无返回值；打开失败时抛出异常。
        """

        if self.serial_port is not None and self.serial_port.is_open:
            return

        if serial is None:
            raise ImportError(
                "当前环境未安装 pyserial，无法使用 SerialBridge。"
                "如果只想本地调试，请把 bridge.type 设为 mock；"
                "如果要连串口，请先执行 pip install pyserial。"
            )

        self.serial_port = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            write_timeout=self.write_timeout,
        )
        print(f"SerialBridge 已连接串口: {self.port} @ {self.baudrate}")

    def send(self, payload: Mapping[str, Any]) -> None:
        """将控制负载通过串口发送给下位机。

        输入:
            payload: 已经整理好的协议字段字典。

        输出:
            无返回值；若发送失败则尝试重连后重发。
        """

        if self.serial_port is None or not self.serial_port.is_open:
            self.connect()

        # 真正的协议打包动作统一放在 protocol.py，避免格式散落在各处。
        packet = build_packet(payload)
        assert self.serial_port is not None

        try:
            self.serial_port.write(packet)
            self.serial_port.flush()
        except Exception as error:
            if serial is None:
                raise error
            if not self._attempt_reconnect():
                raise
            assert self.serial_port is not None
            self.serial_port.write(packet)
            self.serial_port.flush()

    def close(self) -> None:
        """关闭当前串口连接。

        输入:
            无。

        输出:
            无返回值。
        """

        if self.serial_port is not None:
            if self.serial_port.is_open:
                self.serial_port.close()
            self.serial_port = None

    def _attempt_reconnect(self) -> bool:
        """在串口发送异常后尝试自动重连。

        输入:
            无。

        输出:
            重连成功返回 True，失败返回 False。
        """

        self.close()
        for _ in range(self.max_reconnect_attempts):
            try:
                time.sleep(self.reconnect_interval_sec)
                self.connect()
                return True
            except Exception:
                continue
        return False


def build_vehicle_bridge(config: Mapping[str, Any]) -> BaseVehicleBridge:
    """根据配置构建对应的桥接层实例。

    输入:
        config: bridge 节点配置字典。

    输出:
        返回 BaseVehicleBridge 的具体实现对象。
    """

    bridge_type = str(config.get("type", "mock")).lower()
    if bridge_type == "serial":
        return SerialBridge(config.get("serial", {}))
    return MockBridge()
