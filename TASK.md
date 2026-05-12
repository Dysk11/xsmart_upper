v在树莓派上，使用CH340串口usb转ttl模块连接Arduino，Arduino负责控制小车的运动，而树莓派负责处理摄像头数据并发送控制指令给Arduino。

在bridge.py和protocol.py中，修改uart通信结构，在_main中调用它。UART使用115200波特率，8数据位，无校验位，1停止位（8N1）。每条指令由2个帧头（0xAA和0x55）开头，后面跟着2字节的误差值和2字节的转向角度值，浮点数强制转换成整型传输。

UART通信协议设计：115200 8N1 2个帧头 + 2字节误差值 + 2字节转向角度值

示例代码如下

""
import serial

ser = serial.Serial("COM5",115200)

error = 16
angle = 300

data = bytearray([
    0xAA,
    0x55,
    (error>>8)&0xFF,
    error&0xFF,
    (angle>>8)&0xFF,
    angle&0xFF
])

ser.write(data)
""