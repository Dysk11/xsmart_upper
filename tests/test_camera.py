import sys
import cv2
from pathlib import Path

# 将项目根目录加入 sys.path，以便能够导入 core 模块
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from core.camera import CameraReader

def main():
    # 测试配置
    config = {
        "mode": "camera",      # 模式: camera, video, stream
        "device_id": 0,        # 默认摄像头设备号
        "width": 640,          # 测试分辨率宽
        "height": 480,         # 测试分辨率高
        "fps": 30,             # 帧率
        "mirror": True         # 是否镜像翻转
    }

    print("正在初始化摄像头...")
    camera = CameraReader(config)

    try:
        camera.open()
        print("摄像头打开成功！按 'q' 键退出测试。")

        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("读取画面失败，可能摄像头连接断开。")
                break

            # 显示画面
            cv2.imshow("Camera Test", frame)

            # 等待按键，按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("测试结束。")
                break

    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        # 释放资源
        camera.release()
        cv2.destroyAllWindows()
        print("资源已释放。")

if __name__ == "__main__":
    main()
