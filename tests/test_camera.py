import sys
import cv2
from pathlib import Path
from datetime import datetime

# 将项目根目录加入 sys.path，以便能够导入 core 模块
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from core.camera import CameraReader

def main():
    # 测试配置
    config = {
        "mode": "shared_memory",  # 模式: camera, video, shared_memory
        "device_id": 0,        # 默认摄像头设备号
        "width": 640,          # 测试分辨率宽
        "height": 480,         # 测试分辨率高
        "fps": 30,             # 帧率
        "mirror": False,        # 是否镜像翻转
        "shared_memory_name": "shm_ar_video"
    }

    print("正在初始化摄像头...")
    camera = CameraReader(config)
    
    # 配置视频录制保存路径
    output_dir = project_root / "outputs" / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = output_dir / f"record_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None

    try:
        camera.open()
        print(f"摄像头打开成功！视频将保存至: {output_video_path}")
        print("按 'q' 键退出测试并停止录制。")

        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("读取画面失败，可能摄像头连接断开。")
                break
            
            # 动态初始化 VideoWriter (确保和实际读取的分辨率一致)
            if video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(str(output_video_path), fourcc, config["fps"], (w, h))
            
            # 将当前帧写入视频
            video_writer.write(frame)

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
        if video_writer is not None:
            video_writer.release()
        camera.release()
        cv2.destroyAllWindows()
        print("资源已释放，录制已保存。")

if __name__ == "__main__":
    main()
