from video_capture import VideoCapture
from pose_detection import PoseDetection
from pose_detection_body import PoseDetectionBody
from gesture_recognition import GestureRecognition
from ui_display import UIDisplay
import config
import time


def main():
    # 初始化各模块
    video_capture = VideoCapture(config.VIDEO_SOURCE)
    pose_detection = PoseDetection(config.HAND_MODEL_PATH)
    pose_detection_body = PoseDetectionBody(config.BODY_MODEL_PATH)
    gesture_recognition = GestureRecognition()
    ui_display = UIDisplay(video_capture.get_fps(), video_capture.get_frame_height(), video_capture.get_frame_width())

    while True:
        # 获取一帧图像
        frame = video_capture.get_frame()
        if frame is None:
            break

        current_frame_time = time.time()  # 当前帧的时间

        # 进行关键点检测
        results = pose_detection.detect(frame)
        results_body = pose_detection_body.detect(frame)

        # 进行手势识别
        gestures = gesture_recognition.recognize(results, results_body)
        # 显示结果
        ui_display.show(frame, results, results_body, gestures)

        # 退出条件
        if ui_display.check_exit():
            break

    video_capture.release()
    ui_display.close()


if __name__ == "__main__":
    main()
