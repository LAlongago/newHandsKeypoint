import time
import numpy as np


finger_map = {"wrist": 0,
              "thumb_cmc": 1,
              "thumb_mcp": 2,
              "thumb_ip": 3,
              "thumb_tip": 4,
              "index_finger_mcp": 5,
              "index_finger_pip": 6,
              "index_finger_dip": 7,
              "index_finger_tip": 8,
              "middle_finger_mcp": 9,
              "middle_finger_pip": 10,
              "middle_finger_dip": 11,
              "middle_finger_tip": 12,
              "ring_finger_mcp": 13,
              "ring_finger_pip": 14,
              "ring_finger_dip": 15,
              "ring_finger_tip": 16,
              "pinky_mcp": 17,
              "pinky_pip": 18,
              "pinky_dip": 19,
              "pinky_tip": 20}


class GestureRecognition:
    def __init__(self):
        self.previous_finger_tip = None
        self.stationary_start_time = None

        self.previous_keypoints = None
        self.previous_frame_time = None

    def recognize(self, results, results_body, current_frame_time):
        gestures = []
        for result in results:
            result = result.cpu()
            if result.keypoints is None or result.keypoints.xy is None or result.keypoints.conf is None:
                continue
            keypoints = result.keypoints.xy[0]
            if self.is_index_finger_pointing(keypoints[finger_map["index_finger_tip"]]):
                if self.stationary_start_time is None:
                    self.stationary_start_time = time.time()
                elif time.time() - self.stationary_start_time > 3:
                    gestures.append("choose")
            else:
                self.stationary_start_time = None

        for result_b in results_body:
            if result_b is not None:
                keypoints = result_b.keypoints.xy[0]
                if keypoints is not None:
                    gestures.extend(self.detect_running(keypoints, current_frame_time))
                else:
                    continue  # keypoints 不存在，跳过当前循环
            else:
                continue  # result_b 为 None，跳过当前循环

        return gestures

    def is_index_finger_pointing(self, finger_tip):
        if self.previous_finger_tip is None:
            self.previous_finger_tip = finger_tip
            return False

        distance = np.linalg.norm(finger_tip - self.previous_finger_tip)
        print(distance)
        if distance > 50:
            self.previous_finger_tip = finger_tip
        return distance < 50  # 设定阈值为5像素

    def detect_running(self, keypoints, current_frame_time):
        gestures = []
        if self.previous_keypoints is None or self.previous_frame_time is None:
            # 初始化，这是处理视频的第一帧时的情况
            self.previous_keypoints = keypoints
            self.previous_frame_time = current_frame_time
            return gestures

        knee_indices = [13, 14]  # 左膝盖和右膝盖的索引
        velocity_threshold = 0.1  # 膝盖速度的阈值，根据实际情况调整
        acceleration_threshold = 0.5  # 膝盖加速度的阈值，根据实际情况调整

        # 计算时间间隔dt
        dt = current_frame_time - self.previous_frame_time

        # 确保时间间隔是有效的
        if dt <= 0:
            return gestures

        for i in knee_indices:
            current_knee = keypoints[i]
            previous_knee = self.previous_keypoints[i]

            # 计算速度
            velocity = np.linalg.norm(current_knee - previous_knee) / dt

            # 如果速度超过阈值，检查是否在跑步
            if velocity > velocity_threshold:
                # 计算加速度
                acceleration = velocity  # 这里简化了加速度的计算，可根据需要调整
                # 检查加速度是否足够大以触发跑步动作
                if acceleration > acceleration_threshold:
                    gestures.append("running")
                    # 一次循环内只检测一次跑步动作
                    break

        # 更新前一帧的关键点和时间
        self.previous_keypoints = keypoints
        self.previous_frame_time = current_frame_time

        return gestures
