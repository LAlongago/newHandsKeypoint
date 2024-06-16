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

body_map = {"nose": 0,
            "left_eye": 1,
            "right_eye": 2,
            "left_ear": 3,
            "right_ear": 4,
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
            "left_hip": 11,
            "right_hip": 12,
            "left_knee": 13,
            "right_knee": 14,
            "left_ankle": 15,
            "right_ankle": 16}


class GestureRecognition:
    def __init__(self):
        self.previous_finger_tip = None
        self.stationary_start_time = None

        self.previous_keypoints = None
        self.previous_frame_time = None

        self.sitting_start_time = None
        self.previous_wrist = None
        self.using_wrist = None

    def recognize(self, results, results_body, current_frame_time):
        gestures = []
        for result in results:
            result = result.cpu()
            if result.keypoints is None or result.keypoints.xy is None or result.keypoints.conf is None:
                continue
            keypoints = result.keypoints.xy[0]

            # if self.is_index_finger_pointing(keypoints[finger_map["index_finger_tip"]]):
            #     if self.stationary_start_time is None:
            #         self.stationary_start_time = time.time()
            #     elif time.time() - self.stationary_start_time > 3:
            #         gestures.append("choose")
            # else:
            #     self.stationary_start_time = None

        for result_b in results_body:
            result_b = result_b.cpu()
            if result_b.keypoints is None or result_b.keypoints.xy is None or result_b.keypoints.conf is None:
                continue
            keypoints_b = result_b.keypoints.xy[0]

            if self.is_sitting(keypoints_b):
                gestures.append("sitting begin")
                if self.sitting_start_time is None:
                    self.sitting_start_time = time.time()
                elif time.time() - self.sitting_start_time > 1:
                    gestures.append("sitting")
            else:
                self.sitting_start_time = None

            if self.is_waving(keypoints_b):
                gestures.append("waving")

        for result_c in results_body:
            if result_c is not None:
                keypoints = result_c.keypoints.xy[0]
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
        return distance < 50  # 设定阈值为50像素

    def is_sitting(self, keypoints):
        left_shoulder = keypoints[body_map["left_shoulder"]]
        left_knee = keypoints[body_map["left_knee"]]
        right_knee = keypoints[body_map["right_knee"]]
        left_hip = keypoints[body_map["left_hip"]]
        right_hip = keypoints[body_map["right_hip"]]
        base_distance = np.linalg.norm(left_hip - left_shoulder)

        if np.abs(left_hip[1] - left_knee[1]) < 0.4 * base_distance and np.abs(right_hip[1] - right_knee[1]) < 0.4 * base_distance:
            return True

    def is_waving(self, keypoints):
        left_hip = keypoints[body_map["left_hip"]]
        left_shoulder = keypoints[body_map["left_shoulder"]]
        left_wrist = keypoints[body_map["left_wrist"]]
        right_wrist = keypoints[body_map["right_wrist"]]
        left_elbow = keypoints[body_map["left_elbow"]]
        right_elbow = keypoints[body_map["right_elbow"]]
        nose = keypoints[body_map["nose"]]
        base_distance = np.linalg.norm(left_hip - left_shoulder)

        # 把手举到鼻子附近来激活判断
        if np.abs(left_wrist[1] - nose[1]) < 0.25 * base_distance:
            self.using_wrist = "left"
        elif np.abs(right_wrist[1] - nose[1]) < 0.25 * base_distance:
            self.using_wrist = "right"
        else:
            return False

        if np.abs(left_wrist[1] - nose[1]) > base_distance and np.abs(right_wrist[0] - nose[0]) > base_distance:
            self.using_wrist = None
            self.previous_wrist = None
            return False

        if self.previous_wrist is None:
            self.previous_wrist = keypoints[body_map[f"{self.using_wrist}_wrist"]]
            return False
        distance = np.linalg.norm(keypoints[body_map[f"{self.using_wrist}_wrist"]] - self.previous_wrist)
        speed = distance / (1 / 30) # 帧率为30
        self.previous_wrist = keypoints[body_map[f"{self.using_wrist}_wrist"]]
        return speed > 20

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
