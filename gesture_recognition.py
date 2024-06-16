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
        self.sitting_start_time = None
        self.previous_wrist = None
        self.using_wrist = None
        self.previous_left_knee = None
        self.previous_right_knee = None

    def recognize(self, results, results_body):
        gestures = []
        for result in results:
            result = result.cpu()
            if result.keypoints is None or result.keypoints.xy is None or result.keypoints.conf is None:
                continue
            keypoints = result.keypoints.xy[0]

            # 检查手掌是否完全打开
            if self.is_hand_open(keypoints):
                gestures.append("paper")

            if self.is_rock_gesture(keypoints):
                gestures.append("rock")

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

            if self.is_running(keypoints_b):
                gestures.append("running")

        return gestures

    def is_hand_open(self, keypoints):
        wrist = keypoints[finger_map["wrist"]]
        thumb_cmc = keypoints[finger_map["thumb_cmc"]]
        thumb_mcp = keypoints[finger_map["thumb_mcp"]]
        thumb_tip = keypoints[finger_map["thumb_tip"]]
        index_finger_mcp = keypoints[finger_map["index_finger_mcp"]]
        index_finger_tip = keypoints[finger_map["index_finger_tip"]]
        middle_finger_mcp = keypoints[finger_map["middle_finger_mcp"]]
        middle_finger_tip = keypoints[finger_map["middle_finger_tip"]]
        ring_finger_mcp = keypoints[finger_map["ring_finger_mcp"]]
        ring_finger_tip = keypoints[finger_map["ring_finger_tip"]]
        pinky_mcp = keypoints[finger_map["pinky_mcp"]]
        pinky_tip = keypoints[finger_map["pinky_tip"]]

        # 计算手腕到大拇指根部和手腕到每个手指尖的距离
        base_distance = np.linalg.norm(thumb_cmc - wrist)
        d_thumb = np.linalg.norm(thumb_tip - thumb_mcp)
        d_index = np.linalg.norm(index_finger_tip - index_finger_mcp)
        d_middle = np.linalg.norm(middle_finger_tip - middle_finger_mcp)
        d_ring = np.linalg.norm(ring_finger_tip - ring_finger_mcp)
        d_pinky = np.linalg.norm(pinky_tip - pinky_mcp)

        # 利用比例关系来判断手掌是否摊开
        ratio_threshold = 1.3  # 可以根据需要调整该阈值

        return (d_thumb / base_distance > 1 and
                d_index / base_distance > ratio_threshold and
                d_middle / base_distance > ratio_threshold and
                d_ring / base_distance > ratio_threshold and
                d_pinky / base_distance > ratio_threshold)

    def is_rock_gesture(self, keypoints):
        # 假设finger_map已经包含了所有手指的映射
        fingers = ["thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]

        # 检查所有手指的关键点是否存在
        # if any(finger_map[finger] not in keypoints for finger in fingers):
        # return False  # 确保所有手指的关键点都存在

        # 获取手腕点的位置，假设它存在于关键点数据中
        wrist_position = keypoints[finger_map["wrist"]]
        if wrist_position is None or wrist_position.numel() == 0:
            return False

        # 估算手掌中心位置为手腕点和所有手指尖点的平均位置
        all_positions = [keypoints[finger_map[finger]] for finger in fingers] + [wrist_position]
        palm_center = np.mean(all_positions, axis=0)

        finger_positions = [keypoints[finger_map[finger]] for finger in fingers]
        judgedistance = np.linalg.norm(keypoints[0] - keypoints[1])
        # 检查所有手指是否弯曲（即手指尖与估算的手掌中心的距离较短）
        for finger_pos in finger_positions:
            if np.linalg.norm(finger_pos - palm_center) > 1.1 * judgedistance:  # 调整阈值为一个合理的数值
                return False

        return True

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

        if left_hip is None:
            return False
        if right_hip is None:
            return False

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

        if left_wrist is None:
            return False
        if right_wrist is None:
            return False

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
        distance = np.linalg.norm(keypoints[body_map[f"{self.using_wrist}_wrist"]] - self.previous_wrist) / base_distance  # 除以基准距离，使得阈值与不同拍摄角度下的距离一致
        speed = distance / (1 / 30) # 帧率为30
        self.previous_wrist = keypoints[body_map[f"{self.using_wrist}_wrist"]]
        return speed > 1  # 设定阈值为1倍基准距离每秒

    def is_running(self, keypoints):
        left_knee = keypoints[body_map["left_knee"]]
        right_knee = keypoints[body_map["right_knee"]]
        left_ankle = keypoints[body_map["left_ankle"]]
        right_ankle = keypoints[body_map["right_ankle"]]
        left_hip = keypoints[body_map["left_hip"]]
        right_hip = keypoints[body_map["right_hip"]]
        base_distance = np.linalg.norm(keypoints[body_map["left_hip"]] - keypoints[body_map["left_shoulder"]])

        if left_hip[1] < left_knee[1] or right_hip[1] < right_knee[1]:
            return False

        if left_knee is None:
            return False
        if right_knee is None:
            return False

        if self.previous_left_knee is None:
            self.previous_left_knee = left_knee
            return False
        if self.previous_right_knee is None:
            self.previous_right_knee = right_knee
            return False
        left_distance = np.linalg.norm(left_knee - self.previous_left_knee) / base_distance
        right_distance = np.linalg.norm(right_knee - self.previous_right_knee) / base_distance
        self.previous_left_knee = left_knee
        self.previous_right_knee = right_knee
        left_speed = left_distance / (1 / 30)
        right_speed = right_distance / (1 / 30)
        return left_speed > 0.1 and right_speed > 0.1
