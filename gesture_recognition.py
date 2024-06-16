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

    def recognize(self, results, results_body):
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
            keypoints = result_b.keypoints.xy[0]

            if self.is_sitting(keypoints):
                if self.sitting_start_time is None:
                    self.sitting_start_time = time.time()
                elif time.time() - self.sitting_start_time > 3:
                    gestures.append("sitting")
            else:
                self.sitting_start_time = None

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
        left_knee = keypoints[body_map["left_knee"]]
        right_knee = keypoints[body_map["right_knee"]]
        left_ankle = keypoints[body_map["left_ankle"]]
        right_ankle = keypoints[body_map["right_ankle"]]
        base_distance = np.linalg.norm(left_knee - left_ankle)
        left_hip = keypoints[body_map["left_hip"]]
        right_hip = keypoints[body_map["right_hip"]]
        if np.linalg.norm(left_hip - left_knee) < 0.3 * base_distance and np.linalg.norm(
                right_hip - right_knee) < 0.3 * base_distance:
            return True

        return False