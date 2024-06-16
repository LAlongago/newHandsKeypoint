import cv2


# 定义关键点的颜色和字体
point_color = (0, 255, 0)  # 绿色
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1


class UIDisplay:
    def __init__(self):
        self.window_name = "Hand Gesture Recognition"

    def show(self, frame, results, results_body, gestures):
        for result_b in results_body:
            keypoints = result_b.keypoints.xy[0]
            for i, (x, y) in enumerate(keypoints):
                cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)  # 绘制关键点
                cv2.putText(frame, str(i + 1), (int(x) + 5, int(y) - 5), font, font_scale, point_color,
                            thickness)  # 标注序号

        for result in results:
            keypoints = result.keypoints.xy[0]
            for i, (x, y) in enumerate(keypoints):
                cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)  # 绘制关键点
                cv2.putText(frame, str(i + 1), (int(x) + 5, int(y) - 5), font, font_scale, point_color,
                            thickness)  # 标注序号

        for gesture in gestures:
            if gesture == "choose":
                cv2.putText(frame, "Choose Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(self.window_name, frame)

    def check_exit(self):
        return cv2.waitKey(1) & 0xFF == ord("q")

    def close(self):
        cv2.destroyAllWindows()
