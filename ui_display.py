import cv2
import os

# 定义关键点的颜色和字体
point_color = (0, 255, 0)  # 绿色
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 1

class UIDisplay:
    def __init__(self, fps, frame_height, frame_width):
        self.window_name = "Hand Body Gesture Recognition"
        self.output_dir = "results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_path = "results/annotated_video.mp4"
        self.postfix = 1
        while os.path.exists(self.output_path):
            self.postfix += 1
            self.output_path = f"{self.output_dir}/annotated_video_{self.postfix}.mp4"
        self.out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        self.active_gestures = {}
        self.fps = fps
        self.frame_count = 0

        # 创建日志文件
        self.log_path = f"{self.output_dir}/gesture_log.txt"
        self.log_file = open(self.log_path, "w")

    def show(self, frame, results, results_body, gestures):
        self.frame_count += 1
        current_time = self.frame_count / self.fps  # 计算当前时间（秒）

        # 清空旧的显示字典
        self.active_gestures.clear()

        # 处理身体关键点
        for result_b in results_body:
            keypoints = result_b.keypoints.xy[0]
            for i, (x, y) in enumerate(keypoints):
                cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)  # 绘制关键点
                cv2.putText(frame, str(i + 1), (int(x) + 5, int(y) - 5), font, font_scale, point_color,
                            thickness)  # 标注序号

        # 处理手部关键点
        for result in results:
            keypoints = result.keypoints.xy[0]
            for i, (x, y) in enumerate(keypoints):
                cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)  # 绘制关键点
                cv2.putText(frame, str(i + 1), (int(x) + 5, int(y) - 5), font, font_scale, point_color,
                            thickness)  # 标注序号

        # 更新当前活跃的手势
        for gesture in gestures:
            if gesture not in self.active_gestures:
                self.active_gestures[gesture] = len(self.active_gestures) + 1
                self.log_file.write(f"{gesture} detected at {current_time:.2f} seconds\n")

        # 显示手势提示字符
        y_position = 50
        for gesture, position in self.active_gestures.items():
            text = ""
            if gesture == "choose":
                text = "Choose Detected"
            elif gesture == "sitting begin":
                text = "Sitting Begin Detected"
            elif gesture == "sitting":
                text = "Sitting Detected"
            elif gesture == "waving":
                text = "Waving Detected"

            cv2.putText(frame, text, (50, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_position += 30  # 增加 y 坐标，以便在下一行显示

        cv2.imshow(self.window_name, frame)
        self.out.write(frame)

    def check_exit(self):
        return cv2.waitKey(1) & 0xFF == ord("q")

    def close(self):
        cv2.destroyAllWindows()
        self.out.release()
        self.log_file.close()
