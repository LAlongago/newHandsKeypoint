import cv2
from ultralytics import YOLO
import torch
import os

# 加载训练好的YOLOv8l pose模型
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("hands_keypoint_model_l/best.pt")

# 定义视频文件路径
video_path = r"data/test/test10.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化视频写入器，用于保存带有关键点的输出视频
output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = "results/annotated_video.mp4"
postfix = 1
while os.path.exists(output_path):
    postfix += 1
    output_path = f"{output_dir}/annotated_video_{postfix}.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# 定义关键点的颜色和字体
point_color = (0, 255, 0)  # 绿色
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1

# 循环处理视频的每一帧
while cap.isOpened():
    success, frame = cap.read()  # 读取一帧图像
    if success:
        # 对当前帧进行YOLOv8推理
        results = model(frame)

        # 获取关键点信息
        for result in results:
            if result.keypoints is None or result.keypoints.xy is None or result.keypoints.conf is None:
                continue

            keypoints_list = result.keypoints.xy  # 获取所有检测对象的关键点坐标
            confs_list = result.keypoints.conf  # 获取所有检测对象的关键点置信度

            for keypoints, confs in zip(keypoints_list, confs_list):
                # 绘制关键点及其序号和置信度
                for i, (x, y) in enumerate(keypoints):
                    cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)  # 绘制关键点
                    cv2.putText(frame, str(i + 1), (int(x) + 5, int(y) - 5), font, font_scale, point_color,
                                thickness)  # 标注序号
                    cv2.putText(frame, str(round(confs[i].item(), 2)), (int(x) + 5, int(y) + 15), font, font_scale,
                                point_color, thickness)  # 标注置信度

        # 显示带有关键点的帧
        cv2.imshow("YOLOv8 Pose Inference", frame)

        # 写入带有关键点的帧到输出视频
        out.write(frame)

        # 如果按下'q'键，则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# 释放视频捕获对象和关闭窗口
cap.release()
out.release()
cv2.destroyAllWindows()
