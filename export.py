from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8l-pose.pt")  # load an official model
model = YOLO(r"hands_keypoint_model_l/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")