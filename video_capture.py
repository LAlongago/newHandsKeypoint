import cv2


class VideoCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        success, frame = self.cap.read()
        if success:
            return frame
        else:
            return None

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self):
        self.cap.release()
