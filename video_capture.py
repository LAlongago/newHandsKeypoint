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

    def release(self):
        self.cap.release()
