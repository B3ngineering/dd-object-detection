import cv2


class VideoStream:
    def __init__(self, source=0, width=None, height=None):
        """
        source:
            0            -> default webcam
            1, 2, ...    -> other webcams
            'file.mp4'   -> video file
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Error: Could not open video source {source}")

        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        """Read a single frame"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release video resources"""
        self.cap.release()
