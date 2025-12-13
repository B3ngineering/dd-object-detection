from ultralytics import YOLO
import torch


class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.4):
        """
        model_name: yolov8n.pt, yolov8s.pt, etc.
        conf_threshold: minimum confidence for detections
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Class names from COCO
        self.class_names = self.model.names

    def detect(self, frame):
        """
        Run detection on a single frame.
        Returns a list of detections.
        """
        results = self.model(frame, verbose=False)[0]

        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            label = self.class_names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": cls_id,
                "label": label
            })

        return detections