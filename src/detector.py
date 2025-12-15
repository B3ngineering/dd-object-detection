from ultralytics import YOLO
import torch
import numpy as np
from collections import OrderedDict


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


# Used for military label prioritization
MILITARY_LABELS = {
    "soldier",
    "camouflage_soldier",
    "civilian",
    "military_tank",
    "military_truck",
    "military_vehicle",
    "military_aircraft",
    "military_warship",
    "military_artillery",
    "weapon",
    "trench"
}


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


class MultiModelDetector:
    """
    Aggregates detections from multiple models for increased confidence.
    """
    def __init__(self, model_configs, conf_threshold=0.3, iou_threshold=0.5):
        """
        model_configs: list of {"path": str, "weight": float, "priority": int}
        """
        self.models = []
        self.weights = []
        self.priorities = []

        for config in model_configs:
            self.models.append(YOLO(config["path"]))
            self.weights.append(config.get("weight", 1.0))
            self.priorities.append(config.get("priority", 0))

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame):
        """
        Run detection with all models and merge results.
        Returns merged detections.
        """
        all_detections = []

        for idx, (model, weight, priority) in enumerate(zip(self.models,
                                                            self.weights,
                                                            self.priorities)):
            results = model(frame, verbose=False)[0]
            for box in results.boxes:
                all_detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "label": model.names[int(box.cls[0])],
                    "weight": weight,
                    "priority": priority,
                    "model_idx": idx
                })

        merged = self._merge_detections(all_detections)
        return [d for d in merged if d["confidence"] >= self.conf_threshold]

    def _merge_detections(self, detections):
        """
        Merge overlapping detections from different models.
        Does this by grouping detections with IoU above threshold,
        then passing them to _combine_group.
        """
        if not detections:
            return []

        merged = []
        used = set()

        for i, det in enumerate(detections):
            if i in used:
                continue

            group = [det]
            used.add(i)

            for j, other in enumerate(detections):
                if j in used:
                    continue

                iou = calculate_iou(det["bbox"], other["bbox"])
                if iou >= self.iou_threshold:
                    group.append(other)
                    used.add(j)

            merged.append(self._combine_group(group))

        return merged

    def _combine_group(self, group):
        """
        Concisely combine overlapping detections, prioritizing military labels.
        - Military labels always override generic ones (e.g., 'soldier' > 'person').
        - Bounding box is a weighted average (military model's box preferred if present).
        - Confidence is boosted for multi-model agreement (+15% per extra model, max +30%).
        """
        if not group:
            return None
        if len(group) == 1:
            d = group[0]
            return {
                "bbox": [int(x) for x in d["bbox"]],
                "confidence": d["confidence"],
                "label": d["label"],
                "num_models": 1
            }

        # Find best label: prefer military, else most confident
        best = max(
            group,
            key=lambda d: (
                d["label"] in MILITARY_LABELS,
                d["priority"],
                d["confidence"]
            )
        )
        best_label = best["label"]

        # Prefer military model's bbox if available
        military_det = next((d for d in group if d["label"] in MILITARY_LABELS), None)
        if military_det:
            avg_bbox = military_det["bbox"]
        else:
            # Weighted average bbox
            total_weight = sum(d["weight"] for d in group)
            avg_bbox = [
                sum(d["bbox"][k] * d["weight"] for d in group) / total_weight
                for k in range(4)
            ]

        # Confidence: weighted average, boost for agreement
        total_weight = sum(d["weight"] for d in group)
        base_conf = (
            military_det["confidence"] * 0.7 +
            sum(d["confidence"] * d["weight"] for d in group) / total_weight * 0.3
        ) if military_det else sum(d["confidence"] * d["weight"] for d in group) / total_weight
        agreement_boost = min(0.15 * (len(group) - 1), 0.3)
        final_conf = min(base_conf + agreement_boost, 1.0)

        return {
            "bbox": [int(x) for x in avg_bbox],
            "confidence": final_conf,
            "label": best_label,
            "num_models": len(group)
        }


class ObjectTracker:
    """
    Tracks objects across frames using centroid-based tracking.
    Persists object IDs even during temporary occlusion.
    """
    def __init__(self, max_disappeared=30, max_distance=100):
        """
        max_disappeared: frames to keep tracking after object disappears
        max_distance: max pixel distance to consider same object
        """
        self.next_id = 0
        self.objects = OrderedDict()  # id->{"centroid", "bbox", "label", "confidence", "last_seen"}
        self.disappeared = OrderedDict()  # id->frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _get_centroid(self, bbox):
        """Calculate centroid from bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, p1, p2):
        """Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def register(self, detection):
        """Register a new object."""
        centroid = self._get_centroid(detection["bbox"])
        self.objects[self.next_id] = {
            "centroid": centroid,
            "bbox": detection["bbox"],
            "label": detection["label"],
            "confidence": detection["confidence"],
            "num_models": detection.get("num_models", 1),
            "last_seen": 0,
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Update tracker with new detections.
        Returns list of tracked objects with persistent IDs.
        """
        # No detections - mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                self.objects[object_id]["last_seen"] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._get_tracked_objects()

        # No existing objects - register all detections
        if len(self.objects) == 0:
            for det in detections:
                self.register(det)
            return self._get_tracked_objects()

        # Match detections to existing objects
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid]["centroid"] for oid in object_ids]

        detection_centroids = [self._get_centroid(d["bbox"]) for d in detections]

        # Calculate distance matrix
        distances = np.zeros((len(object_centroids), len(detection_centroids)))
        for i, oc in enumerate(object_centroids):
            for j, dc in enumerate(detection_centroids):
                distances[i, j] = self._distance(oc, dc)

        used_objects = set()
        used_detections = set()
        matches = []

        # Sort by distance and greedily assign
        flat_indices = np.argsort(distances.flatten())
        for idx in flat_indices:
            i = idx // len(detection_centroids)
            j = idx % len(detection_centroids)

            if i in used_objects or j in used_detections:
                continue

            if distances[i, j] > self.max_distance:
                continue

            # Only match if labels are compatible
            obj_label = self.objects[object_ids[i]]["label"]
            det_label = detections[j]["label"]
            if not self._labels_compatible(obj_label, det_label):
                continue

            matches.append((i, j))
            used_objects.add(i)
            used_detections.add(j)

        # Update matched objects
        for i, j in matches:
            object_id = object_ids[i]
            det = detections[j]
            centroid = self._get_centroid(det["bbox"])

            self.objects[object_id]["centroid"] = centroid
            self.objects[object_id]["bbox"] = det["bbox"]
            self.objects[object_id]["confidence"] = det["confidence"]
            self.objects[object_id]["num_models"] = det.get("num_models", 1)
            self.objects[object_id]["last_seen"] = 0
            if det["label"] in MILITARY_LABELS:
                self.objects[object_id]["label"] = det["label"]
            self.disappeared[object_id] = 0

        # Handle unmatched objects (disappeared)
        for i in range(len(object_ids)):
            if i not in used_objects:
                object_id = object_ids[i]
                self.disappeared[object_id] += 1
                self.objects[object_id]["last_seen"] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        # Handle unmatched detections (new objects)
        for j in range(len(detections)):
            if j not in used_detections:
                self.register(detections[j])

        return self._get_tracked_objects()

    def _labels_compatible(self, label1, label2):
        """Check if two labels could be the same object."""
        # Same label
        if label1 == label2:
            return True

        # Person-related labels are compatible
        person_labels = {"person", "soldier", "camouflage_soldier", "civilian"}
        if label1 in person_labels and label2 in person_labels:
            return True

        # Vehicle-related labels
        vehicle_labels = {"car", "truck", "bus", "military_truck",
                          "military_vehicle", "civilian_vehicle"}
        if label1 in vehicle_labels and label2 in vehicle_labels:
            return True

        return False

    def _get_tracked_objects(self):
        """Return list of currently tracked objects."""
        results = []
        for object_id, data in self.objects.items():
            results.append({
                "track_id": object_id,
                "bbox": data["bbox"],
                "label": data["label"],
                "confidence": data["confidence"],
                "num_models": data.get("num_models", 1),
                "occluded": data["last_seen"] > 0,
                "frames_since_seen": data["last_seen"]
            })
        return results
