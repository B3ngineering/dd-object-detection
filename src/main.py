import argparse
import cv2
import os
from video_stream import VideoStream
from detector import ObjectDetector, MultiModelDetector, ObjectTracker
from taxonomy import get_taxonomy, THREAT_LEVELS, THREAT_COLORS


def file_or_index(value):
    """Custom argparse type: accepts int (webcam) or valid video path."""
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError('Webcam index must positive')
        return ivalue
    except ValueError:
        if not os.path.exists(value):
            raise argparse.ArgumentTypeError(f'File {value} does not exist')
        return value


def draw_detection(frame, obj):
    x1, y1, x2, y2 = [int(x) for x in obj["bbox"]]
    label = obj["label"]
    conf = obj["confidence"]
    track_id = obj["track_id"]
    occluded = obj.get("occluded", False)
    num_models = obj.get("num_models", 1)

    taxonomy = get_taxonomy(label)
    threat = THREAT_LEVELS.get(taxonomy, 0)
    color = THREAT_COLORS.get(threat, (200, 200, 200))
    if occluded:
        color = tuple(int(c * 0.5) for c in color)
        box_style = 1
    else:
        box_style = 2
    if num_models > 1:
        text = f"ID:{track_id} {taxonomy} ({label}): {conf:.2f} [T{threat}] ({num_models})"
    else:
        text = f"ID:{track_id} {taxonomy} ({label}): {conf:.2f} [T{threat}]"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_style)
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_detection(source, multi_model=True):
    stream = VideoStream(source=source)
    if multi_model:
        detector = MultiModelDetector(
            model_configs=[
                {"path": "models/best_arctic_military.pt", "weight": 1, "priority": 2},  # Custom arctic model
                {"path": "models/threat_detection.pt", "weight": 0.6, "priority": 1},  # Threat model
                {"path": "models/yolov8s.pt", "weight": 0.4, "priority": 0},  # CoCo model
            ],
            conf_threshold=0.4,
            iou_threshold=0.5
        )
    else:
        detector = ObjectDetector(
            model_name="models/yolov8s.pt", conf_threshold=0.4
        )
    tracker = ObjectTracker(max_disappeared=30, max_distance=100)
    window_name = "Detection Stream"
    while True:
        frame = stream.read()
        if frame is None:
            break
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)
        for obj in tracked_objects:
            draw_detection(frame, obj)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1 if isinstance(source, int) else 25) & 0xFF == ord('q'):
            break
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Object Detection: Video file or webcam")
    parser.add_argument("source", type=file_or_index, help="Video file path or webcam index")
    parser.add_argument("--multi-model", action="store_true", help="Force multi-model detection")
    args = parser.parse_args()
    # Auto-select multi-model for files, single-model for webcam unless overridden
    is_webcam = isinstance(args.source, int)
    multi_model = args.multi_model or not is_webcam
    run_detection(args.source, multi_model=multi_model)
