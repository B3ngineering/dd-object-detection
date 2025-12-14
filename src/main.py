import cv2
from video_stream import VideoStream
from detector import ObjectDetector
from taxonomy import MILITARY_TAXONOMY_MAP, THREAT_LEVELS, THREAT_COLORS

# Run object detection on a video file
def run_video_detection(video_path):
    stream = VideoStream(source=video_path)
    detector = ObjectDetector(
        model_name="best.pt",  # Pre-trained model, s for speed/accuracy trade-off
        conf_threshold=0.3
    )

    while True:
        frame = stream.read()
        if frame is None:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            
            # Map to taxonomy
            taxonomy = MILITARY_TAXONOMY_MAP.get(label, "Other")
            threat = THREAT_LEVELS.get(taxonomy, 0)
            color = THREAT_COLORS.get(threat, (200, 200, 200))

            # Display with taxonomy info
            text = f"{taxonomy} ({label}): {conf:.2f} [T{threat}]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Video Stream", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()

# Run object detection on webcam feed
def run_webcam():
    stream = VideoStream(source=0, width=640, height=480)
    detector = ObjectDetector(
        model_name="yolov8s.pt", conf_threshold=0.4
    )

    while True:
        frame = stream.read()
        if frame is None:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            text = f"{label}: {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Webcam Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video_detection("data/video2.mp4")
