import cv2
import numpy as np
from detector import YOLODetector
from tracker_sort import Sort  # SORT tracker with line counting and stable tracking


def initialize_detector(model_name="yolov8n.pt", device="cpu", conf_thresh=0.35):
    """
    Initialize YOLO detector for object detection.
    - model_name: path to YOLO model
    - device: 'cpu' or 'cuda'
    - conf_thresh: confidence threshold for detections
    """
    # Just load the model; nothing fancy
    return YOLODetector(model_name=model_name, device=device, conf_thresh=conf_thresh)


def initialize_tracker(frame_height):
    """
    Initialize the SORT tracker with a counting line in the middle of the frame.
    - frame_height: height of the video frame (to set line position)
    Returns:
    - tracker object
    - line_y: vertical position of the counting line
    """
    line_y = frame_height // 2  # middle of the frame
    tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3, line_y=line_y)
    return tracker, line_y


def filter_person_detections(detections):
    """
    From all detections, keep only persons.
    - detections: output from YOLO detector
    Returns:
    - list of [x1, y1, x2, y2, confidence] for persons
    """
    person_boxes = []
    for x1, y1, x2, y2, conf, cls, name in detections:
        if name == "person":  # we only care about humans
            person_boxes.append([x1, y1, x2, y2, conf])
    return person_boxes


def draw_tracked_objects(frame, tracked_objects):
    """
    Draw bounding boxes, IDs, and center dots on the frame.
    - frame: current video frame
    - tracked_objects: output from SORT tracker [x1, y1, x2, y2, track_id]
    """
    for x1, y1, x2, y2, track_id in tracked_objects:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # center of the box
        # Draw green rectangle around the person
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Draw the ID above the rectangle
        cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # Draw a small red dot at the center
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


def draw_counting_line(frame, tracker, line_y, width):
    """
    Draw horizontal counting line and display entry/exit counts.
    - frame: current video frame
    - tracker: SORT tracker (contains counts)
    - line_y: vertical position of counting line
    - width: frame width
    """
    line_color = (255, 0, 0)  # default blue

    # Change color if a new person entered or exited
    if tracker.entry_count > getattr(tracker, "_last_entry_count", 0):
        line_color = (0, 255, 0)  # green flash for entry
        tracker._last_entry_count = tracker.entry_count
    elif tracker.exit_count > getattr(tracker, "_last_exit_count", 0):
        line_color = (0, 0, 255)  # red flash for exit
        tracker._last_exit_count = tracker.exit_count

    # Draw the horizontal line
    cv2.line(frame, (0, line_y), (width, line_y), line_color, 3)

    # Display current counts on top-left corner
    cv2.putText(frame, f"Entries: {tracker.entry_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"Exits: {tracker.exit_count}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


def process_video(video_path, model_name="yolov8n.pt", device="cpu"):
    """
    Main function to process a video and track/count people.
    - video_path: path to the video file
    - model_name: YOLO model path
    - device: 'cpu' or 'cuda'
    """
    # Load YOLO detector
    det = initialize_detector(model_name, device)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video file.")
        return

    # Read first frame to get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("❌ Unable to read video frame.")
        cap.release()
        return

    height, width, _ = frame.shape

    # Initialize tracker and counting line
    tracker, line_y = initialize_tracker(height)

    print("🚀 Tracking started... Press 'q' to quit.\n")

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # video ended

        # Detect all objects
        detections = det.detect(frame)

        # Keep only persons
        person_boxes = filter_person_detections(detections)

        # Update tracker with current detections
        tracked_objects = tracker.update(
            np.array(person_boxes) if len(person_boxes) > 0 else np.empty((0, 5))
        )

        # Draw bounding boxes, IDs, and centers
        draw_tracked_objects(frame, tracked_objects)

        # Draw counting line and entry/exit counts
        draw_counting_line(frame, tracker, line_y, width)

        # Show the processed frame
        cv2.imshow("Footfall Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # quit on 'q'

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Print final counts
    tracker.print_final_counts()


# Run by providing the path
process_video("./data/video.mov")
