import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from detector import YOLODetector
from tracker_sort import Sort
import tempfile
import os

app = FastAPI(title="Footfall Counter API (Large Files)")


# Initialize Detector (Load model once)
det = YOLODetector(model_name="yolov8n.pt", device="cpu", conf_thresh=0.35)


@app.post("/count_entries_exits")
async def count_entries_exits(video: UploadFile = File(...)):
    """
    Accept a video file (any size), process it with YOLO + SORT,
    and return entry/exit counts.
    """
    # Create a temporary file path
    suffix = os.path.splitext(video.filename)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    try:
        # Stream file in chunks to avoid memory overload
        with temp_file as f:
            while True:
                chunk = await video.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)

        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            return JSONResponse({"error": "Cannot open video file."}, status_code=400)

        # Read first frame for dimensions
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return JSONResponse({"error": "Unable to read video frame."}, status_code=400)

        height, width, _ = frame.shape
        line_y = height // 2
        tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3, line_y=line_y)

        # -------------------------------
        # Process video frames
        # -------------------------------
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect people
            detections = det.detect(frame)
            person_boxes = [
                [x1, y1, x2, y2, conf]
                for (x1, y1, x2, y2, conf, cls, name) in detections
                if name == "person"
            ]

            # Update tracker
            tracked_objects = tracker.update(
                np.array(person_boxes) if person_boxes else np.empty((0, 5))
            )

        cap.release()
        return {"entries": tracker.entry_count, "exits": tracker.exit_count}

    finally:
        # Cleanup temporary file
        os.unlink(temp_file.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

