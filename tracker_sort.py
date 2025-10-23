import numpy as np
from filterpy.kalman import KalmanFilter


# Function to compute Intersection over Union (IoU)
def iou(bb_test, bb_gt):
    """
    Compute IoU between two bounding boxes.

    Parameters:
    - bb_test: bounding box to test [x1, y1, x2, y2]
    - bb_gt: ground-truth bounding box [x1, y1, x2, y2]

    Returns:
    - IoU value (float)
    """
    # Find coordinates of intersection rectangle
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    # Compute width and height of intersection rectangle
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    # Intersection area
    wh = w * h

    # Union area = area1 + area2 - intersection
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


# Tracker class using Kalman Filter
class KalmanBoxTracker:
    """
    Keeps track of one object using a Kalman Filter.
    Each object gets a unique ID. Supports predicting positions
    and updating with new detections.
    """
    count = 0  # Unique ID counter for trackers

    def __init__(self, bbox):
        """
        Initialize the tracker with the first detected bounding box.
        """
        # Initialize Kalman filter with 7 state variables, 4 measurement variables
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix: models constant velocity motion
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement function: we measure [x, y, s, r] directly
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # Set measurement noise
        self.kf.R[2:, 2:] *= 10.

        # Set initial state covariance (uncertainty)
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.

        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state with the given bounding box
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        # Tracker bookkeeping
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Custom attributes for line crossing logic
        self.last_y = None
        self.counted = False

    def update(self, bbox):
        """Update tracker state with a new detection."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """Predict the next position of the tracker."""
        # Avoid negative scale (width/height)
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1

        # Reset hit streak if not updated
        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1

        # Store predicted bbox in history
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Return the current bounding box estimate."""
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """Convert bbox from [x1, y1, x2, y2] to [x, y, s, r]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h)
        return np.array([[x], [y], [s], [r]])

    @staticmethod
    def convert_x_to_bbox(x):
        """Convert state vector [x, y, s, r] back to bbox [x1, y1, x2, y2]."""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))


# SORT tracker with line crossing counting
class Sort:
    """
    SORT tracker implementation with entry/exit counting.
    Tracks objects and counts when they cross a horizontal line.
    """

    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3, line_y=300):
        self.max_age = max_age  # max frames to keep unmatched tracker
        self.min_hits = min_hits  # min hits to consider a track valid
        self.iou_threshold = iou_threshold
        self.trackers = []  # list of KalmanBoxTracker objects
        self.frame_count = 0
        self.line_y = line_y  # y-coordinate of counting line

        # Entry/Exit counters
        self.entry_count = 0
        self.exit_count = 0

    def update(self, dets=np.empty((0, 5))):
        """Update all trackers with new detections and count line crossings."""
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        # Predict each tracker's next position
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Add new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        # Line crossing logic
        for trk in self.trackers:
            d = trk.get_state()[0]
            center_y = (d[1] + d[3]) / 2

            # Check if tracker crosses the line
            if trk.last_y is not None and not trk.counted:
                if trk.last_y < self.line_y <= center_y:
                    self.entry_count += 1
                    trk.counted = True
                elif trk.last_y > self.line_y >= center_y:
                    self.exit_count += 1
                    trk.counted = True
            trk.last_y = center_y

            # Add active trackers to return list
            if trk.time_since_update < 1:
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))

        # Print counts for each frame
        if len(ret) > 0:
            print(f"Current Counts → Entry: {self.entry_count}, Exit: {self.exit_count}")
            return np.concatenate(ret)
        return np.empty((0, 5))

    def print_final_counts(self):
        """Display the final total entry/exit counts."""
        print(f"\n📊 Final Counts — Entry: {self.entry_count} | Exit: {self.exit_count}")


# Function to associate detections with trackers
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Match detections to existing trackers using IoU.
    Returns matched pairs, unmatched detections, unmatched trackers.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)

    # Compute IoU matrix between all detections and trackers
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # Sort matches by descending IoU
    matched_indices = np.array(np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)).T

    unmatched_detections = list(range(len(detections)))
    unmatched_trackers = list(range(len(trackers)))
    matches = []

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            continue
        if m[0] in unmatched_detections and m[1] in unmatched_trackers:
            matches.append(m)
            unmatched_detections.remove(m[0])
            unmatched_trackers.remove(m[1])

    return np.array(matches), np.array(unmatched_detections), np.array(unmatched_trackers)
