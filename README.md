# Footfall Counter: People Entry/Exit Detection using YOLO and SORT


## Project Overview
This project implements a **footfall counting system** that detects and 
tracks people in a video and counts entries and exits across a designated 
line. The system uses:


- **YOLOv8** (Ultralytics) for object detection  
- **SORT** (Simple Online and Realtime Tracking) with Kalman Filter for tracking  
- Line crossing logic to count entries and exits  


A bonus **API** is also included that allows uploading a video and receiving 
the entry/exit counts using **FastAPI**.

---


## Approach

1. **Detection**:  
   Each video frame is processed using YOLOv8 to detect objects. Only `person`
   detections are considered.  

2. **Tracking**:  
   Detected persons are tracked using **SORT**, which uses a Kalman Filter 
   to predict positions in consecutive frames. Each person is assigned a unique ID to maintain consistent tracking.  

3. **Counting Logic**:  
   A horizontal line (usually middle of the frame) is defined.  
   - When a person’s center crosses the line **from top to bottom**, it is counted as an **entry**.  
   - When crossing **from bottom to top**, it is counted as an **exit**.  

4. **Visualization**:  
   Bounding boxes and IDs are drawn on frames along with the counting line. Entry/exit counts are displayed in real-time.  


---


## Video Source

- Example video file: `./data/video.mov` (you can replace this with any video where people enter/exit a scene).  
- The system works with any standard video format readable by OpenCV (MP4, MOV, AVI, etc.).


---

## Counting Logic (Detailed)

This project counts **entries and exits** using the combination of **SORT tracker** and **line-crossing logic**:

1. **SORT Tracker**  
   - Each detected person is assigned a **unique tracker ID** using a **Kalman Filter**.  
   - Kalman Filter predicts the next position of each person even if detection is temporarily lost, providing **stable tracking**.  
   - Each tracker maintains:
     - `last_y`: last vertical center position  
     - `counted`: whether this person has already been counted  

2. **Line Crossing Check**  
   - A horizontal line (`line_y`) is defined (typically the middle of the frame).  
   - For each tracked person:
     - Compute **current center y-coordinate**: `(y1 + y2)/2`.  
     - If `last_y < line_y <= current_y`, the person moved **top → bottom** → counted as **entry**.  
     - If `last_y > line_y >= current_y`, the person moved **bottom → top** → counted as **exit**.  
   - Once counted, `counted = True` to prevent double-counting.  

3. **Tracker Updates**  
   - Each frame:
     - Predict next positions for all trackers.  
     - Match new detections with trackers using **IoU**.  
     - Update matched trackers with new bounding boxes.  
     - Add unmatched detections as new trackers.  
     - Remove trackers that exceed `max_age` frames without updates.  

4. **Output**  
   - The system continuously updates `entry_count` and `exit_count` for the video.  
   - Counts are displayed on video frames and returned via API when using FastAPI.  


---


## Dependencies

Install all required packages using `requirements.txt`:

    ```bash
    pip install -r requirements.txt

## Setup Instructions

### Clone the Repository
    ```bash
    git clone <https://github.com/Jayeshm93/Footfall-People-Counter.git>


## Examples to run

### 1. To see the real-time video counting 

Run to see real-time video detection-tracker and counter:

    ```bash
    python main.py


### 2. People counting screenshots
That blue line is for counting the people for entry and exits.
<img src="resources/Screenshot 2025-10-23 201855.png" alt="Dashboard" width="600"/>


### 3. After processign the all video will print the final all counted people.


### 2. To providing the video and returns counts without seeing the video screening
1. Run the server api:

    ```bash
    python server.py

2. Go in postman and send this call with video file:
    
    ```bash
    http://127.0.0.1:8000/count_entries_exits

3. Get final counter after processing the video in postman:
    
    ```bash
    {
    "entries": 4,
    "exits": 0
    }


---


