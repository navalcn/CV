# Project Brief: Traffic Detection, Tracking, Speed Estimation & Tableau Dashboard

## Goal
Build a computer vision mini-project (course: Deep Learning for Computer Vision) that detects, tracks, and counts vehicles in traffic video using YOLOv8 + ByteTrack, estimates vehicle speed using homography-based ground-plane calibration, exports the results to CSV, and visualizes them in an interactive Tableau dashboard. Target output also includes a course project report (docx) and, longer-term, an IEEE-style paper.

## Status: in progress, mid-debugging
The pipeline runs end-to-end and produces real tracking data, but the **speed values from the first working run were wrong** (avg 259 km/h, max 405 km/h — physically impossible). Root cause diagnosed and a fix (homography calibration) has been built but **not yet run/validated**. See "Known issue" below before doing anything else.

---

## Files already built (in this order of evolution)

| File | Purpose | Status |
|---|---|---|
| `detect_and_export.py` | Simplest version: static-image detection on COCO128 + IoU/mAP eval. No video. | Working, superseded |
| `track_and_count.py` | Adds video + ByteTrack tracking + single-line crossing count (avoids double-counting). | Working, superseded |
| `speed_estimation.py` | Adds two-line, flat-distance speed estimation. | Working but produces wrong speeds (see issue) |
| `combined_pipeline.py` | Merges tracking + counting + two-line speed + annotated output video + summary charts (matplotlib) in one script. | Working but same speed bug as above |
| `extract_calibration_frame.py` | Utility: saves one video frame with a pixel-coordinate grid overlay, to read calibration point coordinates by eye. | Working, use this first |
| `homography_speed_pipeline.py` | **Current main script.** Replaces the flat two-line distance assumption with a proper homography (perspective transform) that maps pixel positions to real-world ground-plane meters, giving position-independent, geometrically correct speed. Also adds a top-down "mini-map" visualization overlay in the output video. | **Built but not yet validated with real calibration values** |
| `requirements.txt` | `ultralytics, pandas, opencv-python, lap, matplotlib, numpy` | Current |
| `Traffic_Detection_Tableau_Draft_Paper.docx` | Draft course report (abstract, objectives, methodology, results placeholders, Tableau dashboard design, conclusion, references). Methodology section already describes tracking + line-crossing counting + speed estimation. Results tables still have placeholder dashes — need real numbers. | Draft, needs real results filled in |

## Known issue: speed calibration (READ BEFORE RUNNING FURTHER)

The video used for testing is an **oblique/angled drone-style traffic shot**, not a straight-down overhead view. This means pixel-to-meter scale is NOT constant across the frame — a vehicle near the camera (bottom of frame) appears much bigger than the same vehicle far from the camera (top of frame). The original two-line method (`combined_pipeline.py`, `speed_estimation.py`) assumed a single flat "X meters between these two lines" — this is wrong when the two lines are at different depths, and produced impossible speeds (avg 259 km/h, max 405 km/h, from a sample run: FPS=30, 1920x1080, Line1 y=486px, Line2 y=756px, distance assumed =15m, actual travel times ~0.13–0.27s / 4–8 frames).

**Fix built:** `homography_speed_pipeline.py` uses `cv2.getPerspectiveTransform` calibrated from 4 pixel points (forming a real-world rectangle on the road) to correctly map any pixel to ground-plane meters, regardless of depth.

**What's NOT done yet:** The `SRC_POINTS`, `LANE_WIDTH_M`, and `LANE_LENGTH_M` constants at the top of `homography_speed_pipeline.py` are only rough placeholder guesses (read approximately off a screenshot, not measured). Before trusting any results from this script:

1. Run `extract_calibration_frame.py --video traffic.mp4 --frame 30` to get `calibration_frame.png`.
2. Zoom into that image and identify 4 points on the road forming a rectangle (e.g. left/right lane edges at a "far" row and a "near" row).
3. Set `LANE_WIDTH_M` (usually safe to assume ~3.5m for a standard motorway lane).
4. Determine `LANE_LENGTH_M` (distance between the far row and near row) as accurately as possible — via a known landmark distance, a confirmed lane-marking dash/gap standard for the road's country, or as a last resort the vehicle-length approximation method (document this as a limitation if used).
5. Edit those constants in `homography_speed_pipeline.py`, then run it and sanity-check that resulting speeds are realistic (e.g. tens to ~130 km/h for a highway, not hundreds).

---

## How to run everything locally (no Colab)

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Get a calibration reference frame
python extract_calibration_frame.py --video traffic.mp4 --frame 30
# -> open calibration_frame.png, read off 4 pixel points + know lane width

# 3. Edit homography_speed_pipeline.py:
#    - SRC_POINTS = [(far-left), (far-right), (near-right), (near-left)]
#    - LANE_WIDTH_M = <your value, e.g. 3.5>
#    - LANE_LENGTH_M = <your measured value>

# 4. Run the main pipeline
python homography_speed_pipeline.py --video traffic.mp4 --speed_limit_kmh 50
```

### Outputs produced (in the working directory)
- `annotated_output.mp4` — video with boxes, track IDs, live speed labels, calibration trapezoid, counting line, and a top-down mini-map overlay
- `tracked_detections.csv` — every detected object, every frame
- `traffic_counts_timeseries.csv` — unique vehicle counts per class, per 5-second time bucket (for a Tableau time-series chart)
- `vehicle_speed_readings.csv` — rolling per-frame speed samples per tracked vehicle
- `vehicle_speed_summary.csv` — one row per vehicle: avg/max speed, over-limit flag
- `summary_charts.png` — quick class-count bar chart + speed histogram (for dropping into the report)

---

## Remaining to-do list

1. **Calibrate and validate** `homography_speed_pipeline.py` (see "Known issue" above) — top priority, blocks everything else.
2. Re-run on the real traffic video, confirm speeds are physically plausible.
3. Import the 4 CSVs into Tableau; build the dashboard (class distribution, confidence/speed histograms, per-class accuracy from a separate COCO128 `model.val()` run, traffic-flow-over-time line chart, detection location heatmap; add class + confidence filters).
4. Fill the real numbers, dashboard screenshots, and annotated-frame screenshots into `Traffic_Detection_Tableau_Draft_Paper.docx` (Results section currently has placeholder dashes).
5. Optional (discussed, not started): if going for an IEEE paper afterward — reformat to two-column IEEE style, add 10–15 more citations (traffic counting via deep learning, YOLO comparisons, MOT benchmarks), add a figures section (architecture diagram, annotated frames, dashboard screenshots), and write a tight, honest "contribution" statement (the homography-based, no-training-required, BI-integrated pipeline framing — not a claim of new detector/tracker architecture).

## Key context worth remembering
- No custom model training anywhere — everything uses pre-trained YOLOv8n. This was an intentional simplicity/scope decision.
- COCO128 (auto-downloaded via `ultralytics`) is used purely for static IoU/mAP benchmarking, separate from the traffic video used for tracking/counting/speed.
- Course: "Deep Learning For Computer Vision" (ADS23703), Global Academy of Technology — project should map to Module 2 (Object Detection) learning outcomes (IoU, mAP, R-CNN/YOLO).
