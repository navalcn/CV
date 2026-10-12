"""
Homography-Based Traffic Speed Estimation
--------------------------------------------------------
Fixes the perspective-distortion problem of the simple two-line method:
instead of assuming a flat pixel-to-meter ratio, this maps every
vehicle's ground-contact point into real-world (x, y) METERS on the
road plane using a perspective transform (homography), calibrated
from 4 points you identify once in your video.

WHY THIS IS NEEDED:
In an oblique (non-overhead) camera view, the same number of pixels
represents very different real-world distances depending on how far
that point is from the camera (near = fewer meters/pixel, far = more
meters/pixel). A flat "distance_m per pixel-gap" assumption -- as in
the simpler two-line method -- silently breaks in this situation and
produces impossible speeds. Homography fixes this by explicitly
modeling the ground plane.

CALIBRATION -- YOU MUST SET THESE 4 POINTS + 2 REAL-WORLD DIMENSIONS:
Run extract_calibration_frame.py first to get a gridded reference image,
then identify 4 points on the road forming a rectangle (e.g. the two
edges of one lane, at a "far" row and a "near" row), and fill in:

    SRC_POINTS   = pixel (x, y) of [far-left, far-right, near-right, near-left]
    LANE_WIDTH_M = real-world width between far-left/far-right (= near-left/near-right)
    LANE_LENGTH_M = real-world distance between the far row and the near row

These 4 defaults below are ROUGH STARTING ESTIMATES read off the example
traffic video discussed in chat -- replace them with your own measured
values for your own video and location.

Run:
  python homography_speed_pipeline.py --video traffic.mp4 --speed_limit_kmh 50

Requires: pip install ultralytics opencv-python pandas lap matplotlib numpy
"""

import argparse
import collections
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================================
# CALIBRATION -- TUNED FOR single_way_traffic.mp4
# ============================================================
# Pixel coordinates (x, y) of 4 points forming a ground rectangle:
# [far-left, far-right, near-right, near-left]
# Spans the 2 active lanes between the left dashed line and right lane line:
# Far row (y=450): left dash x=330, right line x=1240
# Near row (y=750): left dash x=300, right line x=1310
SRC_POINTS = [
    (330, 450),   # far-left
    (1240, 450),  # far-right
    (1310, 750),  # near-right
    (300, 750),   # near-left
]
LANE_WIDTH_M = 7.0     # 2 standard lanes @ 3.5m each
LANE_LENGTH_M = 3.5    # length of the dash marking from y=450 to y=750 (~3.5m)

SPEED_WINDOW_FRAMES = 10  # rolling window for smooth speed estimation
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
SPEED_CLASSES = {"car", "bus", "truck", "motorcycle"}
# COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_COCO_CLASSES = [2, 3, 5, 7]
TIME_BUCKET_SECONDS = 5

# Top-down mini-map display settings
MAP_W, MAP_H = 240, 320
MAP_MARGIN = 20


def detect_lane_lines_and_vanishing_point(frame):
    """
    Automated Vanishing Point Detection:
    Uses Edge Detection + Probabilistic Hough Lines on the roadway region
    to find left and right lane boundaries and compute their intersection (Vanishing Point).
    """
    h, w = frame.shape[:2]
    # Region of interest focused on the highway surface
    roi_y_min, roi_y_max = int(h * 0.35), int(h * 0.85)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Mask outside road ROI
    mask = np.zeros_like(edges)
    mask[roi_y_min:roi_y_max, :] = 255
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=40, minLineLength=60, maxLineGap=20)
    
    left_lines = []
    right_lines = []
    try:
        if lines is not None:
            for l in lines:
                arr = np.array(l).flatten()
                if len(arr) == 4:
                    x1, y1, x2, y2 = int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
                    if x2 == x1: continue
                    slope = (y2 - y1) / float(x2 - x1)
                    # Filter steep lane boundary slopes
                    if -4.0 < slope < -0.3:
                        left_lines.append((x1, y1, x2, y2))
                    elif 0.3 < slope < 4.0:
                        right_lines.append((x1, y1, x2, y2))
                    
        # If lane lines detected with sufficient confidence, refine the 4 reference points
        if len(left_lines) >= 1 and len(right_lines) >= 1:
            # Fit average lines
            def fit_line(lines_list):
                pts = []
                for x1, y1, x2, y2 in lines_list:
                    pts.append([x1, y1])
                    pts.append([x2, y2])
                pts = np.array(pts)
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                return float(vx[0]), float(vy[0]), float(x0[0]), float(y0[0])

            vx_l, vy_l, x0_l, y0_l = fit_line(left_lines)
            vx_r, vy_r, x0_r, y0_r = fit_line(right_lines)
            
            # Calculate intersection (vanishing point)
            if abs(vy_l * vx_r - vy_r * vx_l) > 1e-4:
                m_l = vy_l / (vx_l + 1e-6)
                m_r = vy_r / (vx_r + 1e-6)
                c_l = y0_l - m_l * x0_l
                c_r = y0_r - m_r * x0_r
                if abs(m_l - m_r) > 1e-4:
                    vp_x = (c_r - c_l) / (m_l - m_r)
                    vp_y = m_l * vp_x + c_l
                    
                    # Derive 4 ground points along the detected perspective bounds
                    y_far, y_near = 450.0, 750.0
                    x_far_l = (y_far - c_l) / m_l
                    x_far_r = (y_far - c_r) / m_r
                    x_near_l = (y_near - c_l) / m_l
                    x_near_r = (y_near - c_r) / m_r
                    
                    # Sanity check derived points
                    if 0 < x_far_l < x_far_r < w and 0 < x_near_l < x_near_r < w:
                        auto_points = [
                            (int(x_far_l), int(y_far)),
                            (int(x_far_r), int(y_far)),
                            (int(x_near_r), int(y_near)),
                            (int(x_near_l), int(y_near))
                        ]
                        return auto_points, (int(vp_x), int(vp_y))
    except Exception as e:
        print(f"[Auto-Calibration Notice] Fallback to calibrated road points ({e})")
    return SRC_POINTS, None


def build_homography(custom_points=None):
    pts = custom_points if custom_points is not None else SRC_POINTS
    src = np.float32(pts)
    dst = np.float32([
        [0, 0],
        [LANE_WIDTH_M, 0],
        [LANE_WIDTH_M, LANE_LENGTH_M],
        [0, LANE_LENGTH_M],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return M, pts


def to_world(M, px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, M)
    return float(world[0, 0, 0]), float(world[0, 0, 1])


def world_to_map_xy(wx, wy):
    """Scale world meters into the top-down map canvas."""
    scale_y = (MAP_H - 2 * MAP_MARGIN) / max(LANE_LENGTH_M * 3.0, 1e-6)
    scale_x = (MAP_W - 2 * MAP_MARGIN) / max(LANE_WIDTH_M * 1.5, 1e-6)
    map_x = int(MAP_MARGIN + (wx / max(LANE_WIDTH_M, 1e-6)) * (MAP_W - 2 * MAP_MARGIN) * 0.7)
    map_y = int(MAP_MARGIN + (wy / max(LANE_LENGTH_M, 1e-6)) * (MAP_H - 2 * MAP_MARGIN) * 0.5)
    return map_x, map_y


def draw_minimap(positions):
    """positions: list of (world_x, world_y, class_name, speed_kmh_or_None)"""
    canvas = np.full((MAP_H, MAP_W, 3), 25, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (MAP_W - 1, MAP_H - 1), (80, 80, 80), 1)
    
    # Draw reference lane guides on minimap
    x0, y0 = world_to_map_xy(0, 0)
    x1, y1 = world_to_map_xy(LANE_WIDTH_M, 0)
    x2, y2 = world_to_map_xy(0, LANE_LENGTH_M)
    x3, y3 = world_to_map_xy(LANE_WIDTH_M, LANE_LENGTH_M)
    
    cv2.line(canvas, (x0, max(0, y0)), (x2, min(MAP_H, y2 + 100)), (0, 200, 0), 1)
    cv2.line(canvas, (x1, max(0, y1)), (x3, min(MAP_H, y3 + 100)), (0, 200, 0), 1)
    cv2.putText(canvas, "Top-Down Radar", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    for wx, wy, cls_name, speed in positions:
        mx, my = world_to_map_xy(wx, wy)
        if 5 <= mx < MAP_W - 5 and 5 <= my < MAP_H - 5:
            color = (0, 220, 255) if cls_name in SPEED_CLASSES else (200, 200, 0)
            cv2.circle(canvas, (mx, my), 5, color, -1)
            cv2.circle(canvas, (mx, my), 6, (0, 0, 0), 1)
            if speed is not None:
                cv2.putText(canvas, f"{speed:.0f}", (mx + 6, my + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return canvas


def draw_vehicle_badge(frame, x1, y1, x2, y2, track_id, cls_name, speed_kmh, speed_limit_kmh):
    """Draw a clean bounding box and an integrated ID + Class + Speed badge."""
    is_speed_class = cls_name in SPEED_CLASSES
    is_speeding = (speed_kmh is not None and speed_kmh > speed_limit_kmh)
    
    if not is_speed_class:
        box_color = (200, 200, 200)
    elif is_speeding:
        box_color = (0, 0, 255)      # Red for over speed limit
    elif speed_kmh is not None:
        box_color = (0, 220, 50)     # Green for normal speed
    else:
        box_color = (255, 180, 0)    # Orange during initial frames

    # Bounding box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
    
    # Badge text
    if speed_kmh is not None:
        badge_text = f"ID:{track_id} {cls_name} | {speed_kmh:.1f} km/h"
    elif is_speed_class:
        badge_text = f"ID:{track_id} {cls_name} | ..."
    else:
        badge_text = f"ID:{track_id} {cls_name}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
    
    badge_x1 = int(x1)
    badge_y1 = max(0, int(y1) - th - 10)
    badge_x2 = badge_x1 + tw + 12
    badge_y2 = badge_y1 + th + 10
    
    # Background pill for text
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
    # White or black text for optimal contrast
    text_color = (255, 255, 255) if (is_speeding or not is_speed_class) else (0, 0, 0)
    cv2.putText(frame, badge_text, (badge_x1 + 6, badge_y2 - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)


def main(video_path, speed_limit_kmh=60.0, conf=0.3, count_line_frac=0.55, auto_calibrate=True):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Automated Vanishing Point & Homography Estimation
    active_points = SRC_POINTS
    vp_point = None
    if auto_calibrate:
        ret, sample_frame = cap.read()
        if ret:
            active_points, vp_point = detect_lane_lines_and_vanishing_point(sample_frame)
            if vp_point is not None:
                print(f"[Auto-Calibration] Vanishing Point detected at: ({vp_point[0]}, {vp_point[1]})")
                print(f"[Auto-Calibration] Dynamically adjusted road vertices: {active_points}")
            else:
                print("[Auto-Calibration] Defaulting to baseline road vertices.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    M, calibrated_points = build_homography(active_points)
    model = YOLO("yolov8n.pt")
    cap.release()

    count_line_y = int(frame_height * count_line_frac)

    print(f"Video: {video_path} | FPS: {fps:.1f} | Total Frames: {total_video_frames} | Frame: {frame_width}x{frame_height}")
    print(f"Counting line at y={count_line_y}px | Speed limit: {speed_limit_kmh} km/h")
    print(f"Active Homography Zone: {calibrated_points}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (frame_width, frame_height))

    history_by_id = collections.defaultdict(lambda: collections.deque(maxlen=SPEED_WINDOW_FRAMES))
    smooth_speed_by_id = {}
    last_y_by_id = {}
    class_by_id = {}
    counted_ids = set()
    total_counts = collections.Counter()
    bucket_counts = collections.defaultdict(collections.Counter)
    all_speed_readings = []
    detection_rows = []

    frame_num = 0
    results_stream = model.track(
        source=video_path, conf=conf, persist=True,
        tracker="bytetrack.yaml", stream=True, verbose=False,
        classes=TARGET_COCO_CLASSES,
    )

    for r in results_stream:
        frame_num += 1
        timestamp_sec = frame_num / fps
        bucket = int(timestamp_sec // TIME_BUCKET_SECONDS) * TIME_BUCKET_SECONDS

        frame = r.orig_img.copy()

        # Draw calibration trapezoid + counting line
        pts = np.array(calibrated_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        calib_label = "Auto Calib Zone" if vp_point is not None else "Calibration Zone"
        cv2.putText(frame, calib_label, (calibrated_points[0][0], calibrated_points[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if vp_point is not None and frame_num < 60:
            cv2.circle(frame, vp_point, 5, (0, 255, 255), -1)
            cv2.putText(frame, "VP", (vp_point[0] + 6, vp_point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        cv2.line(frame, (0, count_line_y), (frame_width, count_line_y), (0, 165, 255), 2)
        cv2.putText(frame, "Count Line", (15, count_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

        minimap_positions = []
        current_frame_tracked_boxes = []

        if r.boxes.id is not None:
            for box in r.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                ground_x, ground_y = (x1 + x2) / 2, y2  # bottom-center ground contact

                detection_rows.append({
                    "frame": frame_num, "timestamp_sec": round(timestamp_sec, 2),
                    "track_id": track_id, "class": cls_name,
                    "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                })

                if cls_name not in VEHICLE_CLASSES:
                    continue
                class_by_id.setdefault(track_id, cls_name)

                world_x, world_y = to_world(M, ground_x, ground_y)
                history_by_id[track_id].append((timestamp_sec, world_x, world_y))

                # Smooth speed estimation over rolling window
                hist = history_by_id[track_id]
                if len(hist) >= 4 and cls_name in SPEED_CLASSES:
                    t_old, x_old, y_old = hist[0]
                    t_new, x_new, y_new = hist[-1]
                    dt = t_new - t_old
                    if dt > 0.1:
                        # Forward travel along the road
                        dy_m = (y_new - y_old)
                        dx_m = (x_new - x_old)
                        dist_m = (dx_m ** 2 + dy_m ** 2) ** 0.5
                        raw_speed_kmh = (dist_m / dt) * 3.6

                        # Filter physically unrealistic instantaneous spikes
                        if 5.0 <= raw_speed_kmh <= 180.0:
                            if track_id in smooth_speed_by_id:
                                # Exponential smoothing (alpha = 0.3)
                                smooth_speed = 0.3 * raw_speed_kmh + 0.7 * smooth_speed_by_id[track_id]
                            else:
                                smooth_speed = raw_speed_kmh
                            
                            smooth_speed_by_id[track_id] = smooth_speed
                            all_speed_readings.append({
                                "track_id": track_id, "class": cls_name, "frame": frame_num,
                                "timestamp_sec": round(timestamp_sec, 2),
                                "speed_kmh": round(smooth_speed, 1),
                                "over_limit": smooth_speed > speed_limit_kmh,
                            })

                speed_to_show = smooth_speed_by_id.get(track_id)
                minimap_positions.append((world_x, world_y, cls_name, speed_to_show))
                current_frame_tracked_boxes.append((x1, y1, x2, y2, track_id, cls_name, speed_to_show, ground_x, ground_y))

                # Line-crossing counting logic
                prev_y = last_y_by_id.get(track_id)
                if prev_y is not None and track_id not in counted_ids:
                    if (prev_y < count_line_y <= ground_y) or (prev_y > count_line_y >= ground_y):
                        counted_ids.add(track_id)
                        total_counts[cls_name] += 1
                        bucket_counts[bucket][cls_name] += 1
                last_y_by_id[track_id] = ground_y

        # Draw all bounding boxes with ID + Class + Speed badges
        for x1, y1, x2, y2, track_id, cls_name, speed_val, gx, gy in current_frame_tracked_boxes:
            draw_vehicle_badge(frame, x1, y1, x2, y2, track_id, cls_name, speed_val, speed_limit_kmh)
            cv2.circle(frame, (int(gx), int(gy)), 4, (255, 0, 255), -1)

        # Draw HUD dashboard overlay (top-left)
        hud_bg = frame[10:140, 10:240]
        cv2.rectangle(frame, (10, 10), (240, 140), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (240, 140), (100, 100, 100), 1)
        cv2.putText(frame, "TRAFFIC METRICS", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        
        y_off = 54
        cv2.putText(frame, f"Total Count: {len(counted_ids)}", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        for cls_name in sorted(total_counts.keys()):
            y_off += 20
            if y_off < 135:
                cv2.putText(frame, f"{cls_name}: {total_counts[cls_name]}", (25, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        # Composite top-down minimap (top-right)
        minimap = draw_minimap(minimap_positions)
        mh, mw = minimap.shape[:2]
        frame[10:10 + mh, frame_width - mw - 10:frame_width - 10] = minimap

        writer.write(frame)
        if frame_num % 50 == 0:
            print(f"Processed frame {frame_num}/{total_video_frames} ({frame_num/total_video_frames*100:.1f}%)")

    writer.release()
    print(f"\nSaved annotated_output.mp4 ({frame_num} frames with ID + Speed badges & minimap)")

    # ---------------- CSVs ----------------
    pd.DataFrame(detection_rows).to_csv("tracked_detections.csv", index=False)

    all_buckets = sorted(bucket_counts.keys())
    all_classes = sorted(VEHICLE_CLASSES)
    running = collections.Counter()
    ts_rows = []
    for b in all_buckets:
        for cls in all_classes:
            running[cls] += bucket_counts[b].get(cls, 0)
            ts_rows.append({"time_bucket_sec": b, "class": cls,
                             "count_in_bucket": bucket_counts[b].get(cls, 0),
                             "cumulative_count": running[cls]})
    pd.DataFrame(ts_rows).to_csv("traffic_counts_timeseries.csv", index=False)

    speeds_df = pd.DataFrame(all_speed_readings)
    speeds_df.to_csv("vehicle_speed_readings.csv", index=False)

    # Per-vehicle summary (avg / max speed per track)
    if len(speeds_df):
        summary = speeds_df.groupby(["track_id", "class"])["speed_kmh"].agg(
            avg_speed_kmh="mean", max_speed_kmh="max").reset_index()
        summary["over_limit"] = summary["max_speed_kmh"] > speed_limit_kmh
        summary.to_csv("vehicle_speed_summary.csv", index=False)
    else:
        summary = pd.DataFrame()

    print(f"Saved tracked_detections.csv, traffic_counts_timeseries.csv, "
          f"vehicle_speed_readings.csv, vehicle_speed_summary.csv")

    print("\nFinal unique counts (line-crossing method):")
    for cls, count in total_counts.items():
        print(f"  {cls}: {count}")
    if len(summary):
        print(f"\nPer-vehicle avg speed range: {summary['avg_speed_kmh'].min():.1f} - {summary['avg_speed_kmh'].max():.1f} km/h")
        print(f"Mean fleet speed: {summary['avg_speed_kmh'].mean():.1f} km/h")
        print(f"Vehicles exceeding {speed_limit_kmh} km/h: {int(summary['over_limit'].sum())} / {len(summary)}")

    # ---------------- Summary charts ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if total_counts:
        axes[0].bar(total_counts.keys(), total_counts.values(), color="#1F77B4")
    axes[0].set_title("Unique Vehicle Count by Class")
    axes[0].set_ylabel("Count")

    if len(summary):
        axes[1].hist(summary["avg_speed_kmh"], bins=10, color="#2CA02C", edgecolor="black")
        axes[1].axvline(speed_limit_kmh, color="red", linestyle="--", label=f"Limit ({speed_limit_kmh} km/h)")
        axes[1].legend()
    axes[1].set_title("Average Speed Distribution (per vehicle)")
    axes[1].set_xlabel("Speed (km/h)")
    axes[1].set_ylabel("Number of Vehicles")

    plt.tight_layout()
    plt.savefig("summary_charts.png", dpi=150)
    print("Saved summary_charts.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="single_way_traffic.mp4", help="Path to traffic video file")
    parser.add_argument("--speed_limit_kmh", type=float, default=60.0)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--count_line_frac", type=float, default=0.55,
                         help="Counting line as a fraction of frame height")
    args = parser.parse_args()
    main(args.video, args.speed_limit_kmh, args.conf, args.count_line_frac)

