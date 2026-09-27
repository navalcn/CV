"""
Calibration Frame Extractor
--------------------------------------------------------
Saves a single frame from your video with a pixel-coordinate grid
overlaid, so you can read off (x, y) pixel positions by eye for the
4 homography calibration points -- no interactive clicking needed,
works fine on Colab too.

Run:
  python extract_calibration_frame.py --video traffic.mp4 --frame 30

Then open calibration_frame.png (zoom in) and note the pixel
coordinates of 4 points on the road that form a rectangle whose
real-world width and length you know, e.g.:
  - two points marking the left/right edge of one lane (known lane width)
  - two points further down the same lane, same width apart

Requires: pip install opencv-python
"""

import argparse
import cv2


def main(video_path, frame_number, grid_spacing=100):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"Could not read frame {frame_number}. Try a lower frame number.")
        return

    h, w = frame.shape[:2]

    # Draw grid lines every `grid_spacing` pixels, with coordinate labels
    for x in range(0, w, grid_spacing):
        cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(frame, str(x), (x + 3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for y in range(0, h, grid_spacing):
        cv2.line(frame, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(frame, str(y), (3, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite("calibration_frame.png", frame)
    print(f"Saved calibration_frame.png ({w}x{h}). Green gridlines every {grid_spacing}px.")
    print("Zoom in and note the pixel (x, y) of your 4 calibration points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to traffic video file")
    parser.add_argument("--frame", type=int, default=30, help="Frame number to extract (default 30)")
    parser.add_argument("--grid", type=int, default=100, help="Grid spacing in pixels (default 100)")
    args = parser.parse_args()
    main(args.video, args.frame, args.grid)
