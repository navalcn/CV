@echo off
echo ========================================================
echo Running Traffic Detection, Tracking and Speed Pipeline...
echo ========================================================

.\venv\Scripts\python.exe homography_speed_pipeline.py --video single_way_traffic.mp4 --speed_limit_kmh 60

echo.
echo ========================================================
echo Done! Output files generated:
echo - annotated_output.mp4
echo - tracked_detections.csv
echo - traffic_counts_timeseries.csv
echo - vehicle_speed_readings.csv
echo - vehicle_speed_summary.csv
echo - summary_charts.png
echo ========================================================
pause
