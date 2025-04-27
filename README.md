# FireflyXtract

FireflyXtract is a Python-based tool for extracting telemetry data from Firefly Aerospace's landing video. It uses OCR to parse telemetry data such as velocity, altitude, and time from video streams or local files. The extracted data is processed, interpolated, and visualized for analysis. Inspired by [SpaceXtract](https://github.com/shahar603/SpaceXtract)

![Alt Text](capture_compressed.gif)

---

## Features

- Extract telemetry data from video streams or local files.
- Perform OCR on specific regions of interest (ROIs) in the video.
- Uses physical constraints like derived speed and acceleration lying in expected ranges to drop bad data
- Interpolate data to N-second intervals for smooth analysis.
- Generate plots for velocity, altitude, and acceleration (d/dt(speed)) over time.
- Save extracted data for reuse in `.npz` format.

---

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher (tested in 3.13.2)
- [FFmpeg](https://ffmpeg.org/) (must be in your system's PATH)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (required for OCR functionality)

### Install Required Python Modules

Clone the repository and navigate to the `FireflyXtract` directory:

```bash
git clone https://github.com/chandrabhraman/fireflyxtract.git
cd fireflyxtract/FireflyXtract
```

### Install the required Python modules using pip:
```bash
pip install -r [requirements.txt]
```

### Usage
To extract telemetry data, run the get_telemetry_firefly.py script with the following arguments:

```bash
python3 [get_telemetry_firefly.py] <video_url> <start_time> <end_time> <bbox_time> <bbox_vel> <bbox_alt> <bbox_alt_units> [--stream_quality <quality>] [--flight_data_path <path>]
```

Arguments
`video_url`: URL of the YouTube video or path to a local video file.  
`start_time`: Start time in the video (in seconds).  
`end_time`: End time in the video (in seconds).  
`bbox_time`: Bounding box for the "Time to Go" field (format: x,y,w,h).  
`bbox_vel`: Bounding box for the "Velocity" field (format: x,y,w,h).  
`bbox_alt`: Bounding box for the "Altitude" field (format: x,y,w,h).  
`bbox_alt_units`: Bounding box for the "Altitude Units" field (format: x,y,w,h).  

Optional Arguments  
`--stream_quality`: Desired stream quality (e.g., 720p, 1080p, best). Default is 1080p   
`--flight_data_path`: Path to save extracted flight data in .npz format. Default is flight_data.npz.

Example
```bash
python3 [get_telemetry_firefly.py] "https://www.youtube.com/watch?v=ChEuA1AUJAY" 4380 5203 "64,85,158,33" "75,192,87,35" "74,347,82,33" "103,380,30,20"
```
## Scope of improvement
Better template matching for the telemetry would help. Open to PRs.

# Credits
Credit to the Firefly Aerospace Youtube [video](https://www.youtube.com/watch?v=ChEuA1AUJAY) for Blue Ghost Mission 1.