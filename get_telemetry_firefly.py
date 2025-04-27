import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlink # For getting the stream URL
import argparse
import subprocess # To call streamlink CLI to get the URL
import time as pytime # To avoid conflict with variable 'time'
import re # For parsing text
import pytesseract
import os
from scipy.interpolate import interp1d

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;100000" # 5 seconds 
          
def perform_ocr(roi_image):
    # Preprocessing often helps OCR
    """
    Perform OCR on a given region of interest (ROI) of an image.

    Preprocessing involves converting the image to grayscale.
    The OCR is performed using the default engine (--oem 3)
    and the single line of text (--psm 7) page segmentation mode.
    The character whitelist is restricted to numbers, some punctuation, and units of measurement.
    If the OCR fails for any reason, an empty string is returned.
    """
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)    
    # Use pytesseract to extract text
    try:
        # --psm 7 assumes a single line of text. Adjust if needed.
        # --oem 3 uses the default engine.
        # -c tessedit_char_whitelist=0123456789.:-+/kmahms Helps restrict characters
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.:-+/kmahms,'
        text = pytesseract.image_to_string(gray, config=custom_config)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return "" # Return empty string on error


# --- Data Parsing Functions ---
def parse_time_to_go(text):    
    """ Parses -HH:MM:SS or HH:MM:SS format into total seconds. """
    text = text.replace(" ", "") # Remove spaces
    match = re.match(r"(-?)(\d{1,2}):(\d{2}):(\d{2})", text)
    if match:
        sign, h, m, s = match.groups()
        seconds = int(h) * 3600 + int(m) * 60 + int(s)
        return -seconds if sign == "-" else seconds
    return None # Indicate parsing failure

def parse_velocity(text):
    # Parses velocity value from text string for the following types ("1,500 km/h" or "44 km/h" or "4 km/h") to get say (1500,44,4) and convert the value into m/s.            
    """
    Parse velocity value from text string for the following types 
    ("1,500 km/h" or "44 km/h" or "4 km/h") to get say (1500,44,4) and 
    convert the value into m/s.

    Example:
    >>> parse_velocity("1,500 km/h")
    416.6666666666667
    >>> parse_velocity("44 km/h")
    12.222222222222222
    >>> parse_velocity("4 km/h")
    1.1111111111111112
    """
    pattern = r"\s*(-?[\d,]+)\s*$"
    match = re.match(pattern, text)
    number_str_cleaned = match.group(1).replace(',', '')
    # print(number_str_cleaned)
    if match:
        try:
            vel_kmh = float(number_str_cleaned)
            vel_mps = vel_kmh * 1000.0 / 3600.0 # Convert km/h to m/s
            return vel_mps
        except ValueError:
            return None # Conversion failed
    return None

def parse_altitude_units(text):
    """ Parses altitude string (e.g., "5.2 km") into meters. """
    # Try to extract numbers, potentially with a decimal point
    pattern = r"(km|m)"
    match = re.search(pattern, text)

    if match:
        # Extract the matched group (either 'km' or 'm')
        unit = match.group(1)
        return unit


def parse_altitude(text):
    """ Parses altitude string (e.g., "5.2 km") into meters. """
    # Try to extract numbers, potentially with a decimal point
    # print(text)
    pattern = r"^\s*(\d+\.\d+)\s*$"
    match = re.match(pattern, text)    
    # print(text,match)
    # print(text)
    # pytime.sleep(15)

    if match:
        try:
            alt = float(match.group(1))           
            # print(text,alt)
            return alt
        except ValueError:
            return None # Conversion failed
    return None

def parse_bbox(bbox_str):
    """ Parses bbox string "x,y,w,h" into a tuple of integers. """
    try:
        parts = [int(p.strip()) for p in bbox_str.split(',')]
        if len(parts) == 4:
            return tuple(parts) # (x, y, width, height)
        else:
            raise ValueError("Bounding box must have 4 parts: x,y,w,h")
    except Exception as e:
        raise ValueError(f"Invalid bounding box format: '{bbox_str}'. Error: {e}")

# --- Interpolation ---
# Scipy's `scipy.interpolate.CubicSpline` is the standard way.
# Given the strict library constraints, we will use linear interpolation and note the deviation.
def interpolate_data(timestamps, values):
    """
    Interpolates data to every second using LINEAR interpolation.
    Returns new timestamps and interpolated values.
    """
    if len(timestamps) < 2:
        print("Warning: Need at least two data points for interpolation.")
        return np.array([]), np.array([])

    # Create target timestamps: every second from first to last captured time
    start_time = np.ceil(timestamps[0])
    end_time = np.floor(timestamps[-1])
    if start_time > end_time: # Handle cases with very short duration
         if len(timestamps) >=1:
             return np.array([timestamps[0]]), np.array([values[0]])
         else:
             return np.array([]), np.array([])

    interp_times = np.arange(start_time, end_time + 1, 1.0)

    # Use numpy's linear interpolation
    print("Performing LINEAR interpolation (cubic requested but requires SciPy).")
    interp_values = np.interp(interp_times, timestamps, values)

    return interp_times, interp_values

def interpolate_data_cubic(timestamps, values):
    """
    Interpolates data to every second using CUBIC interpolation.
    Requires SciPy. Falls back to linear interpolation if SciPy is not
    installed or if there are fewer than 4 data points (required for cubic).
    """
    timestamps = np.asarray(timestamps)
    values = np.asarray(values)

    if len(timestamps) < 2:
        print("Warning: Need at least two data points for any interpolation.")
        # Return original single point if only one exists
        if len(timestamps) == 1:
            return timestamps, values
        return np.array([]), np.array([]) # Return empty if zero points

    start_time = np.ceil(timestamps[0])
    end_time = np.floor(timestamps[-1])

    # Handle cases with very short duration or where ceil > floor
    if start_time > end_time:
        print(f"Warning: Cannot generate integer timestamps between ceil({timestamps[0]})={start_time} and floor({timestamps[-1]})={end_time}.")
        print("Returning original data as no full seconds were covered.")
        return timestamps, values # Or potentially just the first/last point? Depends on desired behavior.

    interp_times = np.arange(start_time, end_time + 1, 1.0)

    # Check if conditions allow for cubic interpolation
    try:
        # Create the cubic interpolation function
        # f_interp = interp1d(timestamps, values, kind='cubic', bounds_error=False, fill_value=(values[0], values[-1]))
        f_interp = interp1d(timestamps, values, kind='cubic', bounds_error=False, fill_value="extrapolate")

        # Apply the function to the target times
        interp_values = f_interp(interp_times)
        return interp_times, interp_values

    except ValueError as e:
        print(f"Error during SciPy cubic interpolation: {e}. Falling back to linear.")
        # Fall through to linear interpolation below
        interp_times, interp_values=interpolate_data(timestamps, values)
        return interp_times, interp_values
    

# --- Main Script ---

def main():
    """
    Main script entry point.
    Parses command line arguments, extracts telemetry data from a YouTube video,
    interpolates the data to 1-second intervals, and plots the velocity,
    altitude, and acceleration vs time.
    """
    parser = argparse.ArgumentParser(description="Extract telemetry data from a YouTube lunar landing video.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("start_time", type=int, help="Start time in video (seconds)")
    parser.add_argument("end_time", type=int, help="End time in video (seconds)")
    parser.add_argument("bbox_time", type=str, help="Bounding box for 'Time to Go' (x,y,w,h)")
    parser.add_argument("bbox_vel", type=str, help="Bounding box for 'Velocity' (x,y,w,h)")
    parser.add_argument("bbox_alt", type=str, help="Bounding box for 'Altitude' (x,y,w,h)")
    parser.add_argument("bbox_alt_units", type=str, help="Bounding box for 'Altitude Units' (x,y,w,h)")
    parser.add_argument("--stream_quality", default="1080p", help="Desired stream quality (e.g., 720p, 1080p, best)")
    parser.add_argument("--flight_data_path", default="flight_data.npz", help="Path to save flight data")

    args = parser.parse_args()

    # if flight_data.npz already exists, directly go to plotting section
    if not os.path.exists(args.flight_data_path):

        # Validate times
        if args.start_time >= args.end_time:
            print("Error: Start time must be less than end time.")
            return

        # Parse bounding boxes
        try:
            bbox_time = parse_bbox(args.bbox_time)
            bbox_vel = parse_bbox(args.bbox_vel)
            bbox_alt = parse_bbox(args.bbox_alt)
            bbox_alt_units = parse_bbox(args.bbox_alt_units)
            bboxes = {
                "time": bbox_time,
                "vel": bbox_vel,
                "alt": bbox_alt,
                "alt_units": bbox_alt_units
            }
        except ValueError as e:
            print(f"Error parsing bounding boxes: {e}")
            return

        print("--- Starting Telemetry Extraction ---")
        print(f"Video URL: {args.url}")
        print(f"Time Range: {args.start_time}s - {args.end_time}s")
        print(f"Bounding Boxes:")
        print(f"  Time: {bbox_time}")
        print(f"  Velocity: {bbox_vel}")
        print(f"  Altitude: {bbox_alt}")
        print("-" * 35)

        # --- Get Video Stream URL using Streamlink ---
        stream_url='fireflyM1.webm'
        if not os.path.exists(stream_url):
            # Download
            os.system(f'yt-dlp {args.url}')  
            # Rename any webm file in the current directory to fireflyM1.webm
            for file in os.listdir('.'):
                if file.endswith('.webm'):
                    os.rename(file, 'fireflyM1.webm')            

        # --- Video Processing with OpenCV ---
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"Error: Could not open video stream: {stream_url}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print("Warning: Could not determine video FPS. Assuming 30.")
            fps = 30 # Assume a default FPS if not available

        # Seek to start time
        start_frame = int(args.start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        print(f"Video FPS: {fps:.2f}. Starting at frame ~{start_frame}.")

        extracted_data = [] # List to store (timestamp_s, time_to_go_s, velocity_mps, altitude_m) tuples
        last_values = {"time": None, "vel": None, "alt": None}
        processed_frames = 0
        start_process_time = pytime.time()
        frame_time_step_sec = 5 

        while True:
            ret, frame = cap.read()
            # print(f"Frame: {frame}")

            if not ret:
                print("End of video stream reached or error reading frame.")
                break
            
            current_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_time_sec = current_pos_msec / 1000.0            
            
            print(f"Current time: {current_time_sec:.2f}s")

            # Stop if we've passed the end time
            if current_time_sec > args.end_time:
                print(f"Reached end time ({args.end_time}s).")
                break

            # Skip if before start time (useful if seeking isn't precise)
            if current_time_sec < args.start_time:
                continue

            # Skip if the current_time_sec is not lying within 1/fps of an integral multiple of frame_time_step_sec
            if abs(current_time_sec - np.floor(current_time_sec / frame_time_step_sec) * frame_time_step_sec) < 1/fps:

                processed_frames += 1
                # if processed_frames % (int(fps)*5) == 0: # Print progress every 5 seconds of video
                print(f"Processing frame at video time: {current_time_sec:.2f}s...")        

                # --- Extract ROIs ---
                rois = {}
                for key, (x, y, w, h) in bboxes.items():
                    if x+w > frame.shape[1] or y+h > frame.shape[0] or x<0 or y<0:
                        print(f"Warning: BBox {key} { (x, y, w, h)} is out of frame bounds {frame.shape[:2][::-1]}. Skipping frame.")
                        rois[key] = None # Mark as invalid for this frame
                        continue
                    rois[key] = frame[y:y+h, x:x+w]

                    # Display ROIs with label (as bbox keys) ensuring each box is shown in same colour
                    color = (0, 255, 0) # Green
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, key, (x+w+10, y +h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Display the frame with ROIs                    
                    cv2.imshow("Frame with ROIs", frame)
                    cv2.waitKey(1) # Display for a short time, comment if you don't want to see the video
                    # print(rois)                

                # --- Perform OCR and Parse ---
                current_values = {}
                try:
                    # Time to Go
                    if rois["time"] is not None:
                        ocr_text_time = perform_ocr(rois["time"])
                        current_values["time"] = parse_time_to_go(ocr_text_time)
                    else: current_values["time"] = None

                    # Velocity
                    if rois["vel"] is not None:
                        ocr_text_vel = perform_ocr(rois["vel"])
                        current_values["vel"] = parse_velocity(ocr_text_vel)
                    else: current_values["vel"] = None

                    # Altitude units
                    if rois["alt_units"] is not None:
                        ocr_text_alt = perform_ocr(rois["alt_units"])
                        alt_units_str = parse_altitude_units(ocr_text_alt)                        
                    else: alt_units_str = "km"

                    # Altitude
                    if rois["alt"] is not None:
                        ocr_text_alt = perform_ocr(rois["alt"])
                        if alt_units_str=="m":
                            current_values["alt"] = parse_altitude(ocr_text_alt)                            
                        elif alt_units_str=="km":
                            if parse_altitude(ocr_text_alt) is not None:
                                current_values["alt"] = parse_altitude(ocr_text_alt) * 1000                            
                            else: current_values["alt"] = None
                    else: 
                        current_values["alt"] = None
                        print(current_values["alt"],alt_units_str)

                except Exception as e:
                    print(f"Error during OCR/Parsing at time {current_time_sec:.2f}s: {e}")
                    current_values["time"]=None
                    current_values["vel"]=None
                    current_values["alt"]=None
                    current_values["alt_units"]=None

                    # Continue processing, but these values will be None for this frame

                print(current_values)
                # --- Store data IF any value changed AND is valid ---
                changed = True
                valid_update = True
                temp_values_to_store = {}

                try:
                    # Use new value if valid, otherwise keep last known valid value                
                    for key in ["time", "vel", "alt"]:
                        if current_values[key] is not None:
                            if last_values[key] is not None:                        
                                if current_values[key] == last_values[key]:
                                    changed = False                                    
                        elif current_values[key] is None and last_values[key] is not None:
                            # If current parse failed, reuse last known good value
                            valid_update = False
                            changed = False
                        else:
                            # If current parse failed AND no previous value exists, this update is invalid
                            valid_update = False
                            changed = False
                    
                    if changed==True and valid_update==True:
                        for key in ["time", "vel", "alt"]:
                            # Enforce physical constraints assume time is correctly read
                            if key=="vel":
                                # Current velocity minus previous velocity divided by time difference greater than 3 x lunar g (engine design constraints) , current values should be discarded
                                if last_values["vel"] is not None and current_values["vel"] is not None:
                                    if abs(current_values["vel"] - last_values["vel"]) / (current_values["time"] - last_values["time"]) > 3 * 1.62:
                                        changed = False
                                        valid_update = False
                                        print(f"Velocity change too high: {current_values['vel']:.2f} m/s")
                            elif key=="alt":
                                # Current altitude minus previous altitude divided by time difference greater than 3 x lunar g (engine design constraints) , current values should be discarded
                                if last_values["alt"] is not None and current_values["alt"] is not None:
                                    if abs(current_values["alt"] - last_values["alt"]) / (current_values["time"] - last_values["time"]) > 5000:
                                        changed = False
                                        valid_update = False
                                        print(f"Altitude change too high: {current_values['alt']:.2f} m")      

                            temp_values_to_store[key] = current_values[key]

                        # Print the temp values
                        print(f"Values={temp_values_to_store}")


                    # Store only if at least one value changed from the last stored state
                    # AND we have a valid set of values (or previously known values)
                    if changed and valid_update:
                        # Check if all keys have a non-None value before adding
                        if all(v is not None for v in temp_values_to_store.values()):
                            # Use the video time relative to the start_time argument
                            relative_time_sec = current_time_sec - args.start_time
                            extracted_data.append((
                                relative_time_sec, # Use relative time for plots
                                temp_values_to_store["time"],
                                temp_values_to_store["vel"],
                                temp_values_to_store["alt"]
                            ))
                            # Update last known values *only* when storing
                            last_values = temp_values_to_store.copy()
                            # Debug print
                            # print(f"Data logged at {relative_time_sec:.2f}s: TimeToGo={last_values['time']}, Vel={last_values['vel']:.2f}, Alt={last_values['alt']:.1f}")
                        else:
                            # This handles the initial state where some values might be None
                            # Update last_values with any newly parsed valid values, even if not storing yet
                            for key in ["time", "vel", "alt"]:
                                if temp_values_to_store[key] is not None:
                                        last_values[key] = temp_values_to_store[key]
                except:
                    continue
            
        # --- Cleanup Video Capture ---
        cap.release()
        end_process_time = pytime.time()
        print("-" * 35)
        print(f"Finished processing {processed_frames} frames in {end_process_time - start_process_time:.2f} seconds.")
        print(f"Extracted {len(extracted_data)} data points where values changed.")

        if not extracted_data:
            print("No valid data points extracted. Cannot plot.")
            return

        # --- Prepare Data for Plotting ---
        data_array = np.array(extracted_data)
        timestamps_raw = data_array[:, 0]
        # time_to_go_raw = data_array[:, 1] # Not directly plotted usually
        velocities_raw = data_array[:, 2]
        altitudes_raw = data_array[:, 3]

        # Save the above array data in case of rerun requirement as npz file
        np.savez(args.flight_data_path, timestamps_raw=timestamps_raw, velocities_raw=velocities_raw, altitudes_raw=altitudes_raw)

    # Load the existing flight data from npz file    
    data = np.load(args.flight_data_path)
    timestamps_raw = data["timestamps_raw"]
    velocities_raw = data["velocities_raw"]
    altitudes_raw = data["altitudes_raw"]

    # --- Interpolation ---
    print("Interpolating data to 1-second intervals...")
    if len(timestamps_raw) >= 2:
        # interp_times, interp_velocities = interpolate_data(timestamps_raw, velocities_raw)
        # _, interp_altitudes = interpolate_data(timestamps_raw, altitudes_raw) # Use same time base
        interp_times, interp_velocities = interpolate_data_cubic(timestamps_raw, velocities_raw)
        _, interp_altitudes = interpolate_data_cubic(timestamps_raw, altitudes_raw) # Use same time base
        
    else:
        print("Not enough data points (<2) for interpolation. Plotting raw points.")
        interp_times = timestamps_raw
        interp_velocities = velocities_raw
        interp_altitudes = altitudes_raw

    if len(interp_times) < 2:
         print("Not enough data points (<2) after potential interpolation for acceleration calculation or plotting.")
         return

    # --- Calculate Acceleration ---
    # Acceleration is the rate of change of velocity: a = dv/dt
    # We use numpy.gradient for numerical differentiation on interpolated data
    print("Calculating acceleration...")
    # np.gradient calculates the gradient using central differences (more accurate)
    # The second argument specifies the spacing of the points (our interp_times)
    acceleration = np.gradient(interp_velocities, interp_times)

    # --- Plotting ---
    print("Generating plots...")
    plt.style.use('seaborn-v0_8-darkgrid') # Nicer plot style

    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f'Lunar Lander Telemetry Analysis ({args.start_time}s - {args.end_time}s)', fontsize=16)

    # Plot 1: Velocity vs Time
    axs[0].plot(interp_times, interp_velocities, marker='.', linestyle='-', label='Velocity (Interpolated)', markersize=3)
    axs[0].plot(timestamps_raw, velocities_raw, 'o', color='red', markersize=5, label='Raw Data Points')
    axs[0].set_ylabel("Velocity (m/s)")
    axs[0].set_title("Velocity vs. Time")
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Altitude vs Time
    axs[1].plot(interp_times, interp_altitudes, marker='.', linestyle='-', label='Altitude (Interpolated)', markersize=3)
    axs[1].plot(timestamps_raw, altitudes_raw, 'o', color='red', markersize=5, label='Raw Data Points')
    axs[1].set_ylabel("Altitude (m)")
    axs[1].set_title("Altitude vs. Time")
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: Acceleration vs Time
    axs[2].plot(interp_times, acceleration, marker='.', linestyle='-', label='Acceleration (Calculated)', markersize=3, color='green')
    axs[2].set_ylabel("Acceleration (m/s²)")
    axs[2].set_title("Acceleration vs. Time (from Interpolated Velocity)")
    axs[2].set_xlabel("Time since video start (s)")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.show()

    print("--- Script Finished ---")


if __name__ == "__main__":
    
    main()

    #Describe the usage command and its argumensts  in a docstring: python3 SpaceXtract/src/get_telemetry_firefly.py "https://www.youtube.com/watch?v=ChEuA1AUJAY" 4380 5203 "64,85,158,33" "75,192,87,35" "74,347,82,33" "103,380,30,20"
    # Usage: python3 get_telemetry_firefly.py <video_url> <start_time> <end_time> <bbox_time> <bbox_vel> <bbox_alt> [<bbox_alt_units>] [--stream_quality <quality>] [--flight_data_path <path>]
    # Example:
    # python3 SpaceXtract/src/get_telemetry_firefly.py "https://www.youtube.com/watch?v=ChEuA1AUJAY" 4380 5203 "64,85,158,33" "75,192,87,35" "74,347,82,33" "103,380,30,20"
    


    # CORRECT (full descent)
    # python3 SpaceXtract/src/get_telemetry_firefly.py "https://www.youtube.com/watch?v=ChEuA1AUJAY" 3895 5203 "64,85,158,33" "75,192,87,35" "74,347,82,33"
    # CORRECT (powered descent)

    # python3 SpaceXtract/src/get_telemetry_firefly.py "https://www.youtube.com/watch?v=ChEuA1AUJAY" 4380 5203 "64,85,158,33" "75,192,87,35" "74,347,82,33" "103,380,30,20"
