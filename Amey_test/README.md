# ArUco Marker Detection and Orientation

This project provides a comprehensive solution for detecting ArUco markers and determining their orientation using OpenCV and Python.

## Features

- **Real-time ArUco marker detection** from webcam
- **Pose estimation** with 6DOF (position and orientation)
- **Euler angle calculation** (roll, pitch, yaw)
- **Visual feedback** with coordinate axes and text overlays
- **Image and video processing** capabilities
- **Marker generation** tools for testing
- **Distance calculation** between any two detected markers
- **Relative orientation analysis** (roll, pitch, yaw differences)
- **Direction vector computation** with azimuth and elevation angles
- **Real-time visualization** of marker relationships
- **Comprehensive analysis reports** with detailed metrics

## Files Description

### `aruco_detector.py`
Main detection class with comprehensive ArUco marker detection and pose estimation capabilities.

### `generate_markers.py` 
Script to generate ArUco markers and marker boards for testing.

### `process_media.py`
Command-line tool for processing images and videos.

### `two_marker_analysis.py`
Dedicated script for real-time two-marker relationship analysis with visual feedback.

### `test_two_marker_algorithm.py`
Test script demonstrating the mathematical algorithms with example data and verification.

## Installation

Make sure you have the required packages installed:

```bash
pip install opencv-python opencv-contrib-python numpy
```

## Usage

### 1. Generate Test Markers

First, generate some ArUco markers for testing:

```bash
python generate_markers.py
```

This will create:
- Individual marker images in the `aruco_markers/` directory
- A marker board image (`aruco_board.png`)

Print these markers on paper for testing.

### 2. Real-time Detection from Webcam

Run the main detector script:

```bash
python aruco_detector.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save current frame
- Press `c` to print detection data to console

### 3. Process Images or Videos

Process a single image:
```bash
python process_media.py image.jpg -o output_image.jpg
```

Process a video:
```bash
python process_media.py video.mp4 -o output_video.mp4
```

Auto-detect file type:
```bash
python process_media.py input_file.jpg
```

### 4. Two-Marker Analysis

Run the dedicated two-marker analysis tool:

```bash
python two_marker_analysis.py
```

This will:
- Prompt you to select which two markers to track
- Show real-time distance and orientation relationships
- Provide visual indicators connecting the markers
- Display comprehensive analysis data

**Controls:**
- Press 'q' to quit
- Press 's' to save current analysis and screenshot
- Press 'h' to show analysis history and statistics
- Press 'r' to reset analysis history
- Press '1-9' to change the first marker ID
- Press 'SHIFT + 1-9' to change the second marker ID

#### 5. Test the Algorithm

Test the mathematical algorithms with sample data:

```bash
python test_two_marker_algorithm.py
```

This will run verification tests and demonstrate the mathematical concepts.

## Understanding the Output

For each detected marker, the system provides:

### Position Information
- **X, Y, Z coordinates** in meters relative to the camera
- **Translation vector** showing marker position in 3D space

### Orientation Information
- **Roll**: Rotation around X-axis (red axis)
- **Pitch**: Rotation around Y-axis (green axis)  
- **Yaw**: Rotation around Z-axis (blue axis)
- All angles are in degrees

### Visual Indicators
- **Green rectangle**: Detected marker boundary
- **Coordinate axes**: 
  - Red line: X-axis
  - Green line: Y-axis
  - Blue line: Z-axis
- **Text overlay**: Marker ID, orientation angles, and position

### Two-Marker Analysis Output

When analyzing two markers, you'll see output like:
```
Markers: 0 <-> 1
Distance: 0.2847 meters
Azimuth: 45.3°
Elevation: -12.7°

Relative Orientation:
  Roll: 15.2°
  Pitch: -8.9°
  Yaw: 23.4°
```

This indicates:
- Markers 0 and 1 are 28.47cm apart
- Marker 1 is 45.3° clockwise from marker 0 in the XY plane
- Marker 1 is 12.7° below marker 0
- The relative rotation between markers shows how much marker 1 is rotated compared to marker 0

## Camera Calibration

For accurate pose estimation, you should calibrate your camera and update the camera matrix and distortion coefficients in `aruco_detector.py`:

```python
# Replace these with your camera's calibration parameters
self.camera_matrix = np.array([[800, 0, 320],
                             [0, 800, 240],
                             [0, 0, 1]], dtype=np.float32)

self.dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
```

## Customization

### Dictionary Types
You can change the ArUco dictionary type:
- `cv2.aruco.DICT_4X4_50`
- `cv2.aruco.DICT_4X4_100`
- `cv2.aruco.DICT_5X5_50`
- `cv2.aruco.DICT_6X6_250` (default)
- `cv2.aruco.DICT_7X7_1000`

### Marker Size
Adjust the `marker_size` parameter (in meters) to match your printed markers:
```python
detector = ArucoDetector(marker_size=0.05)  # 5cm markers
```

## Troubleshooting

### No Markers Detected
1. Ensure good lighting conditions
2. Check that markers are printed clearly with good contrast
3. Make sure markers are not too far from the camera
4. Verify the correct dictionary type is being used

### Inaccurate Pose Estimation
1. Calibrate your camera properly
2. Ensure marker size is set correctly
3. Keep markers flat and undeformed
4. Maintain appropriate distance from camera

### Poor Detection Performance
1. Increase image resolution
2. Improve lighting conditions
3. Reduce camera motion
4. Use larger marker sizes

## Example Output

When a marker is detected, you'll see output like:
```
Marker ID 0:
  Roll:  -12.5°
  Pitch: 8.3°
  Yaw:   45.2°
  Position: (0.150, -0.080, 0.420)
```

This indicates:
- Marker with ID 0 is detected
- Rotated -12.5° around X-axis
- Rotated 8.3° around Y-axis  
- Rotated 45.2° around Z-axis
- Located at position (15cm, -8cm, 42cm) from camera

## Applications

This code can be used for:
- **Augmented Reality** applications
- **Robot navigation** and localization
- **Camera pose estimation**
- **Object tracking** and manipulation
- **3D reconstruction** projects
- **Educational** computer vision projects
- **Robotic calibration** - measuring precise distances between reference points
- **Object tracking** - monitoring relative movement of connected objects
- **Assembly verification** - ensuring correct positioning of components
- **Quality control** - measuring manufacturing tolerances
