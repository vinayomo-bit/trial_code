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

## ArUco Bot Navigation

This project now includes a complete robotic navigation system where a robot with an attached ArUco marker can autonomously move toward a target ArUco marker.

### Hardware Components

- **ESP32 WROOM** development board
- **L293D motor driver** IC  
- **2 DC motors** for differential drive
- **ArUco markers** (printed)
- **Power supply** for motors
- **Camera** for marker detection

### Files Description

#### `aruco_bot_controller.ino`
Arduino code for ESP32 that controls the robot motors via L293D driver and communicates with Python via TCP.

#### `aruco_bot_navigation.py`
Python script that detects markers and sends navigation commands to the robot via WiFi.

#### `test_esp32_communication.py`
Test script to verify TCP communication between Python and ESP32 without needing markers.

#### `BOT_SETUP_GUIDE.md`
Comprehensive hardware setup and wiring guide for the robot system.

### Usage

#### 5. Robot Navigation Setup

**Hardware Setup:**
1. Wire ESP32 to L293D motor driver according to the wiring diagram in `BOT_SETUP_GUIDE.md`
2. Update WiFi credentials in `aruco_bot_controller.ino`
3. Upload Arduino code to ESP32
4. Attach ArUco marker (ID 0) to robot
5. Print target marker (ID 2)

**Software Usage:**
```bash
# Test ESP32 communication first
python test_esp32_communication.py

# Run the navigation system
python aruco_bot_navigation.py
```

**Navigation Controls:**
- **SPACEBAR**: Start/stop autonomous navigation
- **E**: Emergency stop
- **S**: Save current frame
- **R**: Reconnect to ESP32
- **Q**: Quit application

#### 6. System Operation

1. **Setup Phase:**
   - Place robot with marker ID 0 in camera view
   - Place target marker ID 2 somewhere in the environment
   - Ensure both markers are visible to camera

2. **Navigation Phase:**
   - Robot automatically calculates direction to target
   - Moves forward when target is ahead
   - Turns left/right to align with target
   - Stops when within threshold distance

3. **Monitoring:**
   - Real-time visualization shows marker detection
   - Console displays navigation commands and status
   - Visual overlay shows connection and detection status

### Navigation Algorithm

The robot navigation uses the two-marker analysis to:

1. **Detect both markers** (bot and target) in camera feed
2. **Calculate relative position** using distance and azimuth angle
3. **Determine movement command:**
   - Forward: Target is ahead (azimuth < 15°)
   - Turn Left: Target is to the left (azimuth < -15°)
   - Turn Right: Target is to the right (azimuth > 15°)
   - Stop: Target reached (distance < 10cm)

4. **Send TCP commands** to ESP32 controller
5. **Execute movement** with differential drive motors

### Technical Specifications

#### Communication
- **Protocol**: TCP socket over WiFi
- **Port**: 8888 (configurable)
- **Commands**: forward, backward, left, right, stop
- **Update Rate**: 2 Hz (configurable)

#### Navigation Parameters
- **Distance Threshold**: 10cm (stop when reached)
- **Angle Threshold**: 15° (turn if target outside this range)
- **Max Detection Distance**: 2m (ignore distant detections)
- **Command Interval**: 0.5 seconds between movements

#### Motor Control
- **PWM Speed Control**: 0-255 range
- **Default Speed**: 200 (forward/backward)
- **Turn Speed**: 150 (differential turning)
- **Movement Duration**: 500ms forward, 300ms turns

### Applications

#### Educational Projects
- Learn robotics and computer vision integration
- Understand coordinate transformations and navigation
- Practice Arduino and Python communication

#### Research and Development
- Autonomous navigation testbed
- Multi-robot coordination experiments
- Indoor positioning system development

#### Practical Applications
- Warehouse automation proof-of-concept
- Service robot navigation
- Interactive art installations
