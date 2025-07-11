# ArUco Bot Navigation Setup Guide

## Overview
This system allows a robot with an ArUco marker to autonomously navigate toward a target ArUco marker using computer vision and WiFi communication.

## Hardware Requirements

### Robot Components
- **ESP32 WROOM Development Board**
- **L293D Motor Driver IC**
- **2 DC Motors** (6V recommended)
- **ArUco Marker** (printed and attached to robot)
- **Power Supply** (6-12V for motors, USB for ESP32)
- **Jumper Wires and Breadboard**

### Additional Equipment
- **Camera** (USB webcam or laptop camera)
- **Computer** running Python
- **WiFi Network** (same network for ESP32 and computer)
- **Target ArUco Marker** (printed)

## Wiring Diagram

```
ESP32 WROOM -> L293D Motor Driver
====================================
GPIO 16     -> IN1 (Left Motor Forward)
GPIO 17     -> IN2 (Left Motor Backward)  
GPIO 18     -> IN3 (Right Motor Forward)
GPIO 19     -> IN4 (Right Motor Backward)
GPIO 21     -> ENA (Left Motor Speed Control)
GPIO 22     -> ENB (Right Motor Speed Control)
3.3V        -> VCC1 (Logic Power)
GND         -> GND

L293D -> Motors & Power
=======================
OUT1        -> Left Motor (+)
OUT2        -> Left Motor (-)
OUT3        -> Right Motor (+)
OUT4        -> Right Motor (-)
VCC2        -> External Power Supply (+) [6-12V]
GND         -> External Power Supply (-) & ESP32 GND
```

## Software Setup

### 1. Arduino IDE Setup
1. Install Arduino IDE
2. Add ESP32 board support:
   - File -> Preferences
   - Add URL: `https://dl.espressif.com/dl/package_esp32_index.json`
   - Tools -> Board -> Boards Manager -> Search "ESP32" -> Install

### 2. ESP32 Configuration
1. Open `aruco_bot_controller.ino` in Arduino IDE
2. **IMPORTANT**: Update WiFi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";           // Your WiFi name
   const char* password = "YOUR_WIFI_PASSWORD";   // Your WiFi password
   ```
3. Upload the code to ESP32
4. Open Serial Monitor (115200 baud) to see IP address

### 3. Python Environment Setup
1. Install required packages:
   ```bash
   pip install opencv-python opencv-contrib-python numpy
   ```
2. Test the setup:
   ```bash
   python aruco_bot_navigation.py
   ```

## ArUco Marker Setup

### 1. Generate Markers
```bash
python generate_markers.py
```
This creates markers in the `aruco_markers/` folder.

### 2. Print and Prepare Markers
- **Bot Marker (ID 0)**: Print at least 5cm x 5cm, attach to robot top
- **Target Marker (ID 2)**: Print at least 5cm x 5cm, place on flat surface
- Use white paper and black ink for best detection
- Laminate or protect from damage

### 3. Marker Placement
- **Bot marker**: Center of robot, facing upward
- **Target marker**: Flat surface, clearly visible to camera
- Ensure good lighting, avoid shadows and glare

## Operation Guide

### 1. Start the System
1. **Power on robot** and verify ESP32 connects to WiFi
2. **Note the IP address** from Serial Monitor
3. **Run Python script**:
   ```bash
   python aruco_bot_navigation.py
   ```
4. **Enter network settings** when prompted
5. **Verify connection** to ESP32

### 2. Navigation Controls
- **SPACEBAR**: Start/Stop navigation
- **E**: Emergency stop
- **S**: Save current frame
- **R**: Reconnect to ESP32
- **Q**: Quit application

### 3. Navigation Process
1. Place both markers in camera view
2. Press SPACEBAR to start navigation
3. Robot will automatically move toward target
4. Monitor status in video window and console

## Troubleshooting

### ESP32 Connection Issues
- **Check WiFi credentials** in Arduino code
- **Verify network connectivity** (ping ESP32 IP)
- **Check firewall settings** (allow port 8888)
- **Restart ESP32** and check Serial Monitor

### Motor Control Problems
- **Verify wiring** according to diagram
- **Check power supply** (6-12V for motors)
- **Test motors individually** using Arduino code
- **Ensure common ground** between ESP32 and motor power

### ArUco Detection Issues
- **Improve lighting** (avoid shadows and glare)
- **Check marker quality** (clear print, flat surface)
- **Adjust camera distance** (30cm - 1m optimal)
- **Clean camera lens**

### Navigation Problems
- **Calibrate camera** for accurate measurements
- **Adjust navigation parameters** in Python code
- **Check marker IDs** match configuration
- **Ensure stable marker placement**

## Configuration Parameters

### Arduino Parameters (in .ino file)
```cpp
int motorSpeed = 200;        // PWM value (0-255)
int turnSpeed = 150;         // PWM for turning (0-255)
int moveTime = 500;          // Movement duration (ms)
int turnTime = 300;          // Turn duration (ms)
```

### Python Parameters (in .py file)
```python
self.distance_threshold = 0.1  # Stop distance (meters)
self.angle_threshold = 15      # Turn angle threshold (degrees)
self.move_interval = 0.5       # Time between commands (seconds)
```

## Advanced Features

### Remote Control Commands
Send these commands via TCP to ESP32:
- `forward` or `f` - Move forward
- `backward` or `b` - Move backward  
- `left` or `l` - Turn left
- `right` or `r` - Turn right
- `stop` or `s` - Stop motors
- `speed:200` - Set motor speed (0-255)
- `status` - Get system status

### Custom Navigation Logic
Modify the `calculate_navigation_command()` function in Python to implement:
- More sophisticated pathfinding
- Obstacle avoidance
- Speed modulation based on distance
- Multi-target navigation

## Safety Considerations
- **Always have emergency stop ready** (E key)
- **Test in safe, open area** first
- **Monitor robot behavior** closely
- **Check wiring** before powering on
- **Use appropriate power ratings** for components

## Performance Tips
- Use **good quality camera** for better detection
- Ensure **stable WiFi connection** for reliable communication
- **Calibrate camera** for accurate distance measurements
- **Optimize marker size** for your operating distance
- **Use consistent lighting** for best results
