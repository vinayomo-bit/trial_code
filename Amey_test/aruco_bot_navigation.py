"""
ArUco Bot Controller - Python Script

This script detects ArUco markers and controls a robot to move toward a target marker.
The bot (marker ID 0) will navigate toward the target (marker ID 2) using TCP communication
with an ESP32 controller.

Features:
- Real-time ArUco detection
- TCP communication with ESP32
- Configurable movement parameters
- Visual feedback and logging
- Emergency stop functionality
"""

import cv2
import socket
import time
import threading
import numpy as np
import math
from aruco_detector import ArucoDetector


class ArucoBot:
    def __init__(self, bot_marker_id=0, target_marker_id=2):
        """
        Initialize the ArUco bot controller
        
        Args:
            bot_marker_id: ArUco marker ID attached to the bot
            target_marker_id: ArUco marker ID of the target to navigate to
        """
        self.detector = ArucoDetector()
        self.bot_marker_id = bot_marker_id
        self.target_marker_id = target_marker_id
        
        # TCP Connection settings
        self.esp32_ip = "192.168.1.100"  # Default IP - will be configured
        self.esp32_port = 8888
        self.socket = None
        self.connected = False
        
        # Navigation parameters
        self.distance_threshold = 0.1  # Stop when within 10cm of target
        self.angle_threshold = 15      # Degrees - turn if angle > threshold
        self.max_distance = 2.0        # Maximum detection distance (meters)
        
        # Movement timing
        self.move_interval = 0.5       # Seconds between movement commands
        self.last_command_time = 0
        
        # Status tracking
        self.bot_position = None
        self.target_position = None
        self.navigation_active = False
        self.emergency_stop = False
        
        # Logging
        self.movement_log = []
        
    def configure_network(self):
        """
        Configure network settings for ESP32 communication
        """
        print("=== ESP32 Network Configuration ===")
        print("Please enter your ESP32 settings:")
        
        # Get IP address
        default_ip = self.esp32_ip
        ip_input = input(f"ESP32 IP address (default: {default_ip}): ").strip()
        if ip_input:
            self.esp32_ip = ip_input
        
        # Get port
        port_input = input(f"ESP32 port (default: {self.esp32_port}): ").strip()
        if port_input and port_input.isdigit():
            self.esp32_port = int(port_input)
        
        print(f"Configuration set: {self.esp32_ip}:{self.esp32_port}")
        
    def connect_to_esp32(self):
        """
        Establish TCP connection to ESP32
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # 10 second timeout
            
            print(f"Connecting to ESP32 at {self.esp32_ip}:{self.esp32_port}...")
            self.socket.connect((self.esp32_ip, self.esp32_port))
            
            # Read welcome message
            response = self.socket.recv(1024).decode().strip()
            print(f"ESP32 Response: {response}")
            
            self.connected = True
            print("✓ Connected to ESP32 successfully!")
            
            # Test connection
            self.send_command("status")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to ESP32: {e}")
            print("Please check:")
            print("1. ESP32 is powered on and running the Arduino code")
            print("2. ESP32 is connected to WiFi")
            print("3. IP address and port are correct")
            print("4. Firewall is not blocking the connection")
            self.connected = False
            return False
    
    def send_command(self, command):
        """
        Send command to ESP32 via TCP
        
        Args:
            command: Command string to send
            
        Returns:
            bool: Success status
        """
        if not self.connected or not self.socket:
            return False
        
        try:
            # Send command
            self.socket.send((command + '\n').encode())
            
            # Wait for acknowledgment
            response = self.socket.recv(1024).decode().strip()
            
            # Log the command
            self.movement_log.append({
                'time': time.time(),
                'command': command,
                'response': response
            })
            
            if len(self.movement_log) > 100:  # Keep last 100 commands
                self.movement_log.pop(0)
            
            return True
            
        except Exception as e:
            print(f"Error sending command '{command}': {e}")
            self.connected = False
            return False
    
    def calculate_navigation_command(self, bot_analysis):
        """
        Calculate the movement command based on bot and target positions
        
        Args:
            bot_analysis: Analysis data from analyze_marker_pair
            
        Returns:
            command: Movement command string
        """
        if not bot_analysis:
            return "stop"
        
        distance = bot_analysis['distance']
        direction_info = bot_analysis['direction_from_1_to_2']
        azimuth = direction_info['azimuth_angle']
        
        # Check if we've reached the target
        if distance < self.distance_threshold:
            return "stop"
        
        # Check if target is too far (might be detection error)
        if distance > self.max_distance:
            return "stop"
        
        # Determine movement based on azimuth angle
        # Azimuth: 0° = straight ahead, +90° = right, -90° = left
        
        if abs(azimuth) < self.angle_threshold:
            # Target is roughly ahead - move forward
            return "forward"
        elif azimuth > self.angle_threshold:
            # Target is to the right - turn right
            return "right"
        elif azimuth < -self.angle_threshold:
            # Target is to the left - turn left
            return "left"
        else:
            return "stop"
    
    def process_navigation(self, detection_data):
        """
        Process marker detection data and send navigation commands
        
        Args:
            detection_data: Detection data from ArUco detector
        """
        if not self.navigation_active or self.emergency_stop:
            return
        
        # Check if enough time has passed since last command
        current_time = time.time()
        if current_time - self.last_command_time < self.move_interval:
            return
        
        # Analyze relationship between bot and target markers
        analysis = self.detector.analyze_marker_pair(
            self.bot_marker_id, self.target_marker_id, detection_data
        )
        
        if analysis:
            # Calculate navigation command
            command = self.calculate_navigation_command(analysis)
            
            # Send command to ESP32
            if self.send_command(command):
                print(f"Navigation: {command} | Distance: {analysis['distance']:.3f}m | "
                      f"Azimuth: {analysis['direction_from_1_to_2']['azimuth_angle']:.1f}°")
                
                self.last_command_time = current_time
                
                # Update positions for visualization
                self.bot_position = analysis['marker1_position']
                self.target_position = analysis['marker2_position']
            else:
                print("Failed to send command - connection lost")
                self.navigation_active = False
        else:
            # Can't see both markers - stop
            if current_time - self.last_command_time > 2.0:  # Send stop every 2 seconds
                self.send_command("stop")
                self.last_command_time = current_time
                print("Markers not visible - stopping")
    
    def add_navigation_overlay(self, frame, detection_data):
        """
        Add navigation information overlay to the frame
        
        Args:
            frame: Input frame
            detection_data: Detection data
            
        Returns:
            frame: Frame with navigation overlay
        """
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add status information
        y_pos = 35
        line_height = 20
        
        # Connection status
        status_color = (0, 255, 0) if self.connected else (0, 0, 255)
        cv2.putText(frame, f"ESP32: {'Connected' if self.connected else 'Disconnected'}", 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y_pos += line_height
        
        # Navigation status
        nav_color = (0, 255, 0) if self.navigation_active else (0, 255, 255)
        cv2.putText(frame, f"Navigation: {'Active' if self.navigation_active else 'Inactive'}", 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, nav_color, 2)
        y_pos += line_height
        
        # Marker detection status
        ids_list = [id_val[0] for id_val in detection_data['ids']] if detection_data['ids'] is not None else []
        bot_detected = self.bot_marker_id in ids_list
        target_detected = self.target_marker_id in ids_list
        
        bot_color = (0, 255, 0) if bot_detected else (0, 0, 255)
        target_color = (0, 255, 0) if target_detected else (0, 0, 255)
        
        cv2.putText(frame, f"Bot (ID {self.bot_marker_id}): {'Detected' if bot_detected else 'Not Found'}", 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bot_color, 1)
        y_pos += line_height
        
        cv2.putText(frame, f"Target (ID {self.target_marker_id}): {'Detected' if target_detected else 'Not Found'}", 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, target_color, 1)
        y_pos += line_height
        
        # Distance and direction if both markers are detected
        if bot_detected and target_detected:
            analysis = self.detector.analyze_marker_pair(
                self.bot_marker_id, self.target_marker_id, detection_data
            )
            if analysis:
                cv2.putText(frame, f"Distance: {analysis['distance']:.3f}m", 
                           (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += line_height
                
                # Draw navigation line
                frame = self.detector.draw_marker_relationship(frame, analysis)
        
        return frame
    
    def run_navigation(self):
        """
        Main navigation loop
        """
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print(f"\n=== ArUco Bot Navigation Started ===")
        print(f"Bot Marker ID: {self.bot_marker_id}")
        print(f"Target Marker ID: {self.target_marker_id}")
        print("\nControls:")
        print("- Press 'SPACE' to start/stop navigation")
        print("- Press 'e' for emergency stop")
        print("- Press 's' to save current frame")
        print("- Press 'r' to reconnect to ESP32")
        print("- Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame for marker detection
            processed_frame, detection_data = self.detector.process_frame(frame)
            
            # Add navigation overlay
            processed_frame = self.add_navigation_overlay(processed_frame, detection_data)
            
            # Process navigation if active
            self.process_navigation(detection_data)
            
            # Display frame
            cv2.imshow('ArUco Bot Navigation', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar
                self.navigation_active = not self.navigation_active
                self.emergency_stop = False
                if self.navigation_active:
                    print("Navigation STARTED")
                else:
                    print("Navigation STOPPED")
                    self.send_command("stop")
            elif key == ord('e'):
                self.emergency_stop = True
                self.navigation_active = False
                self.send_command("stop")
                print("EMERGENCY STOP activated")
            elif key == ord('s'):
                cv2.imwrite(f'navigation_frame_{frame_count}.jpg', processed_frame)
                print(f"Frame saved as navigation_frame_{frame_count}.jpg")
                frame_count += 1
            elif key == ord('r'):
                print("Attempting to reconnect...")
                if self.socket:
                    self.socket.close()
                self.connect_to_esp32()
        
        # Cleanup
        if self.navigation_active:
            self.send_command("stop")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.socket:
            self.socket.close()
    
    def print_movement_log(self):
        """
        Print the movement command log
        """
        print("\n=== Movement Log ===")
        for entry in self.movement_log[-10:]:  # Last 10 commands
            timestamp = time.strftime('%H:%M:%S', time.localtime(entry['time']))
            print(f"{timestamp}: {entry['command']} -> {entry['response']}")


def main():
    """
    Main function to run the ArUco bot controller
    """
    print("ArUco Bot Navigation System")
    print("=" * 40)
    
    # Get marker configuration
    try:
        bot_id = int(input("Enter bot marker ID (default: 0): ") or "0")
        target_id = int(input("Enter target marker ID (default: 2): ") or "2")
    except ValueError:
        print("Invalid input, using defaults (bot=0, target=2)")
        bot_id, target_id = 0, 2
    
    # Create bot controller
    bot = ArucoBot(bot_id, target_id)
    
    # Configure network
    bot.configure_network()
    
    # Connect to ESP32
    if not bot.connect_to_esp32():
        response = input("Continue without ESP32 connection? (y/n): ")
        if response.lower() != 'y':
            return
    
    try:
        # Run navigation
        bot.run_navigation()
        
    except KeyboardInterrupt:
        print("\nNavigation interrupted by user")
    
    finally:
        # Show movement log
        bot.print_movement_log()
        print("ArUco Bot Navigation ended")


if __name__ == "__main__":
    main()
