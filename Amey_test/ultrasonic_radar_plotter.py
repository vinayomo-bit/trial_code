"""
Ultrasonic Radar Plotter - Real-time Data Visualization

This script receives ultrasonic sensor data with servo position from ESP32
and displays it as a real-time radar plot.

Expected data format from ESP32:
"RADAR:angle,distance" (e.g., "RADAR:90,25.5")

Features:
- Real-time polar plot (radar style)
- TCP connection to ESP32
- Configurable parameters
- Data logging capability
- Interactive controls
"""

import socket
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import queue
import json
from datetime import datetime


class UltrasonicRadarPlotter:
    def __init__(self, esp32_ip="192.168.0.146", esp32_port=8888):
        """
        Initialize the radar plotter
        
        Args:
            esp32_ip: IP address of ESP32 with ultrasonic sensor
            esp32_port: TCP port for communication
        """
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        self.socket = None
        self.connected = False
        
        # Radar parameters
        self.max_distance = 200  # Maximum detection range in cm
        self.min_distance = 2    # Minimum detection range in cm
        self.max_angle = 180     # Maximum servo angle
        self.min_angle = 0       # Minimum servo angle
        
        # Data storage
        self.data_queue = queue.Queue()
        self.radar_data = {}  # Dictionary: angle -> distance
        self.sweep_history = []  # Store recent sweeps for trails
        self.max_history = 3     # Number of sweep trails to show
        
        # Threading
        self.running = False
        self.tcp_thread = None
        
        # Plot setup
        self.fig = None
        self.ax = None
        self.line_objects = []
        self.text_objects = []
        
        # Statistics
        self.packets_received = 0
        self.last_update_time = time.time()
        
        # Data logging
        self.log_data = []
        self.enable_logging = True
    
    def connect_to_esp32(self):
        """
        Establish TCP connection to ESP32
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            
            print(f"Connecting to ESP32 at {self.esp32_ip}:{self.esp32_port}...")
            self.socket.connect((self.esp32_ip, self.esp32_port))
            
            # Read welcome message
            response = self.socket.recv(1024).decode().strip()
            print(f"ESP32 Response: {response}")
            
            self.connected = True
            print("✓ Connected to ESP32 radar successfully!")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to ESP32: {e}")
            self.connected = False
            return False
    
    def tcp_listener(self):
        """
        Listen for TCP data from ESP32 in separate thread
        """
        while self.running and self.connected:
            try:
                if self.socket:
                    # Receive data
                    data = self.socket.recv(1024).decode().strip()
                    
                    if data:
                        # Process multiple lines if received together
                        lines = data.split('\n')
                        for line in lines:
                            if line.strip():
                                self.process_radar_data(line.strip())
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"TCP listener error: {e}")
                self.connected = False
                break
        
        print("TCP listener stopped")
    
    def process_radar_data(self, data_line):
        """
        Process incoming radar data
        
        Expected format: "RADAR:angle,distance" or "RADAR:90,25.5"
        """
        try:
            if data_line.startswith("RADAR:"):
                # Extract radar data
                radar_part = data_line[6:]  # Remove "RADAR:" prefix
                angle_str, distance_str = radar_part.split(',')
                
                angle = float(angle_str)
                distance = float(distance_str)
                
                # Validate data
                if (self.min_angle <= angle <= self.max_angle and 
                    self.min_distance <= distance <= self.max_distance):
                    
                    # Store data
                    timestamp = time.time()
                    radar_point = {
                        'angle': angle,
                        'distance': distance,
                        'timestamp': timestamp
                    }
                    
                    # Add to queue for plotting
                    self.data_queue.put(radar_point)
                    
                    # Update statistics
                    self.packets_received += 1
                    self.last_update_time = timestamp
                    
                    # Log data if enabled
                    if self.enable_logging:
                        self.log_data.append(radar_point)
                    
                    # Debug output
                    print(f"Radar: Angle={angle:6.1f}°, Distance={distance:6.1f}cm")
                
                else:
                    print(f"Invalid radar data: angle={angle}, distance={distance}")
            
            else:
                # Handle other ESP32 messages
                print(f"ESP32 Message: {data_line}")
        
        except Exception as e:
            print(f"Error processing radar data '{data_line}': {e}")
    
    def setup_plot(self):
        """
        Setup the radar plot
        """
        # Create figure and polar subplot
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='polar')
        
        # Configure polar plot
        self.ax.set_theta_zero_location('N')  # 0 degrees at top
        self.ax.set_theta_direction(-1)       # Clockwise
        self.ax.set_ylim(0, self.max_distance)
        
        # Set angle limits and labels
        self.ax.set_thetalim(np.radians(self.min_angle), np.radians(self.max_angle))
        
        # Add range circles
        for r in range(50, self.max_distance + 1, 50):
            circle = Circle((0, 0), r, fill=False, color='gray', alpha=0.3, linestyle='--')
            self.ax.add_patch(circle)
            self.ax.text(0, r, f'{r}cm', ha='center', va='bottom', fontsize=8, color='gray')
        
        # Style the plot
        self.ax.set_facecolor('black')
        self.ax.grid(True, color='green', alpha=0.3)
        self.ax.set_title('Ultrasonic Radar - Real-time View', 
                         fontsize=16, fontweight='bold', color='white', pad=20)
        
        # Initialize plot elements
        self.line_objects = []
        self.text_objects = []
        
        # Add sweep line (current servo position)
        self.sweep_line, = self.ax.plot([], [], 'r-', linewidth=2, alpha=0.8, label='Current Sweep')
        
        # Add detection points
        self.detection_points, = self.ax.plot([], [], 'yo', markersize=8, alpha=0.9, label='Detections')
        
        # Add trail points (previous sweeps)
        self.trail_points, = self.ax.plot([], [], 'go', markersize=4, alpha=0.5, label='Trail')
        
        # Add legend
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # Add status text
        self.status_text = self.fig.text(0.02, 0.95, '', fontsize=10, color='white', 
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        """
        Update the radar plot (called by animation)
        """
        # Process new data from queue
        new_points = []
        current_angle = None
        
        while not self.data_queue.empty():
            try:
                radar_point = self.data_queue.get_nowait()
                angle = radar_point['angle']
                distance = radar_point['distance']
                
                # Store in radar data
                self.radar_data[angle] = distance
                current_angle = angle
                
                new_points.append((angle, distance))
                
            except queue.Empty:
                break
        
        # Update sweep line (current servo position)
        if current_angle is not None:
            # Draw line from center to max range
            sweep_angles = [np.radians(current_angle), np.radians(current_angle)]
            sweep_distances = [0, self.max_distance]
            self.sweep_line.set_data(sweep_angles, sweep_distances)
        
        # Update detection points
        if self.radar_data:
            angles = [np.radians(a) for a in self.radar_data.keys()]
            distances = list(self.radar_data.values())
            self.detection_points.set_data(angles, distances)
        
        # Update trail (fade older points)
        self.update_trail()
        
        # Update status text
        self.update_status_text()
        
        return [self.sweep_line, self.detection_points, self.trail_points]
    
    def update_trail(self):
        """
        Update trail points for visual effect
        """
        # Add current radar data to history
        if self.radar_data:
            self.sweep_history.append(dict(self.radar_data))
        
        # Limit history size
        if len(self.sweep_history) > self.max_history:
            self.sweep_history.pop(0)
        
        # Combine trail points
        trail_angles = []
        trail_distances = []
        
        for i, sweep in enumerate(self.sweep_history[:-1]):  # Exclude current sweep
            alpha_factor = (i + 1) / len(self.sweep_history)  # Fade older sweeps
            for angle, distance in sweep.items():
                trail_angles.append(np.radians(angle))
                trail_distances.append(distance)
        
        self.trail_points.set_data(trail_angles, trail_distances)
    
    def update_status_text(self):
        """
        Update status information display
        """
        current_time = time.time()
        uptime = current_time - self.last_update_time if self.last_update_time else 0
        
        status_info = [
            f"Connection: {'✓ Connected' if self.connected else '✗ Disconnected'}",
            f"Packets Received: {self.packets_received}",
            f"Data Points: {len(self.radar_data)}",
            f"Last Update: {uptime:.1f}s ago",
            f"Range: {self.min_distance}-{self.max_distance}cm",
            f"Sweep: {self.min_angle}-{self.max_angle}°"
        ]
        
        self.status_text.set_text('\n'.join(status_info))
    
    def save_radar_data(self, filename=None):
        """
        Save collected radar data to JSON file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"radar_data_{timestamp}.json"
        
        try:
            data_to_save = {
                'metadata': {
                    'esp32_ip': self.esp32_ip,
                    'esp32_port': self.esp32_port,
                    'max_distance': self.max_distance,
                    'packets_received': self.packets_received,
                    'recording_time': datetime.now().isoformat()
                },
                'radar_points': self.log_data
            }
            
            with open(filename, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            print(f"✓ Radar data saved to {filename}")
            
        except Exception as e:
            print(f"✗ Error saving radar data: {e}")
    
    def configure_radar(self):
        """
        Configure radar parameters
        """
        print("\n=== Radar Configuration ===")
        
        # ESP32 connection
        ip_input = input(f"ESP32 IP address (default: {self.esp32_ip}): ").strip()
        if ip_input:
            self.esp32_ip = ip_input
        
        port_input = input(f"ESP32 port (default: {self.esp32_port}): ").strip()
        if port_input and port_input.isdigit():
            self.esp32_port = int(port_input)
        
        # Radar range
        range_input = input(f"Maximum range in cm (default: {self.max_distance}): ").strip()
        if range_input and range_input.isdigit():
            self.max_distance = int(range_input)
        
        print(f"Configuration: {self.esp32_ip}:{self.esp32_port}, Range: {self.max_distance}cm")
    
    def start_radar(self):
        """
        Start the radar plotting system
        """
        print("Ultrasonic Radar Plotter Starting...")
        
        # Connect to ESP32
        if not self.connect_to_esp32():
            response = input("Continue without connection for testing? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Start TCP listener thread
        self.running = True
        if self.connected:
            self.tcp_thread = threading.Thread(target=self.tcp_listener)
            self.tcp_thread.daemon = True
            self.tcp_thread.start()
        
        # Setup and start plot
        self.setup_plot()
        
        print("\n=== Radar Controls ===")
        print("- Close plot window to stop")
        print("- Data is automatically logged")
        print("- Press Ctrl+C in terminal for emergency stop")
        
        try:
            # Start animation
            ani = animation.FuncAnimation(
                self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False
            )
            
            plt.show()
            
        except KeyboardInterrupt:
            print("\nRadar stopped by user")
        
        finally:
            self.stop_radar()
    
    def stop_radar(self):
        """
        Stop the radar system and cleanup
        """
        print("Stopping radar system...")
        
        # Stop threads
        self.running = False
        
        # Close connection
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        # Save data
        if self.log_data:
            self.save_radar_data()
        
        print("Radar system stopped")


def main():
    """
    Main function to run the radar plotter
    """
    print("Ultrasonic Radar Plotter - Real-time Visualization")
    print("=" * 50)
    
    # Create radar plotter
    radar = UltrasonicRadarPlotter()
    
    # Configure if needed
    config_response = input("Configure radar settings? (y/n, default: n): ")
    if config_response.lower() == 'y':
        radar.configure_radar()
    
    try:
        # Start radar
        radar.start_radar()
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        print("Radar plotter ended")


if __name__ == "__main__":
    main()
