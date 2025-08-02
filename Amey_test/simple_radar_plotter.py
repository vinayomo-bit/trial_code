"""
Simple Ultrasonic Radar Plotter - Fixed Version

This script receives ultrasonic sensor data from ESP32 and displays it as a real-time radar plot.
Simplified version that should work reliably.

Expected data format from ESP32: "RADAR:angle,distance"
"""

import socket
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import queue
import sys

class SimpleRadarPlotter:
    def __init__(self, esp32_ip="192.168.1.100", esp32_port=8888):
        """Initialize the radar plotter"""
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        self.socket = None
        self.connected = False
        
        # Data storage
        self.data_queue = queue.Queue()
        self.angles = []
        self.distances = []
        self.current_angle = 0
        
        # Threading
        self.running = False
        self.tcp_thread = None
        
        # Plot elements
        self.fig = None
        self.ax = None
        self.sweep_line = None
        self.points = None
        
        # Statistics
        self.packets_received = 0
        self.max_distance = 200
        
    def connect_to_esp32(self):
        """Connect to ESP32"""
        try:
            print(f"Connecting to ESP32 at {self.esp32_ip}:{self.esp32_port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.esp32_ip, self.esp32_port))
            
            # Read welcome message
            welcome = self.socket.recv(1024).decode().strip()
            print(f"ESP32: {welcome}")
            
            # Send radar_on command
            self.socket.send("radar_on\n".encode())
            response = self.socket.recv(1024).decode().strip()
            print(f"Radar command response: {response}")
            
            self.connected = True
            print("✓ Connected successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            self.connected = False
            return False
    
    def tcp_listener(self):
        """Listen for radar data"""
        buffer = ""
        while self.running and self.connected:
            try:
                data = self.socket.recv(1024).decode()
                if not data:
                    break
                
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line
                
                for line in lines[:-1]:
                    line = line.strip()
                    if line.startswith("RADAR:"):
                        try:
                            # Parse: RADAR:angle,distance
                            parts = line[6:].split(',')
                            angle = float(parts[0])
                            distance = float(parts[1])
                            
                            if 0 <= angle <= 180 and 0 <= distance <= 400:
                                self.data_queue.put((angle, distance))
                                self.packets_received += 1
                                
                                # Simple debug output
                                if self.packets_received % 20 == 0:  # Every 20th packet
                                    print(f"Packets: {self.packets_received}, Angle: {angle:3.0f}°, Distance: {distance:5.1f}cm")
                        except:
                            pass
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"TCP error: {e}")
                break
        
        print("TCP listener stopped")
        self.connected = False
    
    def setup_plot(self):
        """Setup the radar plot"""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Configure polar plot
        self.ax.set_theta_zero_location('N')  # 0° at top
        self.ax.set_theta_direction(-1)       # Clockwise
        self.ax.set_ylim(0, self.max_distance)
        self.ax.set_thetalim(0, np.pi)        # 0 to 180 degrees
        
        # Add range circles and labels
        for r in range(50, self.max_distance + 1, 50):
            circle = plt.Circle((0, 0), r, fill=False, color='gray', alpha=0.3, linestyle='--')
            self.ax.add_patch(circle)
            self.ax.text(0, r, f'{r}cm', ha='center', va='center', fontsize=8, color='gray')
        
        # Style
        self.ax.set_facecolor('black')
        self.ax.grid(True, color='green', alpha=0.3)
        self.ax.set_title('Ultrasonic Radar - Real-time', fontsize=14, color='white', pad=20)
        
        # Initialize plot elements
        self.sweep_line, = self.ax.plot([], [], 'r-', linewidth=2, alpha=0.8)
        self.points, = self.ax.plot([], [], 'yo', markersize=6, alpha=0.8)
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.98, '', fontsize=9, color='white',
                                        verticalalignment='top',
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        """Update the radar display"""
        # Process new data
        new_angles = []
        new_distances = []
        
        # Get all available data points
        while not self.data_queue.empty():
            try:
                angle, distance = self.data_queue.get_nowait()
                new_angles.append(angle)
                new_distances.append(distance)
                self.current_angle = angle
            except queue.Empty:
                break
        
        # Update data arrays
        if new_angles:
            self.angles.extend(new_angles)
            self.distances.extend(new_distances)
            
            # Keep only recent data (last 500 points)
            if len(self.angles) > 500:
                self.angles = self.angles[-500:]
                self.distances = self.distances[-500:]
        
        # Update sweep line
        if hasattr(self, 'current_angle'):
            sweep_angle_rad = np.radians(self.current_angle)
            self.sweep_line.set_data([sweep_angle_rad, sweep_angle_rad], [0, self.max_distance])
        
        # Update detection points
        if self.angles and self.distances:
            angles_rad = [np.radians(a) for a in self.angles]
            self.points.set_data(angles_rad, self.distances)
        
        # Update status
        status_info = [
            f"Status: {'Connected' if self.connected else 'Disconnected'}",
            f"Packets: {self.packets_received}",
            f"Points: {len(self.angles)}",
            f"Current: {self.current_angle:.0f}°"
        ]
        self.status_text.set_text('\n'.join(status_info))
        
        return [self.sweep_line, self.points]
    
    def start(self):
        """Start the radar system"""
        print("Simple Radar Plotter Starting...")
        print(f"Target: {self.esp32_ip}:{self.esp32_port}")
        
        # Get IP if needed
        if self.esp32_ip == "192.168.1.100":
            ip_input = input("Enter ESP32 IP address (or press Enter for default): ").strip()
            if ip_input:
                self.esp32_ip = ip_input
        
        # Connect to ESP32
        if not self.connect_to_esp32():
            choice = input("Continue without connection for demo? (y/n): ")
            if choice.lower() != 'y':
                return
            self.simulate_data()
        
        # Start TCP thread
        self.running = True
        if self.connected:
            self.tcp_thread = threading.Thread(target=self.tcp_listener)
            self.tcp_thread.daemon = True
            self.tcp_thread.start()
        
        # Setup and show plot
        self.setup_plot()
        
        print("\n=== Controls ===")
        print("- Close window to exit")
        print("- Watch console for connection status")
        
        try:
            ani = animation.FuncAnimation(self.fig, self.update_plot, 
                                        interval=100, blit=False, repeat=True)
            plt.show()
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            self.stop()
    
    def simulate_data(self):
        """Simulate radar data for testing"""
        def sim_thread():
            angle = 0
            direction = 1
            while self.running:
                # Simulate sweep
                distance = 50 + 30 * np.sin(np.radians(angle * 2))  # Fake objects
                self.data_queue.put((angle, distance))
                
                angle += direction * 5
                if angle >= 180:
                    angle = 180
                    direction = -1
                elif angle <= 0:
                    angle = 0
                    direction = 1
                
                time.sleep(0.1)
        
        print("Starting simulation mode...")
        sim_thread = threading.Thread(target=sim_thread)
        sim_thread.daemon = True
        sim_thread.start()
    
    def stop(self):
        """Stop the radar system"""
        print("Stopping radar...")
        self.running = False
        
        if self.socket:
            try:
                self.socket.send("radar_off\n".encode())
                self.socket.close()
            except:
                pass
        
        print("Radar stopped")

def main():
    """Main function"""
    print("Simple Ultrasonic Radar Plotter")
    print("=" * 35)
    
    # Check matplotlib
    try:
        import matplotlib
        print(f"Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("Error: matplotlib not installed!")
        print("Install with: pip install matplotlib")
        return
    
    # Create and start radar
    radar = SimpleRadarPlotter()
    
    try:
        radar.start()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program ended")

if __name__ == "__main__":
    main()
