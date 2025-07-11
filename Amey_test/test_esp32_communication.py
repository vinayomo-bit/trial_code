"""
TCP Communication Test Script

This script tests the TCP connection between Python and ESP32
without needing ArUco detection. Use this to verify your setup.
"""

import socket
import time
import threading


class ESP32Tester:
    def __init__(self):
        self.socket = None
        self.connected = False
        
    def connect(self, ip, port):
        """Connect to ESP32"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            
            print(f"Connecting to {ip}:{port}...")
            self.socket.connect((ip, port))
            
            # Read welcome message
            response = self.socket.recv(1024).decode().strip()
            print(f"ESP32 says: {response}")
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def send_command(self, command):
        """Send command to ESP32"""
        if not self.connected:
            print("Not connected!")
            return False
            
        try:
            self.socket.send((command + '\n').encode())
            response = self.socket.recv(1024).decode().strip()
            print(f"Sent: {command} -> Response: {response}")
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            self.connected = False
            return False
    
    def interactive_test(self):
        """Interactive command testing"""
        print("\n=== Interactive Test Mode ===")
        print("Available commands:")
        print("- forward, backward, left, right, stop")
        print("- speed:150 (set speed 0-255)")
        print("- status (get robot status)")
        print("- test (run motor test)")
        print("- quit (exit)")
        
        while self.connected:
            try:
                command = input("\nEnter command: ").strip()
                
                if command.lower() == 'quit':
                    break
                    
                if command:
                    self.send_command(command)
                    
            except KeyboardInterrupt:
                break
    
    def auto_test(self):
        """Automated test sequence"""
        print("\n=== Automated Test Sequence ===")
        
        test_commands = [
            ("status", "Get robot status"),
            ("speed:150", "Set moderate speed"),
            ("forward", "Move forward"),
            ("stop", "Stop"),
            ("left", "Turn left"),
            ("stop", "Stop"),
            ("right", "Turn right"),  
            ("stop", "Stop"),
            ("backward", "Move backward"),
            ("stop", "Final stop")
        ]
        
        for command, description in test_commands:
            print(f"\n{description}...")
            if not self.send_command(command):
                print("Test failed - connection lost")
                break
            time.sleep(2)  # Wait between commands
        
        print("\nAutomated test complete!")
    
    def disconnect(self):
        """Disconnect from ESP32"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("Disconnected from ESP32")


def main():
    """Main test function"""
    print("ESP32 TCP Communication Tester")
    print("=" * 40)
    
    # Get connection details
    ip = input("Enter ESP32 IP address: ").strip()
    if not ip:
        print("No IP address provided")
        return
    
    port_input = input("Enter port (default 8888): ").strip()
    port = int(port_input) if port_input.isdigit() else 8888
    
    # Create tester and connect
    tester = ESP32Tester()
    
    if not tester.connect(ip, port):
        print("Failed to connect. Please check:")
        print("1. ESP32 is powered on")
        print("2. ESP32 is connected to WiFi") 
        print("3. IP address is correct")
        print("4. Arduino code is running")
        return
    
    try:
        print("\nConnection successful!")
        print("\nTest options:")
        print("1. Interactive mode (manual commands)")
        print("2. Automated test sequence")
        
        choice = input("Choose option (1 or 2): ").strip()
        
        if choice == "1":
            tester.interactive_test()
        elif choice == "2":
            tester.auto_test()
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nTest interrupted")
        
    finally:
        tester.disconnect()


if __name__ == "__main__":
    main()
