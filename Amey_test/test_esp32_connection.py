"""
ESP32 Radar Connection Test

Simple script to test connection to ESP32 and receive radar data
without any plotting - just to verify the communication works.
"""

import socket
import time

def test_esp32_connection(ip="192.168.1.100", port=8888):
    """Test basic connection to ESP32"""
    print(f"Testing connection to {ip}:{port}")
    
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        # Connect
        print("Connecting...")
        sock.connect((ip, port))
        
        # Read welcome message
        welcome = sock.recv(1024).decode().strip()
        print(f"ESP32 Welcome: {welcome}")
        
        # Send radar_on command
        print("Sending radar_on command...")
        sock.send("radar_on\n".encode())
        
        # Read response
        response = sock.recv(1024).decode().strip()
        print(f"Command response: {response}")
        
        # Receive radar data for 10 seconds
        print("\nReceiving radar data for 10 seconds...")
        start_time = time.time()
        packet_count = 0
        
        buffer = ""
        while time.time() - start_time < 10:
            try:
                data = sock.recv(1024).decode()
                if not data:
                    break
                
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line
                
                for line in lines[:-1]:
                    line = line.strip()
                    if line.startswith("RADAR:"):
                        packet_count += 1
                        parts = line[6:].split(',')
                        if len(parts) == 2:
                            angle = parts[0]
                            distance = parts[1]
                            print(f"Packet {packet_count:3d}: Angle={angle:>3s}°, Distance={distance:>6s}cm")
                        
                        if packet_count >= 50:  # Stop after 50 packets
                            break
                
                if packet_count >= 50:
                    break
                    
            except socket.timeout:
                print("Timeout waiting for data...")
                break
        
        # Send radar_off command
        print("\nSending radar_off command...")
        sock.send("radar_off\n".encode())
        response = sock.recv(1024).decode().strip()
        print(f"Stop response: {response}")
        
        sock.close()
        print(f"\n✓ Test completed! Received {packet_count} radar packets")
        return True
        
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False

def main():
    """Main test function"""
    print("ESP32 Radar Connection Test")
    print("=" * 30)
    
    # Get IP address
    ip = input("Enter ESP32 IP address (or press Enter for 192.168.1.100): ").strip()
    if not ip:
        ip = "192.168.1.100"
    
    # Test connection
    success = test_esp32_connection(ip)
    
    if success:
        print("\n🎉 Connection test PASSED!")
        print("Your ESP32 is working correctly.")
        print("You can now try the radar plotter.")
    else:
        print("\n❌ Connection test FAILED!")
        print("Check:")
        print("1. ESP32 is powered on")
        print("2. ESP32 is connected to WiFi")
        print("3. IP address is correct")
        print("4. Firewall is not blocking connection")

if __name__ == "__main__":
    main()
