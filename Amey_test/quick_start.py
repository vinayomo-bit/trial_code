"""
ArUco Bot Quick Start Script

This script helps you quickly set up and test the ArUco bot navigation system.
It provides a menu-driven interface to test different components.
"""

import subprocess
import sys
import os


def print_banner():
    """Print the application banner"""
    print("=" * 60)
    print("           ARUCO BOT NAVIGATION SYSTEM")
    print("=" * 60)
    print("Complete robotic navigation using ArUco markers")
    print()


def check_files():
    """Check if all required files are present"""
    required_files = [
        "aruco_detector.py",
        "aruco_bot_navigation.py", 
        "test_esp32_communication.py",
        "aruco_bot_controller.ino",
        "generate_markers.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required files found")
        return True


def show_hardware_checklist():
    """Display hardware setup checklist"""
    print("\n📋 HARDWARE SETUP CHECKLIST")
    print("-" * 40)
    print("□ ESP32 WROOM development board")
    print("□ L293D motor driver IC")
    print("□ 2 DC motors (6V recommended)")
    print("□ Power supply (6-12V for motors)")
    print("□ Jumper wires and breadboard")
    print("□ USB camera or laptop camera")
    print("□ ArUco markers printed (at least 5cm x 5cm)")
    print("□ WiFi network for ESP32 and computer")
    print()
    print("📖 See BOT_SETUP_GUIDE.md for detailed wiring instructions")


def show_software_checklist():
    """Display software setup checklist"""
    print("\n💻 SOFTWARE SETUP CHECKLIST")
    print("-" * 40)
    print("□ Arduino IDE installed with ESP32 support")
    print("□ Python with OpenCV installed")
    print("□ ESP32 programmed with aruco_bot_controller.ino")
    print("□ WiFi credentials updated in Arduino code")
    print("□ ESP32 connected to same WiFi as computer")
    print("□ Note ESP32 IP address from Serial Monitor")


def run_marker_generation():
    """Generate ArUco markers"""
    print("\n🎯 Generating ArUco markers...")
    try:
        subprocess.run([sys.executable, "generate_markers.py"], check=True)
        print("✅ Markers generated successfully!")
        print("📁 Check the 'aruco_markers' folder")
        print("🖨️  Print markers ID 0 (bot) and ID 2 (target)")
    except subprocess.CalledProcessError:
        print("❌ Failed to generate markers")
    except FileNotFoundError:
        print("❌ generate_markers.py not found")


def test_camera():
    """Test camera access"""
    print("\n📷 Testing camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera is working")
                print("📺 Press any key to close camera test...")
                cv2.imshow("Camera Test", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("❌ Camera found but cannot read frames")
            cap.release()
        else:
            print("❌ Cannot access camera")
    except ImportError:
        print("❌ OpenCV not installed. Run: pip install opencv-python")
    except Exception as e:
        print(f"❌ Camera test failed: {e}")


def test_esp32_connection():
    """Test ESP32 TCP connection"""
    print("\n🔌 Testing ESP32 connection...")
    try:
        subprocess.run([sys.executable, "test_esp32_communication.py"])
    except FileNotFoundError:
        print("❌ test_esp32_communication.py not found")


def run_basic_detection():
    """Run basic ArUco detection"""
    print("\n🔍 Starting basic ArUco detection...")
    print("📍 Place markers in camera view and press 'q' to quit")
    try:
        subprocess.run([sys.executable, "aruco_detector.py"])
    except FileNotFoundError:
        print("❌ aruco_detector.py not found")


def run_bot_navigation():
    """Run the full bot navigation system"""
    print("\n🤖 Starting bot navigation system...")
    print("⚠️  Make sure:")
    print("   - ESP32 is powered and connected to WiFi")
    print("   - Robot has marker ID 0 attached")
    print("   - Target marker ID 2 is placed in environment")
    print("   - Both markers are visible to camera")
    
    confirm = input("\nProceed? (y/n): ")
    if confirm.lower().startswith('y'):
        try:
            subprocess.run([sys.executable, "aruco_bot_navigation.py"])
        except FileNotFoundError:
            print("❌ aruco_bot_navigation.py not found")
    else:
        print("Navigation cancelled")


def show_troubleshooting():
    """Show common troubleshooting tips"""
    print("\n🔧 TROUBLESHOOTING GUIDE")
    print("-" * 40)
    print()
    print("🔗 ESP32 Connection Issues:")
    print("   • Check WiFi credentials in Arduino code")
    print("   • Verify ESP32 and computer on same network")
    print("   • Check firewall settings (allow port 8888)")
    print("   • Restart ESP32 and check Serial Monitor")
    print()
    print("🎯 ArUco Detection Issues:")
    print("   • Improve lighting (avoid shadows/glare)")
    print("   • Use high-quality printed markers")
    print("   • Keep markers flat and unfolded")
    print("   • Maintain 30cm-1m distance from camera")
    print()
    print("⚙️  Motor Control Issues:")
    print("   • Verify wiring according to setup guide")
    print("   • Check 6-12V power supply for motors")
    print("   • Ensure common ground connections")
    print("   • Test motors individually first")
    print()
    print("📖 See BOT_SETUP_GUIDE.md for detailed troubleshooting")


def main_menu():
    """Display and handle main menu"""
    while True:
        print("\n" + "=" * 50)
        print("              MAIN MENU")
        print("=" * 50)
        print("1.  🎯 Generate ArUco markers")
        print("2.  📷 Test camera")
        print("3.  🔌 Test ESP32 communication")
        print("4.  🔍 Run basic ArUco detection")
        print("5.  🤖 Run bot navigation system")
        print("6.  📋 Show hardware checklist")
        print("7.  💻 Show software checklist") 
        print("8.  🔧 Troubleshooting guide")
        print("9.  ❌ Exit")
        print()
        
        try:
            choice = input("Select option (1-9): ").strip()
            
            if choice == "1":
                run_marker_generation()
            elif choice == "2":
                test_camera()
            elif choice == "3":
                test_esp32_connection()
            elif choice == "4":
                run_basic_detection()
            elif choice == "5":
                run_bot_navigation()
            elif choice == "6":
                show_hardware_checklist()
            elif choice == "7":
                show_software_checklist()
            elif choice == "8":
                show_troubleshooting()
            elif choice == "9":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-9.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function"""
    print_banner()
    
    print("🔍 Checking system files...")
    if not check_files():
        print("\n❌ Some files are missing. Please ensure all files are in the current directory.")
        return
    
    print("\n🚀 System ready! Choose an option from the menu below.")
    main_menu()


if __name__ == "__main__":
    main()
