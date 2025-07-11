"""
Quick setup and test script for two-marker ArUco analysis

This script helps you quickly test the two-marker analysis system
and ensures all dependencies are properly installed.
"""

import sys
import subprocess
import importlib


def check_dependencies():
    """
    Check if all required dependencies are installed
    """
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True


def test_camera():
    """
    Test if camera is accessible
    """
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ Camera is working")
                return True
            else:
                print("✗ Camera found but cannot read frames")
                return False
        else:
            print("✗ Cannot access camera")
            return False
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False


def run_demo():
    """
    Run a demonstration of the two-marker analysis
    """
    print("\n" + "="*50)
    print("DEMO MENU")
    print("="*50)
    print("1. Run basic ArUco detection")
    print("2. Run two-marker analysis")
    print("3. Test algorithms with sample data")
    print("4. Generate test markers")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                print("\nStarting basic ArUco detection...")
                print("Press 'q' to quit when the window opens")
                subprocess.run([sys.executable, "aruco_detector.py"])
                break
                
            elif choice == '2':
                print("\nStarting two-marker analysis...")
                print("You'll be prompted to select marker IDs")
                subprocess.run([sys.executable, "two_marker_analysis.py"])
                break
                
            elif choice == '3':
                print("\nRunning algorithm tests...")
                subprocess.run([sys.executable, "test_two_marker_algorithm.py"])
                break
                
            elif choice == '4':
                print("\nGenerating test markers...")
                subprocess.run([sys.executable, "generate_markers.py"])
                print("Check the 'aruco_markers' folder for generated markers")
                break
                
            elif choice == '5':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error running demo: {e}")
            break


def print_instructions():
    """
    Print setup and usage instructions
    """
    print("\n" + "="*60)
    print("ARUCO TWO-MARKER ANALYSIS SETUP")
    print("="*60)
    
    print("\n📋 SETUP INSTRUCTIONS:")
    print("1. Print ArUco markers from the 'aruco_markers' folder")
    print("2. Place markers on a flat surface")
    print("3. Ensure good lighting conditions")
    print("4. Position camera to see both markers clearly")
    
    print("\n🎯 FOR BEST RESULTS:")
    print("• Use markers that are at least 5cm x 5cm when printed")
    print("• Keep markers flat and unfolded")
    print("• Maintain 30-100cm distance from camera")
    print("• Ensure markers are not too close to each other")
    print("• Use consistent lighting without glare")
    
    print("\n📊 WHAT YOU'LL MEASURE:")
    print("• Distance between marker centers (in meters)")
    print("• Relative orientation (roll, pitch, yaw differences)")
    print("• Direction from one marker to another")
    print("• Real-time tracking of marker relationships")
    
    print("\n⌨️  KEYBOARD CONTROLS:")
    print("• 'q' - Quit application")
    print("• 's' - Save current analysis")
    print("• 'h' - Show analysis history")
    print("• 'r' - Reset history")
    print("• '1-9' - Change marker IDs")


def main():
    """
    Main setup and test function
    """
    print("ArUco Two-Marker Analysis - Quick Setup")
    print("="*50)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        return
    
    # Test camera
    print("\n📷 Testing camera access...")
    if not test_camera():
        print("\n⚠️  Camera issues detected. You may need to:")
        print("• Check if another application is using the camera")
        print("• Try a different camera index")
        print("• Ensure camera permissions are granted")
        print("• Connect an external camera if needed")
        input("\nPress Enter to continue anyway...")
    
    # Print instructions
    print_instructions()
    
    # Run demo
    run_demo()


if __name__ == "__main__":
    main()
