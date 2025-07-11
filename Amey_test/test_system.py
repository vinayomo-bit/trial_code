import cv2
import numpy as np
from aruco_detector import ArucoDetector


def test_detection():
    """
    Test ArUco detection with a generated marker
    """
    print("Testing ArUco Detection System")
    print("=" * 40)
    
    # Create a test marker
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_size = 200
    marker_id = 42
    
    # Generate marker image
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
    
    # Add border
    border_size = 50
    test_image = np.ones((marker_size + 2 * border_size, 
                         marker_size + 2 * border_size), dtype=np.uint8) * 255
    test_image[border_size:border_size + marker_size, 
               border_size:border_size + marker_size] = marker_image
    
    # Convert to color
    test_image_color = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
    
    # Initialize detector
    detector = ArucoDetector()
    
    # Process the test image
    processed_image, detection_data = detector.process_frame(test_image_color)
    
    # Display results
    print(f"Test marker ID: {marker_id}")
    
    if detection_data['ids'] is not None:
        print("✓ Detection successful!")
        for orientation in detection_data['orientations']:
            print(f"\nDetected Marker ID: {orientation['id']}")
            print(f"Roll:  {orientation['roll']:.2f}°")
            print(f"Pitch: {orientation['pitch']:.2f}°")
            print(f"Yaw:   {orientation['yaw']:.2f}°")
            print(f"Position: {orientation['position']}")
    else:
        print("✗ No markers detected")
    
    # Show the test image
    cv2.imshow('Test Marker', test_image_color)
    cv2.imshow('Detection Result', processed_image)
    
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detection_data['ids'] is not None


def test_camera():
    """
    Test camera access
    """
    print("\nTesting Camera Access")
    print("=" * 25)
    
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("✓ Camera access successful!")
        
        # Test frame capture
        ret, frame = cap.read()
        if ret:
            print("✓ Frame capture successful!")
            height, width = frame.shape[:2]
            print(f"Frame size: {width}x{height}")
        else:
            print("✗ Frame capture failed!")
        
        cap.release()
        return True
    else:
        print("✗ Camera access failed!")
        print("  - Check if camera is connected")
        print("  - Check if camera is being used by another application")
        return False


def main():
    """
    Run all tests
    """
    print("ArUco Detection System Test Suite")
    print("=" * 50)
    
    # Test 1: Basic detection
    detection_test = test_detection()
    
    # Test 2: Camera access
    camera_test = test_camera()
    
    # Summary
    print("\nTest Summary")
    print("=" * 15)
    print(f"Detection Test: {'PASS' if detection_test else 'FAIL'}")
    print(f"Camera Test:    {'PASS' if camera_test else 'FAIL'}")
    
    if detection_test and camera_test:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python generate_markers.py' to create test markers")
        print("2. Print the markers on paper")
        print("3. Run 'python aruco_detector.py' for real-time detection")
    else:
        print("\n✗ Some tests failed. Please check the issues above.")


if __name__ == "__main__":
    main()
