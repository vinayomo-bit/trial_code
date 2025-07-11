"""
Test script for two-marker distance and orientation algorithm

This script demonstrates the mathematical algorithms used for calculating
distance and relative orientation between ArUco markers with example data.
"""

import numpy as np
import cv2
import math
from aruco_detector import ArucoDetector


def create_test_data():
    """
    Create sample marker detection data for testing
    """
    # Sample translation vectors (positions in 3D space)
    tvec1 = np.array([[[0.1, 0.2, 0.5]]], dtype=np.float32)  # Marker 1 position
    tvec2 = np.array([[[0.4, 0.3, 0.6]]], dtype=np.float32)  # Marker 2 position
    
    # Sample rotation vectors (orientations)
    # These represent rotations in radians around x, y, z axes
    rvec1 = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)  # Marker 1 orientation
    rvec2 = np.array([[[0.2, 0.4, 0.1]]], dtype=np.float32)  # Marker 2 orientation
    
    return tvec1, tvec2, rvec1, rvec2


def test_distance_calculation():
    """
    Test the distance calculation algorithm
    """
    print("=== Testing Distance Calculation ===")
    
    # Create test data
    tvec1, tvec2, _, _ = create_test_data()
    
    # Initialize detector
    detector = ArucoDetector()
    
    # Calculate distance
    distance = detector.calculate_distance_between_markers(tvec1, tvec2)
    
    # Manual calculation for verification
    pos1 = tvec1[0][0]
    pos2 = tvec2[0][0]
    manual_distance = np.sqrt((pos2[0] - pos1[0])**2 + 
                             (pos2[1] - pos1[1])**2 + 
                             (pos2[2] - pos1[2])**2)
    
    print(f"Marker 1 Position: {pos1}")
    print(f"Marker 2 Position: {pos2}")
    print(f"Distance (algorithm): {distance:.6f} meters")
    print(f"Distance (manual): {manual_distance:.6f} meters")
    print(f"Difference: {abs(distance - manual_distance):.10f} meters")
    print()


def test_direction_calculation():
    """
    Test the direction vector calculation
    """
    print("=== Testing Direction Vector Calculation ===")
    
    # Create test data
    tvec1, tvec2, _, _ = create_test_data()
    
    # Initialize detector
    detector = ArucoDetector()
    
    # Calculate direction
    direction_info = detector.calculate_direction_vector(tvec1, tvec2)
    
    # Manual calculations for verification
    pos1 = tvec1[0][0]
    pos2 = tvec2[0][0]
    manual_direction = pos2 - pos1
    manual_distance = np.linalg.norm(manual_direction)
    manual_normalized = manual_direction / manual_distance if manual_distance > 0 else np.zeros(3)
    
    print(f"Direction Vector: {direction_info['direction_vector']}")
    print(f"Manual Direction: {manual_direction}")
    print(f"Normalized Direction: {direction_info['normalized_direction']}")
    print(f"Manual Normalized: {manual_normalized}")
    print(f"Distance: {direction_info['distance']:.6f} meters")
    print(f"Azimuth Angle: {direction_info['azimuth_angle']:.3f}°")
    print(f"Elevation Angle: {direction_info['elevation_angle']:.3f}°")
    print()


def test_relative_orientation():
    """
    Test the relative orientation calculation
    """
    print("=== Testing Relative Orientation Calculation ===")
    
    # Create test data
    _, _, rvec1, rvec2 = create_test_data()
    
    # Initialize detector
    detector = ArucoDetector()
    
    # Calculate relative orientation
    relative_orient = detector.calculate_relative_orientation(rvec1, rvec2)
    
    # Get individual orientations for comparison
    roll1, pitch1, yaw1 = detector.rotation_vector_to_euler(rvec1)
    roll2, pitch2, yaw2 = detector.rotation_vector_to_euler(rvec2)
    
    print(f"Marker 1 Orientation: Roll={roll1:.3f}°, Pitch={pitch1:.3f}°, Yaw={yaw1:.3f}°")
    print(f"Marker 2 Orientation: Roll={roll2:.3f}°, Pitch={pitch2:.3f}°, Yaw={yaw2:.3f}°")
    print(f"Relative Orientation:")
    print(f"  Roll: {relative_orient['relative_roll']:.3f}°")
    print(f"  Pitch: {relative_orient['relative_pitch']:.3f}°")
    print(f"  Yaw: {relative_orient['relative_yaw']:.3f}°")
    print()


def test_comprehensive_analysis():
    """
    Test the comprehensive two-marker analysis
    """
    print("=== Testing Comprehensive Analysis ===")
    
    # Create mock detection data
    tvec1, tvec2, rvec1, rvec2 = create_test_data()
    
    detection_data = {
        'ids': np.array([[0], [1]]),
        'tvecs': np.array([tvec1, tvec2]),
        'rvecs': np.array([rvec1, rvec2]),
        'corners': None,
        'orientations': []
    }
    
    # Initialize detector
    detector = ArucoDetector()
    
    # Perform comprehensive analysis
    analysis = detector.analyze_marker_pair(0, 1, detection_data)
    
    if analysis:
        print("Comprehensive Analysis Results:")
        print(f"Marker IDs: {analysis['marker1_id']} and {analysis['marker2_id']}")
        print(f"Distance: {analysis['distance']:.6f} meters")
        print(f"Direction (Azimuth): {analysis['direction_from_1_to_2']['azimuth_angle']:.3f}°")
        print(f"Direction (Elevation): {analysis['direction_from_1_to_2']['elevation_angle']:.3f}°")
        print(f"Relative Orientation:")
        print(f"  Roll: {analysis['relative_orientation']['relative_roll']:.3f}°")
        print(f"  Pitch: {analysis['relative_orientation']['relative_pitch']:.3f}°")
        print(f"  Yaw: {analysis['relative_orientation']['relative_yaw']:.3f}°")
    else:
        print("Analysis failed")
    print()


def demonstrate_mathematical_concepts():
    """
    Demonstrate the mathematical concepts behind the algorithms
    """
    print("=== Mathematical Concepts Demonstration ===")
    
    print("1. Distance Calculation:")
    print("   Formula: d = √[(x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²]")
    print("   This is the standard Euclidean distance in 3D space.")
    print()
    
    print("2. Direction Vector:")
    print("   Formula: v⃗ = P₂ - P₁")
    print("   Where P₁ and P₂ are the 3D positions of the markers.")
    print("   Normalized: v̂ = v⃗ / |v⃗|")
    print()
    
    print("3. Spherical Coordinates:")
    print("   Azimuth: θ = atan2(dy, dx) - angle in XY plane")
    print("   Elevation: φ = asin(dz / distance) - angle from XY plane")
    print()
    
    print("4. Relative Orientation:")
    print("   R_relative = R₂ × R₁ᵀ")
    print("   Where R₁ and R₂ are rotation matrices of the markers.")
    print("   This gives the rotation needed to align marker 1 with marker 2.")
    print()
    
    print("5. Rotation Vector to Euler Angles:")
    print("   Converts axis-angle representation to Roll-Pitch-Yaw.")
    print("   Uses Rodrigues' rotation formula and rotation matrix decomposition.")
    print()


def create_visualization_example():
    """
    Create a simple visualization of the concepts
    """
    print("=== Creating Visualization Example ===")
    
    # Create a blank image for visualization
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Draw coordinate system
    cv2.line(img, (100, 500), (200, 500), (0, 0, 255), 2)  # X-axis (red)
    cv2.line(img, (100, 500), (100, 400), (0, 255, 0), 2)  # Y-axis (green)
    cv2.putText(img, "X", (205, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Y", (85, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw two example markers
    marker1_pos = (200, 400)
    marker2_pos = (400, 300)
    
    cv2.circle(img, marker1_pos, 20, (255, 255, 0), -1)  # Marker 1 (yellow)
    cv2.circle(img, marker2_pos, 20, (255, 0, 255), -1)  # Marker 2 (magenta)
    
    cv2.putText(img, "Marker 1", (marker1_pos[0] - 30, marker1_pos[1] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "Marker 2", (marker2_pos[0] - 30, marker2_pos[1] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw distance line
    cv2.line(img, marker1_pos, marker2_pos, (0, 255, 255), 2)
    
    # Calculate and display distance (in pixel units for this example)
    distance = np.sqrt((marker2_pos[0] - marker1_pos[0])**2 + 
                      (marker2_pos[1] - marker1_pos[1])**2)
    
    midpoint = ((marker1_pos[0] + marker2_pos[0]) // 2, 
                (marker1_pos[1] + marker2_pos[1]) // 2)
    
    cv2.putText(img, f"Distance: {distance:.1f} pixels", midpoint, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Add title and instructions
    cv2.putText(img, "Two-Marker Distance and Orientation Analysis", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "Press any key to close", 
                (50, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Save and show the visualization
    cv2.imwrite('two_marker_visualization.png', img)
    print("Visualization saved as 'two_marker_visualization.png'")
    
    cv2.imshow('Two-Marker Analysis Concept', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    Main function to run all tests
    """
    print("Two-Marker Distance and Orientation Algorithm Tests")
    print("=" * 60)
    print()
    
    # Run all tests
    test_distance_calculation()
    test_direction_calculation()
    test_relative_orientation()
    test_comprehensive_analysis()
    demonstrate_mathematical_concepts()
    
    # Ask user if they want to see the visualization
    response = input("Would you like to see a visualization example? (y/n): ")
    if response.lower().startswith('y'):
        create_visualization_example()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()
