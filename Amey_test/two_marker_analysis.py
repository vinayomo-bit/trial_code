"""
Two-Marker Analysis Script for ArUco Markers

This script demonstrates how to calculate distance and relative orientation
between two detected ArUco markers.

Features:
- Real-time distance calculation between markers
- Relative orientation analysis
- Direction vector computation
- Visual relationship indicators
- Detailed console output
"""

import cv2
import numpy as np
from aruco_detector import ArucoDetector


class TwoMarkerAnalyzer:
    def __init__(self, marker_id_1=0, marker_id_2=1):
        """
        Initialize the two-marker analyzer
        
        Args:
            marker_id_1: ID of the first marker to track
            marker_id_2: ID of the second marker to track
        """
        self.detector = ArucoDetector()
        self.marker_id_1 = marker_id_1
        self.marker_id_2 = marker_id_2
        self.analysis_history = []
        
    def run_analysis(self):
        """
        Run real-time two-marker analysis
        """
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print(f"Two-Marker Analysis Started")
        print(f"Tracking markers: {self.marker_id_1} and {self.marker_id_2}")
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current analysis")
        print("- Press 'h' to show analysis history")
        print("- Press 'r' to reset history")
        print("- Press '1-9' to change first marker ID")
        print("- Press 'SHIFT + 1-9' to change second marker ID")
        
        analysis_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame for marker detection
            processed_frame, detection_data = self.detector.process_frame(frame)
            
            # Perform two-marker analysis
            analysis = self.detector.analyze_marker_pair(
                self.marker_id_1, self.marker_id_2, detection_data
            )
            
            if analysis:
                # Draw relationship visualization
                processed_frame = self.detector.draw_marker_relationship(processed_frame, analysis)
                
                # Add detailed overlay information
                processed_frame = self.add_analysis_overlay(processed_frame, analysis)
                
                # Store in history (keep last 100 analyses)
                self.analysis_history.append(analysis)
                if len(self.analysis_history) > 100:
                    self.analysis_history.pop(0)
            else:
                # Display message when markers are not found
                cv2.putText(processed_frame, 
                           f"Looking for markers {self.marker_id_1} and {self.marker_id_2}...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Two-Marker Analysis', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                if analysis:
                    self.save_analysis(analysis, analysis_count)
                    cv2.imwrite(f'two_marker_analysis_{analysis_count}.jpg', processed_frame)
                    print(f"Analysis saved as two_marker_analysis_{analysis_count}")
                    analysis_count += 1
                else:
                    print("No analysis data to save")
            elif key == ord('h'):
                self.show_analysis_history()
            elif key == ord('r'):
                self.analysis_history.clear()
                print("Analysis history reset")
            elif key >= ord('1') and key <= ord('9'):
                # Check if shift is pressed (capital letters indicate shift)
                if cv2.waitKey(1) & 0xFF != key:  # Simple shift detection
                    self.marker_id_2 = key - ord('0')
                    print(f"Second marker ID changed to: {self.marker_id_2}")
                else:
                    self.marker_id_1 = key - ord('0')
                    print(f"First marker ID changed to: {self.marker_id_1}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
    
    def add_analysis_overlay(self, frame, analysis):
        """
        Add detailed analysis information overlay to the frame
        
        Args:
            frame: Input frame
            analysis: Analysis data
            
        Returns:
            frame: Frame with overlay information
        """
        # Create semi-transparent overlay area
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add text information
        y_pos = 35
        line_height = 20
        
        texts = [
            f"Markers: {analysis['marker1_id']} <-> {analysis['marker2_id']}",
            f"Distance: {analysis['distance']:.4f} m",
            f"Azimuth: {analysis['direction_from_1_to_2']['azimuth_angle']:.1f}°",
            f"Elevation: {analysis['direction_from_1_to_2']['elevation_angle']:.1f}°",
            "",
            f"Relative Orientation:",
            f"  Roll: {analysis['relative_orientation']['relative_roll']:.1f}°",
            f"  Pitch: {analysis['relative_orientation']['relative_pitch']:.1f}°",
            f"  Yaw: {analysis['relative_orientation']['relative_yaw']:.1f}°"
        ]
        
        for i, text in enumerate(texts):
            if text:  # Skip empty lines
                cv2.putText(frame, text, (15, y_pos + i * line_height),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_analysis(self, analysis, count):
        """
        Save analysis data to a text file
        
        Args:
            analysis: Analysis data
            count: Analysis count for filename
        """
        filename = f'two_marker_analysis_{count}.txt'
        
        with open(filename, 'w') as f:
            f.write("=== Two-Marker Analysis Report ===\n\n")
            f.write(f"Marker IDs: {analysis['marker1_id']} and {analysis['marker2_id']}\n")
            f.write(f"Timestamp: Analysis #{count}\n\n")
            
            f.write("Distance and Direction:\n")
            f.write(f"  Distance: {analysis['distance']:.6f} meters\n")
            f.write(f"  Direction Vector: {analysis['direction_from_1_to_2']['direction_vector']}\n")
            f.write(f"  Azimuth Angle: {analysis['direction_from_1_to_2']['azimuth_angle']:.3f}°\n")
            f.write(f"  Elevation Angle: {analysis['direction_from_1_to_2']['elevation_angle']:.3f}°\n\n")
            
            f.write("Marker Positions:\n")
            f.write(f"  Marker {analysis['marker1_id']}: {analysis['marker1_position']}\n")
            f.write(f"  Marker {analysis['marker2_id']}: {analysis['marker2_position']}\n\n")
            
            f.write("Individual Orientations:\n")
            f.write(f"  Marker {analysis['marker1_id']}:\n")
            f.write(f"    Roll: {analysis['marker1_orientation']['roll']:.3f}°\n")
            f.write(f"    Pitch: {analysis['marker1_orientation']['pitch']:.3f}°\n")
            f.write(f"    Yaw: {analysis['marker1_orientation']['yaw']:.3f}°\n")
            f.write(f"  Marker {analysis['marker2_id']}:\n")
            f.write(f"    Roll: {analysis['marker2_orientation']['roll']:.3f}°\n")
            f.write(f"    Pitch: {analysis['marker2_orientation']['pitch']:.3f}°\n")
            f.write(f"    Yaw: {analysis['marker2_orientation']['yaw']:.3f}°\n\n")
            
            f.write("Relative Orientation:\n")
            f.write(f"  Relative Roll: {analysis['relative_orientation']['relative_roll']:.3f}°\n")
            f.write(f"  Relative Pitch: {analysis['relative_orientation']['relative_pitch']:.3f}°\n")
            f.write(f"  Relative Yaw: {analysis['relative_orientation']['relative_yaw']:.3f}°\n")
    
    def show_analysis_history(self):
        """
        Display analysis history statistics
        """
        if not self.analysis_history:
            print("No analysis history available")
            return
        
        distances = [a['distance'] for a in self.analysis_history]
        
        print(f"\n=== Analysis History Statistics ===")
        print(f"Total analyses: {len(self.analysis_history)}")
        print(f"Distance statistics:")
        print(f"  Mean: {np.mean(distances):.4f} m")
        print(f"  Std Dev: {np.std(distances):.4f} m")
        print(f"  Min: {np.min(distances):.4f} m")
        print(f"  Max: {np.max(distances):.4f} m")
        print(f"  Current: {distances[-1]:.4f} m")
        
        # Show last few analyses
        print(f"\nLast 5 distance measurements:")
        for i, distance in enumerate(distances[-5:]):
            print(f"  {len(distances) - 5 + i + 1}: {distance:.4f} m")


def main():
    """
    Main function to run two-marker analysis
    """
    print("Two-Marker ArUco Analysis")
    print("This script analyzes the relationship between two ArUco markers")
    
    # Get marker IDs from user
    try:
        marker1 = int(input("Enter first marker ID (default 0): ") or "0")
        marker2 = int(input("Enter second marker ID (default 1): ") or "1")
    except ValueError:
        print("Invalid input, using default marker IDs 0 and 1")
        marker1, marker2 = 0, 1
    
    # Create and run analyzer
    analyzer = TwoMarkerAnalyzer(marker1, marker2)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
