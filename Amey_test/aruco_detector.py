import cv2
import numpy as np
import math


class ArucoDetector:
    def __init__(self, dictionary_type=cv2.aruco.DICT_6X6_250, marker_size=0.05):
        """
        Initialize ArUco detector
        
        Args:
            dictionary_type: ArUco dictionary type (default: DICT_6X6_250)
            marker_size: Real-world size of the marker in meters
        """
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.marker_size = marker_size
        
        # Camera calibration parameters (you should calibrate your camera for accurate results)
        # These are example values - replace with your camera's calibration
        self.camera_matrix = np.array([[800, 0, 320],
                                     [0, 800, 240],
                                     [0, 0, 1]], dtype=np.float32)
        
        self.dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)
    
    def detect_markers(self, frame):
        """
        Detect ArUco markers in the frame
        
        Args:
            frame: Input image/frame
            
        Returns:
            corners: Detected marker corners
            ids: Detected marker IDs
            rejected: Rejected marker candidates
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected
    
    def estimate_pose(self, corners):
        """
        Estimate pose (position and orientation) of detected markers
        
        Args:
            corners: Marker corners from detection
            
        Returns:
            rvecs: Rotation vectors
            tvecs: Translation vectors
        """
        if len(corners) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            return rvecs, tvecs
        return None, None
    
    def rotation_vector_to_euler(self, rvec):
        """
        Convert rotation vector to Euler angles (roll, pitch, yaw)
        
        Args:
            rvec: Rotation vector
            
        Returns:
            roll, pitch, yaw in degrees
        """
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Extract Euler angles from rotation matrix
        sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + 
                      rotation_matrix[1, 0] * rotation_matrix[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = math.atan2(-rotation_matrix[2, 0], sy)
            z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = math.atan2(-rotation_matrix[2, 0], sy)
            z = 0
        
        # Convert from radians to degrees
        roll = math.degrees(x)
        pitch = math.degrees(y)
        yaw = math.degrees(z)
        
        return roll, pitch, yaw
    
    def draw_markers_and_axes(self, frame, corners, ids, rvecs, tvecs):
        """
        Draw detected markers and their coordinate axes
        
        Args:
            frame: Input frame
            corners: Marker corners
            ids: Marker IDs
            rvecs: Rotation vectors
            tvecs: Translation vectors
            
        Returns:
            frame: Frame with drawn markers and axes
        """
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            if rvecs is not None and tvecs is not None:
                for i in range(len(ids)):
                    # Draw coordinate axes for each marker
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                    rvecs[i], tvecs[i], self.marker_size * 0.5)
                    
                    # Get Euler angles
                    roll, pitch, yaw = self.rotation_vector_to_euler(rvecs[i])
                    
                    # Display marker ID and orientation
                    marker_id = ids[i][0]
                    text_position = tuple(map(int, corners[i][0][0]))
                    
                    # Display marker ID
                    cv2.putText(frame, f"ID: {marker_id}", 
                              (text_position[0], text_position[1] - 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Display orientation
                    cv2.putText(frame, f"Roll: {roll:.1f}°", 
                              (text_position[0], text_position[1] - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    cv2.putText(frame, f"Pitch: {pitch:.1f}°", 
                              (text_position[0], text_position[1] - 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    cv2.putText(frame, f"Yaw: {yaw:.1f}°", 
                              (text_position[0], text_position[1]),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    # Display position
                    x, y, z = tvecs[i][0]
                    cv2.putText(frame, f"Pos: ({x:.2f}, {y:.2f}, {z:.2f})", 
                              (text_position[0], text_position[1] + 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame to detect markers and estimate their pose
        
        Args:
            frame: Input frame
            
        Returns:
            processed_frame: Frame with detected markers and orientation info
            detection_data: Dictionary containing detection results
        """
        # Detect markers
        corners, ids, rejected = self.detect_markers(frame)
        
        # Estimate pose
        rvecs, tvecs = self.estimate_pose(corners)
        
        # Draw markers and axes
        processed_frame = self.draw_markers_and_axes(frame, corners, ids, rvecs, tvecs)
        
        # Prepare detection data
        detection_data = {
            'ids': ids,
            'corners': corners,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'orientations': []
        }
        
        # Calculate orientations for each detected marker
        if ids is not None and rvecs is not None:
            for i, marker_id in enumerate(ids):
                roll, pitch, yaw = self.rotation_vector_to_euler(rvecs[i])
                detection_data['orientations'].append({
                    'id': marker_id[0],
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'position': tvecs[i][0]
                })
        
        return processed_frame, detection_data
    
    def calculate_distance_between_markers(self, tvec1, tvec2):
        """
        Calculate Euclidean distance between two markers
        
        Args:
            tvec1: Translation vector of first marker
            tvec2: Translation vector of second marker
            
        Returns:
            distance: Distance between markers in meters
        """
        # Extract 3D positions - handle different array shapes
        if tvec1.ndim == 3:
            pos1 = tvec1[0][0]  # Shape: (1, 1, 3)
        else:
            pos1 = tvec1[0]     # Shape: (1, 3)
            
        if tvec2.ndim == 3:
            pos2 = tvec2[0][0]  # Shape: (1, 1, 3)
        else:
            pos2 = tvec2[0]     # Shape: (1, 3)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(pos2 - pos1)
        return distance
    
    def calculate_relative_orientation(self, rvec1, rvec2):
        """
        Calculate relative orientation between two markers
        
        Args:
            rvec1: Rotation vector of first marker
            rvec2: Rotation vector of second marker
            
        Returns:
            relative_angles: Dictionary containing relative roll, pitch, yaw in degrees
        """
        # Convert rotation vectors to rotation matrices
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        
        # Calculate relative rotation matrix
        # R_relative = R2 * R1^T (rotation from marker1 to marker2)
        R_relative = R2 @ R1.T
        
        # Convert relative rotation matrix to Euler angles
        relative_rvec, _ = cv2.Rodrigues(R_relative)
        relative_roll, relative_pitch, relative_yaw = self.rotation_vector_to_euler(relative_rvec)
        
        return {
            'relative_roll': relative_roll,
            'relative_pitch': relative_pitch,
            'relative_yaw': relative_yaw
        }
    
    def calculate_direction_vector(self, tvec1, tvec2):
        """
        Calculate direction vector from marker1 to marker2
        
        Args:
            tvec1: Translation vector of first marker
            tvec2: Translation vector of second marker
            
        Returns:
            direction_info: Dictionary containing direction vector and angles
        """
        # Extract 3D positions - handle different array shapes
        if tvec1.ndim == 3:
            pos1 = tvec1[0][0]  # Shape: (1, 1, 3)
        else:
            pos1 = tvec1[0]     # Shape: (1, 3)
            
        if tvec2.ndim == 3:
            pos2 = tvec2[0][0]  # Shape: (1, 1, 3)
        else:
            pos2 = tvec2[0]     # Shape: (1, 3)
        
        # Calculate direction vector (from marker1 to marker2)
        direction_vector = pos2 - pos1
        
        # Normalize the direction vector
        distance = np.linalg.norm(direction_vector)
        if distance > 0:
            normalized_direction = direction_vector / distance
        else:
            normalized_direction = np.array([0, 0, 0])
        
        # Calculate spherical coordinates (azimuth and elevation angles)
        # Azimuth: angle in XY plane from X-axis
        azimuth = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
        
        # Elevation: angle from XY plane to the vector
        elevation = math.degrees(math.asin(direction_vector[2] / distance)) if distance > 0 else 0
        
        return {
            'direction_vector': direction_vector,
            'normalized_direction': normalized_direction,
            'azimuth_angle': azimuth,
            'elevation_angle': elevation,
            'distance': distance
        }
    
    def analyze_marker_pair(self, id1, id2, detection_data):
        """
        Comprehensive analysis of relationship between two specific markers
        
        Args:
            id1: ID of first marker
            id2: ID of second marker
            detection_data: Detection data from process_frame
            
        Returns:
            analysis: Dictionary containing all relationship metrics
        """
        if detection_data['ids'] is None:
            return None
        
        # Find indices of the specified markers
        ids_list = [id_val[0] for id_val in detection_data['ids']]
        
        try:
            idx1 = ids_list.index(id1)
            idx2 = ids_list.index(id2)
        except ValueError:
            return None  # One or both markers not found
        
        # Extract pose data
        tvec1 = detection_data['tvecs'][idx1]
        tvec2 = detection_data['tvecs'][idx2]
        rvec1 = detection_data['rvecs'][idx1]
        rvec2 = detection_data['rvecs'][idx2]
        
        # Calculate all relationship metrics
        distance = self.calculate_distance_between_markers(tvec1, tvec2)
        relative_orientation = self.calculate_relative_orientation(rvec1, rvec2)
        direction_info = self.calculate_direction_vector(tvec1, tvec2)
        
        # Get individual marker orientations
        roll1, pitch1, yaw1 = self.rotation_vector_to_euler(rvec1)
        roll2, pitch2, yaw2 = self.rotation_vector_to_euler(rvec2)
        
        analysis = {
            'marker1_id': id1,
            'marker2_id': id2,
            'distance': distance,
            'marker1_position': tvec1[0],
            'marker2_position': tvec2[0],
            'marker1_orientation': {'roll': roll1, 'pitch': pitch1, 'yaw': yaw1},
            'marker2_orientation': {'roll': roll2, 'pitch': pitch2, 'yaw': yaw2},
            'relative_orientation': relative_orientation,
            'direction_from_1_to_2': direction_info
        }
        
        return analysis
    
    def draw_marker_relationship(self, frame, analysis):
        """
        Draw visual indicators of the relationship between two markers
        
        Args:
            frame: Input frame
            analysis: Analysis data from analyze_marker_pair
            
        Returns:
            frame: Frame with relationship visualization
        """
        if analysis is None:
            return frame
        
        # Project 3D positions to 2D image coordinates
        pos1_3d = np.array([[analysis['marker1_position']]], dtype=np.float32)
        pos2_3d = np.array([[analysis['marker2_position']]], dtype=np.float32)
        
        pos1_2d, _ = cv2.projectPoints(pos1_3d, np.zeros(3), np.zeros(3), 
                                      self.camera_matrix, self.dist_coeffs)
        pos2_2d, _ = cv2.projectPoints(pos2_3d, np.zeros(3), np.zeros(3), 
                                      self.camera_matrix, self.dist_coeffs)
        
        pt1 = tuple(map(int, pos1_2d[0][0]))
        pt2 = tuple(map(int, pos2_2d[0][0]))
        
        # Draw line connecting the markers
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Draw distance text at midpoint
        midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        distance_text = f"Dist: {analysis['distance']:.3f}m"
        cv2.putText(frame, distance_text, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 255), 2)
        
        # Draw direction arrow
        arrow_length = 30
        direction = np.array(pt2) - np.array(pt1)
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction) * arrow_length
            arrow_end = tuple(map(int, np.array(pt1) + direction))
            cv2.arrowedLine(frame, pt1, arrow_end, (255, 0, 255), 3, tipLength=0.3)
        
        return frame

def main():
    """
    Main function to run ArUco marker detection from webcam
    """
    # Initialize detector
    detector = ArucoDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("ArUco Marker Detection with Two-Marker Analysis Started")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    print("- Press 'c' to print detection data to console")
    print("- Press 'r' to analyze relationship between markers 0 and 1")
    print("- Press 't' to toggle continuous two-marker analysis")
    
    frame_count = 0
    continuous_analysis = False
    target_marker_1 = 0  # Default first marker ID
    target_marker_2 = 1  # Default second marker ID
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        processed_frame, detection_data = detector.process_frame(frame)
        
        # Perform two-marker analysis if enabled
        marker_analysis = None
        if continuous_analysis or cv2.waitKey(1) & 0xFF == ord('r'):
            marker_analysis = detector.analyze_marker_pair(target_marker_1, target_marker_2, detection_data)
            if marker_analysis:
                processed_frame = detector.draw_marker_relationship(processed_frame, marker_analysis)
                
                # Display relationship info on frame
                y_offset = 30
                info_texts = [
                    f"Markers {target_marker_1} <-> {target_marker_2}",
                    f"Distance: {marker_analysis['distance']:.3f}m",
                    f"Azimuth: {marker_analysis['direction_from_1_to_2']['azimuth_angle']:.1f}°",
                    f"Elevation: {marker_analysis['direction_from_1_to_2']['elevation_angle']:.1f}°",
                    f"Rel. Yaw: {marker_analysis['relative_orientation']['relative_yaw']:.1f}°"
                ]
                
                for i, text in enumerate(info_texts):
                    cv2.putText(processed_frame, text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display frame
        cv2.imshow('ArUco Marker Detection', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'aruco_detection_{frame_count}.jpg', processed_frame)
            print(f"Frame saved as aruco_detection_{frame_count}.jpg")
            frame_count += 1
        elif key == ord('c'):
            if detection_data['ids'] is not None:
                print("\nDetection Data:")
                for orientation in detection_data['orientations']:
                    print(f"Marker ID {orientation['id']}:")
                    print(f"  Roll: {orientation['roll']:.2f}°")
                    print(f"  Pitch: {orientation['pitch']:.2f}°")
                    print(f"  Yaw: {orientation['yaw']:.2f}°")
                    print(f"  Position: {orientation['position']}")
                    print()
            else:
                print("No markers detected")
        elif key == ord('r'):
            # Print detailed relationship analysis
            if marker_analysis:
                print(f"\n=== Marker Relationship Analysis ===")
                print(f"Marker {target_marker_1} to Marker {target_marker_2}:")
                print(f"Distance: {marker_analysis['distance']:.4f} meters")
                print(f"Direction Vector: {marker_analysis['direction_from_1_to_2']['direction_vector']}")
                print(f"Azimuth Angle: {marker_analysis['direction_from_1_to_2']['azimuth_angle']:.2f}°")
                print(f"Elevation Angle: {marker_analysis['direction_from_1_to_2']['elevation_angle']:.2f}°")
                print(f"Relative Orientation:")
                print(f"  Roll: {marker_analysis['relative_orientation']['relative_roll']:.2f}°")
                print(f"  Pitch: {marker_analysis['relative_orientation']['relative_pitch']:.2f}°")
                print(f"  Yaw: {marker_analysis['relative_orientation']['relative_yaw']:.2f}°")
                print(f"Individual Orientations:")
                print(f"  Marker {target_marker_1}: Roll={marker_analysis['marker1_orientation']['roll']:.2f}°, "
                      f"Pitch={marker_analysis['marker1_orientation']['pitch']:.2f}°, "
                      f"Yaw={marker_analysis['marker1_orientation']['yaw']:.2f}°")
                print(f"  Marker {target_marker_2}: Roll={marker_analysis['marker2_orientation']['roll']:.2f}°, "
                      f"Pitch={marker_analysis['marker2_orientation']['pitch']:.2f}°, "
                      f"Yaw={marker_analysis['marker2_orientation']['yaw']:.2f}°")
            else:
                print(f"Could not find both markers {target_marker_1} and {target_marker_2}")
        elif key == ord('t'):
            continuous_analysis = not continuous_analysis
            print(f"Continuous two-marker analysis: {'ON' if continuous_analysis else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
