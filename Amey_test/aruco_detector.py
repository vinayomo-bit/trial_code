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


def main():
    """
    Main function to run ArUco marker detection from webcam
    """
    # Initialize detector
    detector = ArucoDetector()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("ArUco Marker Detection Started")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    print("- Press 'c' to print detection data to console")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        processed_frame, detection_data = detector.process_frame(frame)
        
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
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
