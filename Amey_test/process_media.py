import cv2
import numpy as np
import argparse
import os
from aruco_detector import ArucoDetector


def process_image(image_path, output_path=None):
    """
    Process a single image for ArUco marker detection
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image (optional)
    """
    # Initialize detector
    detector = ArucoDetector()
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Process the image
    processed_image, detection_data = detector.process_frame(image)
    
    # Print detection results
    if detection_data['ids'] is not None:
        print(f"Found {len(detection_data['ids'])} ArUco marker(s):")
        for orientation in detection_data['orientations']:
            print(f"\nMarker ID {orientation['id']}:")
            print(f"  Roll:  {orientation['roll']:.2f}°")
            print(f"  Pitch: {orientation['pitch']:.2f}°")
            print(f"  Yaw:   {orientation['yaw']:.2f}°")
            print(f"  Position: ({orientation['position'][0]:.3f}, "
                  f"{orientation['position'][1]:.3f}, "
                  f"{orientation['position'][2]:.3f})")
    else:
        print("No ArUco markers detected in the image")
    
    # Save output image if path is provided
    if output_path:
        cv2.imwrite(output_path, processed_image)
        print(f"Output saved to: {output_path}")
    
    # Display image
    cv2.imshow('ArUco Detection Result', processed_image)
    print("\nPress any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path, output_path=None):
    """
    Process a video file for ArUco marker detection
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (optional)
    """
    # Initialize detector
    detector = ArucoDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detections_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        processed_frame, detection_data = detector.process_frame(frame)
        
        # Count detections
        if detection_data['ids'] is not None:
            detections_count += 1
        
        # Write frame to output video
        if out is not None:
            out.write(processed_frame)
        
        # Display frame
        cv2.imshow('ArUco Video Processing', processed_frame)
        
        # Print progress
        if frame_count % 30 == 0:  # Print every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% - Detections in {detections_count}/{frame_count} frames")
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing interrupted by user")
            break
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
        print(f"Output video saved to: {output_path}")
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections: {detections_count}")
    print(f"Detection rate: {(detections_count/frame_count)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='ArUco Marker Detection for Images and Videos')
    parser.add_argument('input', help='Input file path (image or video)')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-t', '--type', choices=['auto', 'image', 'video'], 
                       default='auto', help='Input type (auto-detect by default)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    # Determine file type
    file_type = args.type
    if file_type == 'auto':
        ext = os.path.splitext(args.input)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            file_type = 'image'
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            file_type = 'video'
        else:
            print(f"Error: Unknown file type for '{args.input}'")
            print("Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
            print("Supported video formats: .mp4, .avi, .mov, .mkv, .wmv, .flv")
            return
    
    # Process file
    if file_type == 'image':
        process_image(args.input, args.output)
    elif file_type == 'video':
        process_video(args.input, args.output)


if __name__ == "__main__":
    main()
