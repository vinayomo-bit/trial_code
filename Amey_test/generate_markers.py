import cv2
import numpy as np
import os


def generate_aruco_markers(dictionary_type=cv2.aruco.DICT_6X6_250, 
                          marker_ids=[0, 1, 2, 3, 4], 
                          marker_size=200,
                          output_dir="aruco_markers"):
    """
    Generate ArUco markers and save them as images
    
    Args:
        dictionary_type: ArUco dictionary type
        marker_ids: List of marker IDs to generate
        marker_size: Size of the marker in pixels
        output_dir: Directory to save the markers
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the ArUco dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
    
    print(f"Generating ArUco markers...")
    
    for marker_id in marker_ids:
        # Generate the marker
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
        
        # Add white border for better detection
        border_size = 50
        bordered_image = np.ones((marker_size + 2 * border_size, 
                                marker_size + 2 * border_size), dtype=np.uint8) * 255
        bordered_image[border_size:border_size + marker_size, 
                      border_size:border_size + marker_size] = marker_image
        
        # Save the marker
        filename = os.path.join(output_dir, f"marker_{marker_id}.png")
        cv2.imwrite(filename, bordered_image)
        
        print(f"Generated marker ID {marker_id}: {filename}")
    
    print(f"\nAll markers saved in '{output_dir}' directory")
    print("You can print these markers and use them for testing")


def generate_marker_board(dictionary_type=cv2.aruco.DICT_6X6_250,
                         markers_x=3, markers_y=3,
                         marker_size=100, marker_separation=20,
                         output_file="aruco_board.png"):
    """
    Generate a board with multiple ArUco markers
    
    Args:
        dictionary_type: ArUco dictionary type
        markers_x: Number of markers in X direction
        markers_y: Number of markers in Y direction
        marker_size: Size of each marker in pixels
        marker_separation: Separation between markers in pixels
        output_file: Output filename
    """
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
    
    # Create the board
    board = cv2.aruco.GridBoard((markers_x, markers_y), marker_size, 
                               marker_separation, dictionary)
    
    # Generate the board image
    board_size = (markers_x * (marker_size + marker_separation) - marker_separation + 100,
                 markers_y * (marker_size + marker_separation) - marker_separation + 100)
    
    board_image = board.generateImage(board_size)
    
    # Save the board
    cv2.imwrite(output_file, board_image)
    print(f"Generated ArUco board: {output_file}")


if __name__ == "__main__":
    print("ArUco Marker Generator")
    print("=" * 30)
    
    # Generate individual markers
    generate_aruco_markers(
        dictionary_type=cv2.aruco.DICT_6X6_250,
        marker_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        marker_size=200
    )
    
    print()
    
    # Generate a marker board
    generate_marker_board(
        dictionary_type=cv2.aruco.DICT_6X6_250,
        markers_x=3,
        markers_y=3,
        marker_size=100,
        marker_separation=20
    )
    
    print("\nMarker generation complete!")
    print("Print the markers on paper and use them with the detector.")
...existing code...
