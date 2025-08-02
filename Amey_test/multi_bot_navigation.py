"""
Multi-Bot ArUco Navigation System

This script manages multiple robots simultaneously, each with their own ArUco marker and target.
Each bot navigates independently to its assigned target using TCP communication with ESP32 controllers.

Features:
- Real-time ArUco detection for multiple markers
- Independent navigation for each bot
- TCP communication with multiple ESP32s
- Configurable movement parameters per bot
- Visual feedback and logging for all bots
- Emergency stop functionality for individual or all bots
"""

import cv2
import socket
import time
import threading
import numpy as np
import math
import json
import os
from aruco_detector import ArucoDetector


class MultiBotController:
    def __init__(self, config_file="bot_config.json"):
        """
        Initialize the multi-bot controller
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.detector = ArucoDetector()
        self.bots = {}  # Dictionary to store bot instances
        self.navigation_active = False
        self.emergency_stop = False
        
        # Load configuration from JSON file
        self.config = self.load_config(config_file)
        
        # Apply settings from config (no defaults)
        nav_settings = self.config["navigation_settings"]  # Must exist
        self.debug_mode = nav_settings["debug_mode"]
        self.use_improved_nav = nav_settings["use_improved_nav"]
        self.show_triangle = nav_settings["show_triangle"]
        
        # Global settings (no defaults)
        self.move_interval = nav_settings["move_interval"]
        
        # Virtual marker system
        self.virtual_markers = {}  # Dictionary to store virtual marker positions
        self.mouse_callback_set = False
        
        # Animation system for signal effects
        self.animation_time = 0
        self.signal_animations = {}  # Store signal animation states for each bot
    
    def load_config(self, config_file):
        """
        Load bot configuration from JSON file
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            dict: Configuration data
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file {config_file} not found! Please create the JSON configuration file.")
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✓ Configuration loaded from {config_file}")
            
            # Validate required sections
            required_sections = ["bots", "camera_settings", "navigation_settings"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section '{section}' in configuration file")
            
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {config_file}: {e}")
        except Exception as e:
            raise Exception(f"Error loading config: {e}")
    
    def load_bots_from_config(self):
        """
        Load all bots from configuration
        """
        bots_config = self.config["bots"]  # Must exist
        nav_settings = self.config["navigation_settings"]  # Must exist
        
        print(f"\n=== Loading {len(bots_config)} bots from configuration ===")
        
        for bot_config in bots_config:
            # Check if bot is enabled (must be specified in config)
            if bot_config["enabled"]:
                # Create bot with config settings (all required)
                bot = SingleBot(
                    bot_config["bot_id"],
                    bot_config["bot_marker_id"],
                    bot_config["target_marker_id"],
                    bot_config["esp32_ip"],
                    bot_config["esp32_port"]
                )
                
                # Apply navigation settings from config (all required)
                bot.distance_threshold = nav_settings["distance_threshold"]
                bot.angle_threshold = nav_settings["angle_threshold"]
                bot.max_distance = nav_settings["max_distance"]
                bot.move_interval = nav_settings["move_interval"]
                
                self.bots[bot_config["bot_id"]] = bot
                
                print(f"✓ {bot_config['bot_id']}: Marker {bot_config['bot_marker_id']} → "
                      f"Target {bot_config['target_marker_id']} @ {bot_config['esp32_ip']}:{bot_config['esp32_port']}")
            else:
                print(f"⚠ {bot_config['bot_id']}: Disabled in configuration")
        
        print(f"Loaded {len(self.bots)} active bots")
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function for placing virtual markers
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Place virtual marker at click position
            self.virtual_markers["virtual"] = (x, y)
            print(f"Virtual marker placed at: ({x}, {y})")
    
    def create_virtual_marker_analysis(self, bot_marker_id, virtual_pos, detection_data):
        """
        Create analysis data for virtual marker navigation
        
        Args:
            bot_marker_id: ID of the bot marker
            virtual_pos: (x, y) position of virtual marker in pixels
            detection_data: Detection data from ArUco detector
            
        Returns:
            dict: Analysis data similar to ArUco marker analysis
        """
        try:
            # Find bot marker in detection data
            ids_list = [id_val[0] for id_val in detection_data['ids']] if detection_data['ids'] is not None else []
            
            if bot_marker_id not in ids_list:
                return None
            
            # Get bot marker data
            bot_marker_index = ids_list.index(bot_marker_id)
            bot_corners = detection_data['corners'][bot_marker_index][0]
            bot_center = np.mean(bot_corners, axis=0)
            
            # Calculate distance (in pixels, convert to approximate meters)
            distance_pixels = np.linalg.norm(np.array(virtual_pos) - bot_center)
            distance_meters = distance_pixels * 0.001  # Rough conversion: 1000 pixels ≈ 1 meter
            
            # Calculate direction vector
            direction_vector = np.array(virtual_pos) - bot_center
            
            # Calculate azimuth angle
            azimuth_angle = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
            
            # Simple bot orientation calculation from marker corners
            # Get the vector from first corner to second corner as orientation reference
            corner_vector = bot_corners[1] - bot_corners[0]
            bot_yaw = math.degrees(math.atan2(corner_vector[1], corner_vector[0]))
            
            # Create simple orientation structure
            bot_orientation = {
                'yaw': bot_yaw,
                'pitch': 0.0,
                'roll': 0.0
            }
            
            # Create analysis structure similar to ArUco marker analysis
            analysis = {
                'distance': distance_meters,
                'marker1_position': bot_center,
                'marker2_position': np.array(virtual_pos),
                'marker1_orientation': bot_orientation,
                'direction_from_1_to_2': {
                    'direction_vector': direction_vector,
                    'azimuth_angle': azimuth_angle
                }
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error creating virtual marker analysis: {e}")
            return None
    
    def save_config(self, config_file="bot_config.json"):
        """
        Save current configuration to JSON file
        
        Args:
            config_file: Path to save configuration
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✓ Configuration saved to {config_file}")
        except Exception as e:
            print(f"✗ Error saving config: {e}")
        
    def add_bot(self, bot_id, bot_marker_id, target_marker_id, esp32_ip, esp32_port=8888):
        """
        Add a new bot to the system
        
        Args:
            bot_id: Unique identifier for this bot (e.g., "Bot1", "Bot2", "Bot3")
            bot_marker_id: ArUco marker ID attached to this bot
            target_marker_id: ArUco marker ID of the target for this bot
            esp32_ip: IP address of the ESP32 controlling this bot
            esp32_port: Port number for ESP32 communication
        """
        bot = SingleBot(bot_id, bot_marker_id, target_marker_id, esp32_ip, esp32_port)
        self.bots[bot_id] = bot
        print(f"Added {bot_id}: Marker {bot_marker_id} → Target {target_marker_id} @ {esp32_ip}:{esp32_port}")
    
    def connect_all_bots(self):
        """
        Establish connections to all ESP32 controllers
        """
        print("\n=== Connecting to all ESP32 controllers ===")
        success_count = 0
        
        for bot_id, bot in self.bots.items():
            print(f"\nConnecting {bot_id}...")
            if bot.connect_to_esp32():
                success_count += 1
                print(f"✓ {bot_id} connected successfully")
                # Explicitly disable continuous mode to ensure timed movements
                bot.send_command("continuous_off")
                time.sleep(0.2)  # Small delay between connections
            else:
                print(f"✗ {bot_id} connection failed")
        
        print(f"\nConnection summary: {success_count}/{len(self.bots)} bots connected")
        return success_count > 0
    
    def process_navigation_all_bots(self, detection_data):
        """
        Process navigation for all bots simultaneously
        
        Args:
            detection_data: Detection data from ArUco detector
        """
        if not self.navigation_active or self.emergency_stop:
            return
        
        current_time = time.time()
        
        for bot_id, bot in self.bots.items():
            if bot.connected and (current_time - bot.last_command_time) >= bot.move_interval:
                # Check if this bot uses virtual marker
                if bot.target_marker_id == "virtual":
                    # Handle virtual marker navigation
                    if "virtual" in self.virtual_markers:
                        virtual_pos = self.virtual_markers["virtual"]
                        analysis = self.create_virtual_marker_analysis(
                            bot.bot_marker_id, virtual_pos, detection_data
                        )
                    else:
                        analysis = None  # No virtual marker placed yet
                else:
                    # Handle normal ArUco marker navigation
                    analysis = self.detector.analyze_marker_pair(
                        bot.bot_marker_id, bot.target_marker_id, detection_data
                    )
                
                if analysis:
                    # Debug if enabled
                    if self.debug_mode:
                        print(f"\n--- {bot_id} Debug ---")
                        self.debug_coordinate_system(analysis, bot_id)
                    
                    # Calculate navigation command
                    if self.use_improved_nav:
                        command = bot.calculate_navigation_command_v2(analysis)
                    else:
                        command = bot.calculate_navigation_command(analysis)
                    
                    # Send command to ESP32
                    if bot.send_command(command):
                        # Print navigation info
                        if self.use_improved_nav:
                            bot_yaw = analysis['marker1_orientation']['yaw']
                            direction_vector = analysis['direction_from_1_to_2']['direction_vector']
                            target_angle = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
                            relative_angle = target_angle - bot_yaw
                            while relative_angle > 180:
                                relative_angle -= 360
                            while relative_angle < -180:
                                relative_angle += 360
                            
                            print(f"{bot_id}: {command} | Dist: {analysis['distance']:.2f}m | "
                                  f"Yaw: {bot_yaw:.1f}° | Rel: {relative_angle:.1f}°")
                        else:
                            raw_azimuth = analysis['direction_from_1_to_2']['azimuth_angle']
                            nav_angle = raw_azimuth - 90.0
                            if nav_angle > 180:
                                nav_angle -= 360
                            elif nav_angle < -180:
                                nav_angle += 360
                            
                            print(f"{bot_id}: {command} | Dist: {analysis['distance']:.2f}m | "
                                  f"Nav: {nav_angle:.1f}°")
                        
                        bot.last_command_time = current_time
                        bot.bot_position = analysis['marker1_position']
                        bot.target_position = analysis['marker2_position']
                    else:
                        print(f"{bot_id}: Connection lost")
                        bot.connected = False
                else:
                    # Can't see both markers for this bot
                    if (current_time - bot.last_command_time) > 2.0:  # Send stop every 2 seconds
                        bot.send_command("stop")
                        bot.last_command_time = current_time
                        if self.debug_mode:
                            if bot.target_marker_id == "virtual":
                                print(f"{bot_id}: Virtual marker not placed or bot not visible - stopping")
                            else:
                                print(f"{bot_id}: Markers not visible - stopping")
    
    def add_navigation_overlay(self, frame, detection_data):
        """
        Add navigation information overlay for all bots
        
        Args:
            frame: Input frame
            detection_data: Detection data
            
        Returns:
            frame: Frame with navigation overlay
        """
        # Create semi-transparent overlay
        overlay = frame.copy()
        overlay_height = 50 + (len(self.bots) * 60)  # Dynamic height based on number of bots
        cv2.rectangle(overlay, (10, 10), (500, overlay_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add status information
        y_pos = 30
        line_height = 15
        
        # Global status
        nav_color = (0, 255, 0) if self.navigation_active else (0, 255, 255)
        cv2.putText(frame, f"Multi-Bot Navigation: {'Active' if self.navigation_active else 'Inactive'}", 
                   (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, nav_color, 2)
        y_pos += line_height + 10
        
        # Individual bot status
        ids_list = [id_val[0] for id_val in detection_data['ids']] if detection_data['ids'] is not None else []
        
        for bot_id, bot in self.bots.items():
            # Bot header
            cv2.putText(frame, f"{bot_id}:", 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_pos += line_height
            
            # Connection status
            conn_color = (0, 255, 0) if bot.connected else (0, 0, 255)
            cv2.putText(frame, f"  Connection: {'Connected' if bot.connected else 'Disconnected'}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, conn_color, 1)
            y_pos += line_height
            
            # Marker detection status
            bot_detected = bot.bot_marker_id in ids_list
            
            if bot.target_marker_id == "virtual":
                # Virtual marker status
                virtual_available = "virtual" in self.virtual_markers
                marker_status = f"  Bot M{bot.bot_marker_id}: {'✓' if bot_detected else '✗'} | Virtual Target: {'✓' if virtual_available else '✗ (Click to place)'}"
                marker_color = (0, 255, 0) if (bot_detected and virtual_available) else (255, 255, 0) if (bot_detected or virtual_available) else (0, 0, 255)
            else:
                # Normal ArUco marker status
                target_detected = bot.target_marker_id in ids_list
                marker_status = f"  Bot M{bot.bot_marker_id}: {'✓' if bot_detected else '✗'} | Target M{bot.target_marker_id}: {'✓' if target_detected else '✗'}"
                marker_color = (0, 255, 0) if (bot_detected and target_detected) else (255, 255, 0) if (bot_detected or target_detected) else (0, 0, 255)
            
            cv2.putText(frame, marker_status, 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, marker_color, 1)
            y_pos += line_height
            
            # Distance if both markers detected
            if bot.target_marker_id == "virtual":
                # Virtual marker distance
                if bot_detected and "virtual" in self.virtual_markers:
                    try:
                        virtual_pos = self.virtual_markers["virtual"]
                        analysis = self.create_virtual_marker_analysis(
                            bot.bot_marker_id, virtual_pos, detection_data
                        )
                        if analysis:
                            distance_color = (0, 255, 0) if analysis['distance'] < 0.2 else (255, 255, 0)
                            cv2.putText(frame, f"  Distance: {analysis['distance']:.3f}m", 
                                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, distance_color, 1)
                            
                            # Draw line to virtual marker
                            bot_center = analysis['marker1_position'].astype(int)
                            virtual_center = analysis['marker2_position'].astype(int)
                            cv2.line(frame, tuple(bot_center), tuple(virtual_center), (255, 0, 255), 2)
                            cv2.arrowedLine(frame, tuple(bot_center), tuple(virtual_center), (255, 0, 255), 2)
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Error drawing virtual marker for {bot_id}: {e}")
            else:
                # Normal ArUco marker distance
                bot_detected = bot.bot_marker_id in ids_list
                target_detected = bot.target_marker_id in ids_list
                if bot_detected and target_detected:
                    try:
                        analysis = self.detector.analyze_marker_pair(
                            bot.bot_marker_id, bot.target_marker_id, detection_data
                        )
                        if analysis:
                            distance_color = (0, 255, 0) if analysis['distance'] < 0.2 else (255, 255, 0)
                            cv2.putText(frame, f"  Distance: {analysis['distance']:.3f}m", 
                                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, distance_color, 1)
                            
                            # Draw navigation line
                            frame = self.detector.draw_marker_relationship(frame, analysis)
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Error drawing marker relationship for {bot_id}: {e}")
            
            y_pos += 5  # Extra spacing between bots
        
        # Draw triangle if enabled and we have bot positions
        if self.show_triangle:
            frame = self.draw_bot_triangle(frame, detection_data)
        
        # Draw virtual markers
        frame = self.draw_virtual_markers(frame)
        
        # Draw signal animations for all bots
        frame = self.draw_signal_animations(frame, detection_data)
        
        return frame
    
    def debug_coordinate_system(self, analysis, bot_id):
        """
        Debug function for coordinate system (simplified for multi-bot)
        """
        direction_vector = analysis['direction_from_1_to_2']['direction_vector']
        bot_yaw = analysis['marker1_orientation']['yaw']
        
        target_angle = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
        relative_angle = target_angle - bot_yaw
        while relative_angle > 180:
            relative_angle -= 360
        while relative_angle < -180:
            relative_angle += 360
        
        print(f"{bot_id} Debug: Yaw={bot_yaw:.1f}°, RelAngle={relative_angle:.1f}°, Dist={analysis['distance']:.3f}m")
    
    def draw_bot_triangle(self, frame, detection_data):
        """
        Draw a triangle connecting the three bots if all are visible
        """
        if len(self.bots) < 3:
            return frame
        
        # Get all detected marker IDs
        ids_list = [id_val[0] for id_val in detection_data['ids']] if detection_data['ids'] is not None else []
        
        # Collect bot positions that are currently visible
        bot_positions = {}
        bot_centers = {}
        
        for bot_id, bot in self.bots.items():
            if bot.bot_marker_id in ids_list:
                # Find the bot marker in detection data
                for i, marker_id in enumerate(ids_list):
                    if marker_id == bot.bot_marker_id:
                        # Get marker corners and calculate center
                        corners = detection_data['corners'][i][0]
                        center_x = int(np.mean(corners[:, 0]))
                        center_y = int(np.mean(corners[:, 1]))
                        bot_positions[bot_id] = (center_x, center_y)
                        bot_centers[bot_id] = (center_x, center_y)
                        break
        
        # If we have at least 3 bots visible, draw triangle
        if len(bot_positions) >= 3:
            # Get first 3 bot positions
            positions = list(bot_positions.values())[:3]
            bot_names = list(bot_positions.keys())[:3]
            
            # Draw triangle lines
            triangle_color = (0, 255, 255)  # Yellow triangle
            thickness = 3
            
            # Draw the three sides of triangle
            cv2.line(frame, positions[0], positions[1], triangle_color, thickness)
            cv2.line(frame, positions[1], positions[2], triangle_color, thickness)
            cv2.line(frame, positions[2], positions[0], triangle_color, thickness)
            
            # Draw bot labels at each corner
            for i, (bot_name, pos) in enumerate(zip(bot_names, positions)):
                cv2.circle(frame, pos, 8, triangle_color, -1)
                cv2.putText(frame, bot_name, 
                           (pos[0] + 15, pos[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, triangle_color, 2)
            
            # Calculate triangle area and perimeter
            area = self.calculate_triangle_area(positions)
            perimeter = self.calculate_triangle_perimeter(positions)
            
            # Display triangle info
            info_y = frame.shape[0] - 80
            cv2.putText(frame, f"Triangle Area: {area:.1f} pixels²", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, triangle_color, 2)
            cv2.putText(frame, f"Triangle Perimeter: {perimeter:.1f} pixels", 
                       (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, triangle_color, 2)
            
            # Check if triangle is approximately equilateral
            side_lengths = [
                np.linalg.norm(np.array(positions[0]) - np.array(positions[1])),
                np.linalg.norm(np.array(positions[1]) - np.array(positions[2])),
                np.linalg.norm(np.array(positions[2]) - np.array(positions[0]))
            ]
            
            max_side = max(side_lengths)
            min_side = min(side_lengths)
            if max_side > 0 and (max_side - min_side) / max_side < 0.2:  # Within 20% difference
                cv2.putText(frame, "Triangle: ~Equilateral", 
                           (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Triangle: Irregular", 
                           (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        return frame
    
    def calculate_triangle_area(self, positions):
        """
        Calculate triangle area using the cross product method
        """
        if len(positions) < 3:
            return 0
        
        p1, p2, p3 = positions[0], positions[1], positions[2]
        # Area = 0.5 * |det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])|
        area = abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)
        return area
    
    def calculate_triangle_perimeter(self, positions):
        """
        Calculate triangle perimeter
        """
        if len(positions) < 3:
            return 0
        
        side1 = np.linalg.norm(np.array(positions[0]) - np.array(positions[1]))
        side2 = np.linalg.norm(np.array(positions[1]) - np.array(positions[2]))
        side3 = np.linalg.norm(np.array(positions[2]) - np.array(positions[0]))
        
        return side1 + side2 + side3
    
    def draw_virtual_markers(self, frame):
        """
        Draw virtual markers on the frame
        """
        try:
            for marker_id, position in self.virtual_markers.items():
                x, y = int(position[0]), int(position[1])
                
                # Draw virtual marker as a cross with circle
                marker_color = (255, 0, 255)  # Magenta color for virtual markers
                size = 15
                thickness = 3
                
                # Draw cross
                cv2.line(frame, (x - size, y), (x + size, y), marker_color, thickness)
                cv2.line(frame, (x, y - size), (x, y + size), marker_color, thickness)
                
                # Draw circle around it
                cv2.circle(frame, (x, y), size + 5, marker_color, 2)
                
                # Add label
                cv2.putText(frame, f"Virtual-{marker_id}", 
                           (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, marker_color, 2)
        except Exception as e:
            if self.debug_mode:
                print(f"Error drawing virtual markers: {e}")
        
        return frame
    
    def draw_signal_animations(self, frame, detection_data):
        """
        Draw animated signal antenna effects for all bots
        """
        try:
            current_time = time.time()
            self.animation_time = current_time
            
            # Get all detected marker IDs
            ids_list = [id_val[0] for id_val in detection_data['ids']] if detection_data['ids'] is not None else []
            
            # Draw signal animations for each bot
            for bot_id, bot in self.bots.items():
                if bot.bot_marker_id in ids_list:
                    # Find the bot marker in detection data
                    for i, marker_id in enumerate(ids_list):
                        if marker_id == bot.bot_marker_id:
                            # Get marker corners and calculate center
                            corners = detection_data['corners'][i][0]
                            center_x = int(np.mean(corners[:, 0]))
                            center_y = int(np.mean(corners[:, 1]))
                            
                            # Draw signal animation
                            frame = self.draw_bot_signal_animation(frame, bot_id, bot, (center_x, center_y), current_time)
                            break
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error drawing signal animations: {e}")
        
        return frame
    
    def draw_bot_signal_animation(self, frame, bot_id, bot, center_pos, current_time):
        """
        Draw signal antenna animation for a single bot
        
        Args:
            frame: The frame to draw on
            bot_id: Bot identifier
            bot: Bot object
            center_pos: (x, y) center position of the bot
            current_time: Current timestamp
        """
        try:
            x, y = center_pos
            
            # Initialize signal animation state for this bot if not exists
            if bot_id not in self.signal_animations:
                self.signal_animations[bot_id] = {
                    'last_signal_time': current_time,
                    'signal_phase': 0
                }
            
            signal_state = self.signal_animations[bot_id]
            
            # Signal timing - emit signal every 2 seconds
            signal_interval = 2.0
            time_since_last = current_time - signal_state['last_signal_time']
            
            # Check if it's time for a new signal burst
            if time_since_last >= signal_interval:
                signal_state['last_signal_time'] = current_time
                signal_state['signal_phase'] = 0
            
            # Calculate animation progress (0 to 1 over 1.5 seconds)
            animation_duration = 1.5
            progress = min(time_since_last / animation_duration, 1.0)
            
            # Only draw if we're in animation phase
            if progress < 1.0:
                # Connection status color
                if bot.connected:
                    base_color = (0, 255, 0)  # Green for connected
                else:
                    base_color = (0, 0, 255)  # Red for disconnected
                
                # Draw multiple signal rings
                max_rings = 3
                for ring in range(max_rings):
                    ring_delay = ring * 0.3  # Stagger rings by 0.3 seconds
                    ring_progress = max(0, min((progress * animation_duration - ring_delay) / 1.2, 1.0))
                    
                    if ring_progress > 0:
                        # Calculate ring properties
                        max_radius = 60
                        current_radius = int(ring_progress * max_radius)
                        
                        # Fade out as ring expands
                        alpha = int(255 * (1.0 - ring_progress))
                        signal_color = (
                            min(255, base_color[0] + alpha // 4),
                            min(255, base_color[1]),
                            min(255, base_color[2] + alpha // 4)
                        )
                        
                        # Draw ring
                        if current_radius > 0:
                            thickness = max(1, int(3 * (1.0 - ring_progress)))
                            cv2.circle(frame, (x, y), current_radius, signal_color, thickness)
                
                # Draw antenna lines (radiating outward)
                num_lines = 8
                for i in range(num_lines):
                    angle = (i * 360 / num_lines) + (progress * 360 * 0.5)  # Rotate slowly
                    angle_rad = math.radians(angle)
                    
                    # Line length based on progress
                    line_length = int(20 + progress * 40)
                    end_x = x + int(line_length * math.cos(angle_rad))
                    end_y = y + int(line_length * math.sin(angle_rad))
                    
                    # Line color with fade
                    line_alpha = int(255 * (1.0 - progress) * 0.7)
                    line_color = (
                        min(255, base_color[0] + line_alpha // 3),
                        min(255, base_color[1]),
                        min(255, base_color[2] + line_alpha // 3)
                    )
                    
                    # Draw antenna line
                    cv2.line(frame, (x, y), (end_x, end_y), line_color, 2)
                
                # Draw central antenna symbol
                antenna_color = base_color
                cv2.circle(frame, (x, y), 5, antenna_color, -1)  # Central dot
                cv2.circle(frame, (x, y), 8, antenna_color, 2)   # Outer ring
                
                # Draw signal strength indicator based on connection
                if bot.connected:
                    # Connected - draw full signal bars
                    for bar in range(3):
                        bar_height = 8 + bar * 4
                        bar_x = x + 15 + bar * 4
                        cv2.rectangle(frame, (bar_x, y - bar_height), (bar_x + 2, y), (0, 255, 0), -1)
                else:
                    # Disconnected - draw weak signal or X
                    cv2.line(frame, (x + 15, y - 8), (x + 25, y + 2), (0, 0, 255), 2)
                    cv2.line(frame, (x + 25, y - 8), (x + 15, y + 2), (0, 0, 255), 2)
            
            # Draw bot status indicator
            status_color = (0, 255, 0) if bot.connected else (0, 0, 255)
            cv2.putText(frame, f"{bot_id}", (x + 30, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error drawing signal animation for {bot_id}: {e}")
        
        return frame
    
    def emergency_stop_all(self):
        """
        Emergency stop for all bots
        """
        self.emergency_stop = True
        self.navigation_active = False
        
        for bot_id, bot in self.bots.items():
            if bot.connected:
                bot.send_command("stop")
        
        print("EMERGENCY STOP - All bots stopped!")
    
    def stop_individual_bot(self, bot_id):
        """
        Stop a specific bot
        """
        if bot_id in self.bots and self.bots[bot_id].connected:
            self.bots[bot_id].send_command("stop")
            print(f"{bot_id} stopped")
    
    def run_navigation(self):
        """
        Main navigation loop for all bots
        """
        # Get camera settings from config (all required)
        camera_settings = self.config["camera_settings"]
        camera_index = camera_settings["camera_index"]
        resolution = camera_settings["resolution"]
        
        # Initialize camera with settings from config
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera resolution from config
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution["height"])
        
        # Get actual resolution (camera may not support exact values)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution set to: {actual_width}x{actual_height}")
        
        print(f"\n=== Multi-Bot ArUco Navigation Started ===")
        print(f"Managing {len(self.bots)} bots")
        for bot_id, bot in self.bots.items():
            print(f"  {bot_id}: Marker {bot.bot_marker_id} → Target {bot.target_marker_id}")
        
        print("\nControls:")
        print("- Press 'SPACE' to start/stop navigation for all bots")
        print("- Press 'e' for emergency stop (all bots)")
        print("- Press '1', '2', '3', '4' to stop individual bots")
        print("- Press 'c' to disable continuous mode (enable timed movements)")
        print("- Press 'd' to toggle debug mode")
        print("- Press 'n' to toggle navigation method")
        print("- Press 's' to save current frame")
        print("- Press 'r' to reconnect all bots")
        print("- Press 't' to test all bots")
        print("- Press 'v' to toggle triangle visualization")
        print("- Press 'x' to save current configuration")
        print("- Press 'm' to clear virtual markers")
        print("- Click on screen to place virtual marker for Bot with ID 6")
        print("- Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame for marker detection
            processed_frame, detection_data = self.detector.process_frame(frame)
            
            # Add navigation overlay
            processed_frame = self.add_navigation_overlay(processed_frame, detection_data)
            
            # Process navigation for all bots
            self.process_navigation_all_bots(detection_data)
            
            # Display frame with window size from config
            resolution = self.config["camera_settings"]["resolution"]
            cv2.namedWindow('Multi-Bot ArUco Navigation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Multi-Bot ArUco Navigation', resolution["width"], resolution["height"])
            
            # Set mouse callback for virtual marker placement
            if not self.mouse_callback_set:
                cv2.setMouseCallback('Multi-Bot ArUco Navigation', self.mouse_callback)
                self.mouse_callback_set = True
            
            cv2.imshow('Multi-Bot ArUco Navigation', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar
                self.navigation_active = not self.navigation_active
                self.emergency_stop = False
                if self.navigation_active:
                    print("Navigation STARTED for all bots")
                else:
                    print("Navigation STOPPED for all bots")
                    for bot in self.bots.values():
                        if bot.connected:
                            bot.send_command("stop")
            elif key == ord('e'):
                self.emergency_stop_all()
            elif key == ord('1'):
                self.stop_individual_bot("Bot1")
            elif key == ord('2'):
                self.stop_individual_bot("Bot2")
            elif key == ord('3'):
                self.stop_individual_bot("Bot3")
            elif key == ord('4'):
                self.stop_individual_bot("Bot4")
            elif key == ord('c'):
                # Toggle continuous mode for all bots
                print("Toggling continuous mode for all bots...")
                for bot_id, bot in self.bots.items():
                    if bot.connected:
                        bot.send_command("continuous_off")  # Start with non-continuous
                        time.sleep(0.1)
                print("Continuous mode disabled for all connected bots (timed movements)")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('n'):
                self.use_improved_nav = not self.use_improved_nav
                nav_method = "Improved" if self.use_improved_nav else "Original"
                print(f"Navigation method: {nav_method}")
            elif key == ord('s'):
                cv2.imwrite(f'multi_bot_frame_{frame_count}.jpg', processed_frame)
                print(f"Frame saved as multi_bot_frame_{frame_count}.jpg")
                frame_count += 1
            elif key == ord('r'):
                print("Reconnecting all bots...")
                self.connect_all_bots()
            elif key == ord('t'):
                print("Testing all bots...")
                for bot_id, bot in self.bots.items():
                    if bot.connected:
                        bot.send_command("test")
                        time.sleep(0.5)
            elif key == ord('v'):
                self.show_triangle = not self.show_triangle
                triangle_status = "ON" if self.show_triangle else "OFF"
                print(f"Triangle visualization: {triangle_status}")
            elif key == ord('x'):
                # Save current configuration
                self.config["navigation_settings"]["show_triangle"] = self.show_triangle
                self.config["navigation_settings"]["debug_mode"] = self.debug_mode
                self.config["navigation_settings"]["use_improved_nav"] = self.use_improved_nav
                self.save_config()
            elif key == ord('m'):
                # Clear virtual markers
                self.virtual_markers.clear()
                print("Virtual markers cleared")
        
        # Cleanup
        if self.navigation_active:
            for bot in self.bots.values():
                if bot.connected:
                    bot.send_command("stop")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Close all connections
        for bot in self.bots.values():
            if bot.socket:
                bot.socket.close()
    
    def print_movement_logs(self):
        """
        Print movement logs for all bots
        """
        print("\n=== Movement Logs for All Bots ===")
        for bot_id, bot in self.bots.items():
            print(f"\n{bot_id} Log:")
            for entry in bot.movement_log[-5:]:  # Last 5 commands per bot
                timestamp = time.strftime('%H:%M:%S', time.localtime(entry['time']))
                print(f"  {timestamp}: {entry['command']} -> {entry['response']}")


class SingleBot:
    """
    Individual bot controller - simplified version for multi-bot operation
    """
    def __init__(self, bot_id, bot_marker_id, target_marker_id, esp32_ip, esp32_port):
        self.bot_id = bot_id
        self.bot_marker_id = bot_marker_id
        self.target_marker_id = target_marker_id
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        
        # Connection
        self.socket = None
        self.connected = False
        
        # Navigation parameters (will be set from config)
        self.distance_threshold = None
        self.angle_threshold = None
        self.max_distance = None
        self.move_interval = None
        
        # Status tracking
        self.last_command_time = 0
        self.bot_position = None
        self.target_position = None
        self.movement_log = []
    
    def connect_to_esp32(self):
        """Connect to this bot's ESP32"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            
            self.socket.connect((self.esp32_ip, self.esp32_port))
            response = self.socket.recv(1024).decode().strip()
            print(f"  {self.bot_id} response: {response}")
            
            self.connected = True
            self.send_command("status")
            return True
            
        except Exception as e:
            print(f"  {self.bot_id} connection error: {e}")
            self.connected = False
            return False
    
    def send_command(self, command):
        """Send command to this bot's ESP32"""
        if not self.connected or not self.socket:
            return False
        
        try:
            # Send command
            self.socket.send((command + '\n').encode())
            
            # Wait for acknowledgment (same as original)
            response = self.socket.recv(1024).decode().strip()
            
            # Log the command
            self.movement_log.append({
                'time': time.time(),
                'command': command,
                'response': response
            })
            
            if len(self.movement_log) > 50:
                self.movement_log.pop(0)
            
            return True
            
        except Exception as e:
            if command != "stop":  # Don't spam stop command errors
                print(f"{self.bot_id} command error: {e}")
            self.connected = False
            return False
    
    def calculate_navigation_command(self, analysis):
        """Original navigation method"""
        if not analysis:
            return "stop"
        
        distance = analysis['distance']
        raw_azimuth = analysis['direction_from_1_to_2']['azimuth_angle']
        
        if distance < self.distance_threshold:
            return "stop"
        if distance > self.max_distance:
            return "stop"
        
        nav_angle = raw_azimuth - 90.0
        if nav_angle > 180:
            nav_angle -= 360
        elif nav_angle < -180:
            nav_angle += 360
        
        if abs(nav_angle) < self.angle_threshold:
            return "forward"
        elif nav_angle > self.angle_threshold:
            return "right"
        elif nav_angle < -self.angle_threshold:
            return "left"
        else:
            return "stop"
    
    def calculate_navigation_command_v2(self, analysis):
        """Improved navigation method considering bot orientation"""
        if not analysis:
            return "stop"
        
        distance = analysis['distance']
        
        if distance < self.distance_threshold:
            return "stop"
        if distance > self.max_distance:
            return "stop"
        
        bot_yaw = analysis['marker1_orientation']['yaw']
        direction_vector = analysis['direction_from_1_to_2']['direction_vector']
        target_angle = math.degrees(math.atan2(direction_vector[1], direction_vector[0]))
        relative_angle = target_angle - bot_yaw
        
        while relative_angle > 180:
            relative_angle -= 360
        while relative_angle < -180:
            relative_angle += 360
        
        if abs(relative_angle) < self.angle_threshold:
            return "forward"
        elif relative_angle > self.angle_threshold:
            return "right"
        elif relative_angle < -self.angle_threshold:
            return "left"
        else:
            return "stop"


def main():
    """
    Main function to set up and run multi-bot navigation
    """
    print("Multi-Bot ArUco Navigation System")
    print("=" * 40)
    
    # Use bot_config.json by default
    config_file = "bot_config.json"
    print(f"Loading configuration from: {config_file}")
    
    try:
        # Create multi-bot controller with config
        controller = MultiBotController(config_file)
        
        # Load bots from configuration
        controller.load_bots_from_config()
        
        if not controller.bots:
            print("No enabled bots found in configuration!")
            return
        
        # Connect to all bots
        if not controller.connect_all_bots():
            response = input("\nSome connections failed. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Run navigation
        controller.run_navigation()
        
    except FileNotFoundError as e:
        print(f"Configuration Error: {e}")
        print("Please create a bot_config.json file with proper configuration.")
        return
    
    except (ValueError, KeyError) as e:
        print(f"Configuration Error: {e}")
        print("Please check your JSON configuration file for missing or invalid values.")
        return
    
    except KeyboardInterrupt:
        print("\nNavigation interrupted by user")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        try:
            # Show movement logs
            controller.print_movement_logs()
            print("Multi-Bot Navigation ended")
        except:
            pass


if __name__ == "__main__":
    main()
