/*
 * ArUco Bot Controller - ESP32 with 2 Continuous Servo Motors
 * 
 * This code controls a robot with two continuous servo motors.
 * The robot receives movement commands via TCP socket from Python ArUco detection system.
 * 
 * Hardware Connections:
 * ESP32 -> Servo Motors
 * GPIO 4  -> Left Servo Signal (servo2)
 * GPIO 32 -> Right Servo Signal (servo4)
 * 
 * Power:
 * Servo VCC -> 5V or 6V (check servo specifications)
 * Servo GND -> ESP32 GND
 * Common Ground between ESP32 and servo power supply
 * 
 * Servo Control:
 * - 90 degrees = Stop
 * - 0 degrees = Full speed one direction
 * - 180 degrees = Full speed opposite direction
 * 
 * Movement Logic:
 * Forward:  Left=0°,  Right=180°
 * Backward: Left=180°, Right=0°
 * Left:     Left=180°, Right=180°
 * Right:    Left=0°,   Right=0°
 */

#include <WiFi.h>
#include <WiFiServer.h>
#include <ESP32Servo.h>

// Servo Control Pins
#define LEFT_SERVO_PIN    4   // servo2
#define RIGHT_SERVO_PIN   32  // servo4

// WiFi Configuration - CHANGE THESE VALUES
const char* ssid = "YOUR_WIFI_SSID";           // Replace with your WiFi name
const char* password = "YOUR_WIFI_PASSWORD";   // Replace with your WiFi password
const int serverPort = 8888;                   // TCP server port

// Servo Objects
Servo leftServo;   // servo2
Servo rightServo;  // servo4

// Define servo speed constants (0-180 degree values)
const int STOP = 90;
const int FORWARD_LEFT = 0;
const int FORWARD_RIGHT = 180;
const int BACKWARD_LEFT = 180;
const int BACKWARD_RIGHT = 0;

// Speed settings
int moveSpeed = 30;    // Speed offset from STOP (90±30)
int turnSpeed = 45;    // Turn speed offset from STOP (90±45)

// Movement timing
int moveTime = 500;          // Movement duration in milliseconds
int turnTime = 300;          // Turn duration in milliseconds
bool continuousMode = false; // Set to true for continuous movement until stop command

// Network
WiFiServer server(serverPort);
WiFiClient client;

// Status LED (built-in)
#define LED_PIN 2

void setup() {
  Serial.begin(115200);
  Serial.println("ArUco Bot Controller (Servo Version) Starting...");
  
  // Initialize servos
  initializeServos();
  
  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Connect to WiFi
  connectToWiFi();
  
  // Start TCP server
  server.begin();
  Serial.println("TCP Server started");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.print("Port: ");
  Serial.println(serverPort);
  Serial.println("Waiting for Python client connection...");
  
  // Test servos
  testServos();
}

void loop() {
  // Check for new client connections
  if (!client.connected()) {
    client = server.available();
    if (client) {
      Serial.println("Client connected!");
      digitalWrite(LED_PIN, HIGH);  // Turn on LED when connected
      client.println("ArUco Bot Ready (Servo Version)");
    }
  }
  
  // Handle client commands
  if (client && client.connected() && client.available()) {
    String command = client.readStringUntil('\n');
    command.trim();
    
    if (command.length() > 0) {
      Serial.println("Received: " + command);
      executeCommand(command);
      
      // Send acknowledgment
      client.println("OK");
    }
  }
  
  // Blink LED when waiting for connection
  if (!client.connected()) {
    digitalWrite(LED_PIN, millis() % 1000 < 500);
  }
  
  delay(10);
}

void initializeServos() {
  // Attach servos to pins
  leftServo.attach(LEFT_SERVO_PIN);
  rightServo.attach(RIGHT_SERVO_PIN);
  
  // Stop both servos initially
  stopServos();
  
  Serial.println("Servos initialized and stopped");
}

void connectToWiFi() {
  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 50) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("WiFi connected successfully!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal strength: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println();
    Serial.println("Failed to connect to WiFi!");
    Serial.println("Please check your credentials and try again.");
    // Continue anyway for testing
  }
}

void executeCommand(String command) {
  command.toLowerCase();
  
  if (command == "forward" || command == "f") {
    moveForward();
  }
  else if (command == "backward" || command == "b") {
    moveBackward();
  }
  else if (command == "left" || command == "l") {
    turnLeft();
  }
  else if (command == "right" || command == "r") {
    turnRight();
  }
  else if (command == "stop" || command == "s") {
    stopServos();
  }
  else if (command.startsWith("move_speed:")) {
    int newSpeed = command.substring(11).toInt();
    if (newSpeed >= 10 && newSpeed <= 90) {
      moveSpeed = newSpeed;
      Serial.println("Move speed set to: " + String(moveSpeed));
    }
  }
  else if (command.startsWith("turn_speed:")) {
    int newTurnSpeed = command.substring(11).toInt();
    if (newTurnSpeed >= 10 && newTurnSpeed <= 90) {
      turnSpeed = newTurnSpeed;
      Serial.println("Turn speed set to: " + String(turnSpeed));
    }
  }
  else if (command.startsWith("move_time:")) {
    int newMoveTime = command.substring(10).toInt();
    if (newMoveTime >= 0 && newMoveTime <= 5000) {
      moveTime = newMoveTime;
      Serial.println("Move time set to: " + String(moveTime) + "ms");
    }
  }
  else if (command.startsWith("turn_time:")) {
    int newTurnTime = command.substring(10).toInt();
    if (newTurnTime >= 0 && newTurnTime <= 5000) {
      turnTime = newTurnTime;
      Serial.println("Turn time set to: " + String(turnTime) + "ms");
    }
  }
  else if (command == "continuous_on") {
    continuousMode = true;
    Serial.println("Continuous mode enabled");
  }
  else if (command == "continuous_off") {
    continuousMode = false;
    Serial.println("Continuous mode disabled");
  }
  else if (command == "status") {
    sendStatus();
  }
  else if (command == "test") {
    testServos();
  }
  else if (command == "calibrate") {
    calibrateServos();
  }
  else if (command == "test_individual") {
    testIndividualServos();
  }
  else {
    Serial.println("Unknown command: " + command);
  }
}

void moveForward() {
  Serial.println("Moving forward");
  
  // Both servos forward
  leftServo.write(FORWARD_LEFT);      // servo2 forward (0)
  rightServo.write(FORWARD_RIGHT);    // servo4 forward (180)
  
  if (!continuousMode) {
    delay(moveTime);
    stopServos();
  }
}

void moveBackward() {
  Serial.println("Moving backward");
  
  // Both servos backward
  leftServo.write(BACKWARD_LEFT);     // servo2 backward (180)
  rightServo.write(BACKWARD_RIGHT);   // servo4 backward (0)
  
  if (!continuousMode) {
    delay(moveTime);
    stopServos();
  }
}

void turnLeft() {
  Serial.println("Turning left");
  
  // Left servo backward, right servo forward
  leftServo.write(BACKWARD_LEFT);     // servo2 backward (180)
  rightServo.write(FORWARD_RIGHT);    // servo4 forward (180)
  
  if (!continuousMode) {
    delay(turnTime);
    stopServos();
  }
}

void turnRight() {
  Serial.println("Turning right");
  
  // Left servo forward, right servo backward
  leftServo.write(FORWARD_LEFT);      // servo2 forward (0)
  rightServo.write(BACKWARD_RIGHT);   // servo4 backward (0)
  
  if (!continuousMode) {
    delay(turnTime);
    stopServos();
  }
}

void stopServos() {
  Serial.println("Stopping servos");
  leftServo.write(STOP);   // servo2 stop (90)
  rightServo.write(STOP);  // servo4 stop (90)
}

void testServos() {
  Serial.println("Testing servos...");
  
  Serial.println("Forward test");
  moveForward();
  if (continuousMode) delay(1000);
  delay(1000);
  
  Serial.println("Left turn test");
  turnLeft();
  if (continuousMode) delay(1000);
  delay(1000);
  
  Serial.println("Right turn test");
  turnRight();
  if (continuousMode) delay(1000);
  delay(1000);
  
  Serial.println("Backward test");
  moveBackward();
  if (continuousMode) delay(1000);
  delay(1000);
  
  stopServos();
  Serial.println("Servo test complete");
}

void calibrateServos() {
  Serial.println("Calibrating servos - testing movement patterns");
  
  Serial.println("Testing STOP position (90 degrees)");
  leftServo.write(STOP);
  rightServo.write(STOP);
  delay(2000);
  
  Serial.println("Testing LEFT servo forward (0 degrees)");
  leftServo.write(FORWARD_LEFT);
  rightServo.write(STOP);
  delay(2000);
  
  Serial.println("Testing LEFT servo backward (180 degrees)");
  leftServo.write(BACKWARD_LEFT);
  rightServo.write(STOP);
  delay(2000);
  
  Serial.println("Testing RIGHT servo forward (180 degrees)");
  leftServo.write(STOP);
  rightServo.write(FORWARD_RIGHT);
  delay(2000);
  
  Serial.println("Testing RIGHT servo backward (0 degrees)");
  leftServo.write(STOP);
  rightServo.write(BACKWARD_RIGHT);
  delay(2000);
  
  // Return to stop
  stopServos();
  Serial.println("Calibration complete.");
}

void testIndividualServos() {
  Serial.println("Testing individual servos...");
  
  Serial.println("Testing LEFT servo only:");
  Serial.println("LEFT forward");
  leftServo.write(FORWARD_LEFT);
  rightServo.write(STOP);
  delay(2000);
  
  Serial.println("LEFT backward");
  leftServo.write(BACKWARD_LEFT);
  rightServo.write(STOP);
  delay(2000);
  
  Serial.println("Testing RIGHT servo only:");
  Serial.println("RIGHT forward");
  leftServo.write(STOP);
  rightServo.write(FORWARD_RIGHT);
  delay(2000);
  
  Serial.println("RIGHT backward");
  leftServo.write(STOP);
  rightServo.write(BACKWARD_RIGHT);
  delay(2000);
  
  stopServos();
  Serial.println("Individual servo test complete");
}

void sendStatus() {
  if (client && client.connected()) {
    client.println("STATUS:");
    client.println("Servo Type: Continuous Rotation (2 motors)");
    client.println("Left Servo Pin: " + String(LEFT_SERVO_PIN));
    client.println("Right Servo Pin: " + String(RIGHT_SERVO_PIN));
    client.println("Move Speed Offset: " + String(moveSpeed));
    client.println("Turn Speed Offset: " + String(turnSpeed));
    client.println("Stop Position: " + String(STOP) + " degrees");
    client.println("Move Time: " + String(moveTime) + "ms");
    client.println("Turn Time: " + String(turnTime) + "ms");
    client.println("Continuous Mode: " + String(continuousMode ? "ON" : "OFF"));
    client.println("WiFi Signal: " + String(WiFi.RSSI()) + " dBm");
    client.println("Free Heap: " + String(ESP.getFreeHeap()) + " bytes");
  }
}

// Interrupt handler for emergency stop (optional)
void emergencyStop() {
  stopServos();
  Serial.println("EMERGENCY STOP!");
}