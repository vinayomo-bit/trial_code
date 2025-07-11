/*
 * ArUco Bot Controller - ESP32 with L293D Motor Driver
 * 
 * This code controls a robot with two DC motors using L293D driver.
 * The robot receives movement commands via TCP socket from Python ArUco detection system.
 * 
 * Hardware Connections:
 * ESP32 -> L293D
 * GPIO 16 -> IN1 (Left Motor Forward)
 * GPIO 17 -> IN2 (Left Motor Backward)
 * GPIO 18 -> IN3 (Right Motor Forward)
 * GPIO 19 -> IN4 (Right Motor Backward)
 * GPIO 21 -> ENA (Left Motor Speed - PWM)
 * GPIO 22 -> ENB (Right Motor Speed - PWM)
 * 
 * Power:
 * L293D VCC1 -> 3.3V (Logic)
 * L293D VCC2 -> 6-12V (Motor Power)
 * Common Ground between ESP32, L293D, and motor power supply
 */

#include <WiFi.h>
#include <WiFiServer.h>

// Motor Control Pins
#define LEFT_MOTOR_IN1    16
#define LEFT_MOTOR_IN2    17
#define RIGHT_MOTOR_IN1   18
#define RIGHT_MOTOR_IN2   19
#define LEFT_MOTOR_PWM    21  // ENA
#define RIGHT_MOTOR_PWM   22  // ENB

// WiFi Configuration - CHANGE THESE VALUES
const char* ssid = "YOUR_WIFI_SSID";           // Replace with your WiFi name
const char* password = "YOUR_WIFI_PASSWORD";   // Replace with your WiFi password
const int serverPort = 8888;                   // TCP server port

// Motor Control Variables
int motorSpeed = 200;        // PWM value (0-255)
int turnSpeed = 150;         // PWM value for turning
int moveTime = 500;          // Movement duration in milliseconds
int turnTime = 300;          // Turn duration in milliseconds

// Network
WiFiServer server(serverPort);
WiFiClient client;

// Status LED (built-in)
#define LED_PIN 2

void setup() {
  Serial.begin(115200);
  Serial.println("ArUco Bot Controller Starting...");
  
  // Initialize motor pins
  initializeMotors();
  
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
  
  // Test motors
  testMotors();
}

void loop() {
  // Check for new client connections
  if (!client.connected()) {
    client = server.available();
    if (client) {
      Serial.println("Client connected!");
      digitalWrite(LED_PIN, HIGH);  // Turn on LED when connected
      client.println("ArUco Bot Ready");
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

void initializeMotors() {
  pinMode(LEFT_MOTOR_IN1, OUTPUT);
  pinMode(LEFT_MOTOR_IN2, OUTPUT);
  pinMode(RIGHT_MOTOR_IN1, OUTPUT);
  pinMode(RIGHT_MOTOR_IN2, OUTPUT);
  pinMode(LEFT_MOTOR_PWM, OUTPUT);
  pinMode(RIGHT_MOTOR_PWM, OUTPUT);
  
  // Stop all motors initially
  stopMotors();
  
  Serial.println("Motors initialized");
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
    stopMotors();
  }
  else if (command.startsWith("speed:")) {
    int newSpeed = command.substring(6).toInt();
    if (newSpeed >= 0 && newSpeed <= 255) {
      motorSpeed = newSpeed;
      Serial.println("Speed set to: " + String(motorSpeed));
    }
  }
  else if (command.startsWith("turn_speed:")) {
    int newTurnSpeed = command.substring(11).toInt();
    if (newTurnSpeed >= 0 && newTurnSpeed <= 255) {
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
  else if (command == "status") {
    sendStatus();
  }
  else if (command == "test") {
    testMotors();
  }
  else {
    Serial.println("Unknown command: " + command);
  }
}

void moveForward() {
  Serial.println("Moving forward");
  
  // Left motor forward
  digitalWrite(LEFT_MOTOR_IN1, HIGH);
  digitalWrite(LEFT_MOTOR_IN2, LOW);
  analogWrite(LEFT_MOTOR_PWM, motorSpeed);
  
  // Right motor forward
  digitalWrite(RIGHT_MOTOR_IN1, HIGH);
  digitalWrite(RIGHT_MOTOR_IN2, LOW);
  analogWrite(RIGHT_MOTOR_PWM, motorSpeed);
  
  delay(moveTime);
  stopMotors();
}

void moveBackward() {
  Serial.println("Moving backward");
  
  // Left motor backward
  digitalWrite(LEFT_MOTOR_IN1, LOW);
  digitalWrite(LEFT_MOTOR_IN2, HIGH);
  analogWrite(LEFT_MOTOR_PWM, motorSpeed);
  
  // Right motor backward
  digitalWrite(RIGHT_MOTOR_IN1, LOW);
  digitalWrite(RIGHT_MOTOR_IN2, HIGH);
  analogWrite(RIGHT_MOTOR_PWM, motorSpeed);
  
  delay(moveTime);
  stopMotors();
}

void turnLeft() {
  Serial.println("Turning left");
  
  // Left motor backward (or stop)
  digitalWrite(LEFT_MOTOR_IN1, LOW);
  digitalWrite(LEFT_MOTOR_IN2, HIGH);
  analogWrite(LEFT_MOTOR_PWM, turnSpeed);
  
  // Right motor forward
  digitalWrite(RIGHT_MOTOR_IN1, HIGH);
  digitalWrite(RIGHT_MOTOR_IN2, LOW);
  analogWrite(RIGHT_MOTOR_PWM, turnSpeed);
  
  delay(turnTime);
  stopMotors();
}

void turnRight() {
  Serial.println("Turning right");
  
  // Left motor forward
  digitalWrite(LEFT_MOTOR_IN1, HIGH);
  digitalWrite(LEFT_MOTOR_IN2, LOW);
  analogWrite(LEFT_MOTOR_PWM, turnSpeed);
  
  // Right motor backward (or stop)
  digitalWrite(RIGHT_MOTOR_IN1, LOW);
  digitalWrite(RIGHT_MOTOR_IN2, HIGH);
  analogWrite(RIGHT_MOTOR_PWM, turnSpeed);
  
  delay(turnTime);
  stopMotors();
}

void stopMotors() {
  digitalWrite(LEFT_MOTOR_IN1, LOW);
  digitalWrite(LEFT_MOTOR_IN2, LOW);
  digitalWrite(RIGHT_MOTOR_IN1, LOW);
  digitalWrite(RIGHT_MOTOR_IN2, LOW);
  analogWrite(LEFT_MOTOR_PWM, 0);
  analogWrite(RIGHT_MOTOR_PWM, 0);
}

void testMotors() {
  Serial.println("Testing motors...");
  
  Serial.println("Forward test");
  moveForward();
  delay(1000);
  
  Serial.println("Left turn test");
  turnLeft();
  delay(1000);
  
  Serial.println("Right turn test");
  turnRight();
  delay(1000);
  
  Serial.println("Backward test");
  moveBackward();
  delay(1000);
  
  stopMotors();
  Serial.println("Motor test complete");
}

void sendStatus() {
  if (client && client.connected()) {
    client.println("STATUS:");
    client.println("Motor Speed: " + String(motorSpeed));
    client.println("Turn Speed: " + String(turnSpeed));
    client.println("Move Time: " + String(moveTime) + "ms");
    client.println("Turn Time: " + String(turnTime) + "ms");
    client.println("WiFi Signal: " + String(WiFi.RSSI()) + " dBm");
    client.println("Free Heap: " + String(ESP.getFreeHeap()) + " bytes");
  }
}

// Interrupt handler for emergency stop (optional)
void emergencyStop() {
  stopMotors();
  Serial.println("EMERGENCY STOP!");
}
