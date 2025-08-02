#include <WiFi.h>
#include <WiFiServer.h>
#include <ESP32Servo.h>

// ————— Forward declarations —————
void initializeMotors();
void initializeUltrasonic();
void initializeServo();
void connectToWiFi();
void executeCommand(String command);
void executeOpenCommand();
void startRadar();
void stopRadar();
void setRadarAngle(int angle);
void updateRadarSweep();
void sendRadarData();
void sendRadarReading(int angle, float distance);
float readUltrasonic();
void testMotors();
void testRadarSystem();
void moveForward();
void moveBackward();
void turnLeft();
void turnRight();
void stopMotors();
void sendStatus();
void emergencyStop();

// ————— Pin definitions —————
// Motor Control Pins
#define LEFT_MOTOR_IN1    27
#define LEFT_MOTOR_IN2    14
#define RIGHT_MOTOR_IN1   33
#define RIGHT_MOTOR_IN2   25
// Ultrasonic Sensor Pins
#define TRIG_PIN          5
#define ECHO_PIN          18
// Servo Pin for Radar Sweeping
#define SERVO_PIN         4
// Status LED
#define LED_PIN           2

// ————— WiFi & Server —————
const char* ssid       = "Bot_wro";      // your SSID
const char* password   = "80141075";     // your password
const int   serverPort = 8888;
WiFiServer server(serverPort);
WiFiClient client;

// ————— Motion & Radar Settings —————
int motorSpeed            = 200;
int turnSpeed             = 150;
int moveTime              = 500;
int turnTime              = 300;
Servo radarServo;
bool radarEnabled         = false;
int currentAngle          = 0;
int sweepDirection        = 1;
int sweepStep             = 5;
int sweepDelay            = 100;
int radarSendInterval     = 50;
unsigned long lastSweepTime      = 0;
unsigned long lastRadarSend      = 0;
unsigned long lastUltrasonicRead = 0;
int ultrasonicReadInterval       = 30;
float lastDistance                = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("ArUco Bot Controller with Ultrasonic Radar Starting...");

  initializeMotors();
  initializeUltrasonic();
  initializeServo();

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  connectToWiFi();
  server.begin();
  Serial.println("TCP Server started");
  Serial.print("IP Address: "); Serial.println(WiFi.localIP());
  Serial.print("Port: ");       Serial.println(serverPort);

  testMotors();
  testRadarSystem();
}

void loop() {
  // Accept new client
  if (!client.connected()) {
    client = server.available();
    if (client) {
      Serial.println("Client connected!");
      digitalWrite(LED_PIN, HIGH);
      client.println("ArUco Bot with Radar Ready");
    }
  }

  // Handle incoming commands
  if (client && client.connected() && client.available()) {
    String command = client.readStringUntil('\n');
    command.trim();
    if (command.length()) {
      Serial.println("Received: " + command);
      executeCommand(command);
      client.println("OK");
    }
  }

  // Radar sweep & data send
  if (radarEnabled) {
    updateRadarSweep();
    sendRadarData();
  }

  // Blink LED when disconnected
  if (!client.connected()) {
    digitalWrite(LED_PIN, millis() % 1000 < 500);
  }

  delay(10);
}

// ————— Initialization Routines —————
void initializeMotors() {
  pinMode(LEFT_MOTOR_IN1, OUTPUT);
  pinMode(LEFT_MOTOR_IN2, OUTPUT);
  pinMode(RIGHT_MOTOR_IN1, OUTPUT);
  pinMode(RIGHT_MOTOR_IN2, OUTPUT);
  stopMotors();
  Serial.println("Motors initialized");
}

void initializeUltrasonic() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  Serial.println("Ultrasonic sensor initialized");
}

void initializeServo() {
  radarServo.attach(SERVO_PIN);
  radarServo.write(0);
  currentAngle = 0;
  delay(500);
  Serial.println("Radar servo initialized");
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 50) {
    delay(500);
    Serial.print('.');
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP: "); Serial.println(WiFi.localIP());
    Serial.print("RSSI: "); Serial.print(WiFi.RSSI()); Serial.println(" dBm");
  } else {
    Serial.println("\nFailed to connect to WiFi.");
  }
}

// ————— Command Execution —————
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
  else if (command == "open") {
    executeOpenCommand();
  }
  else if (command == "radar_on") {
    startRadar();
  }
  else if (command == "radar_off") {
    stopRadar();
  }
  else if (command == "radar_test") {
    testRadarSystem();
  }
  else if (command.startsWith("radar_angle:")) {
    int angle = command.substring(12).toInt();
    setRadarAngle(angle);
  }
  else if (command.startsWith("speed:")) {
    int s = command.substring(6).toInt();
    if (s>=0 && s<=255) motorSpeed = s;
  }
  else if (command.startsWith("turn_speed:")) {
    int ts = command.substring(11).toInt();
    if (ts>=0 && ts<=255) turnSpeed = ts;
  }
  else if (command.startsWith("move_time:")) {
    int mt = command.substring(10).toInt();
    if (mt>=0 && mt<=5000) moveTime = mt;
  }
  else if (command.startsWith("turn_time:")) {
    int tt = command.substring(10).toInt();
    if (tt>=0 && tt<=5000) turnTime = tt;
  }
  else if (command.startsWith("sweep_speed:")) {
    int sd = command.substring(12).toInt();
    if (sd>=10 && sd<=1000) sweepDelay = sd;
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

void executeOpenCommand() {
  Serial.println("Executing OPEN command");
  int saved = currentAngle;
  radarServo.write(180); delay(200);
  radarServo.write(0);   delay(200);
  radarServo.write(90);  delay(300);
  radarServo.write(saved);
  currentAngle = saved;
  Serial.println("Open completed");
}

// ————— Radar Control —————
void startRadar() {
  radarEnabled = true;
  currentAngle  = 0;
  sweepDirection = 1;
  radarServo.write(currentAngle);
  Serial.println("Radar ON");
  if (client && client.connected()) client.println("RADAR_STATUS:ON");
}

void stopRadar() {
  radarEnabled = false;
  Serial.println("Radar OFF");
  if (client && client.connected()) client.println("RADAR_STATUS:OFF");
}

void setRadarAngle(int angle) {
  if (angle < 0 || angle > 180) return;
  currentAngle = angle;
  radarServo.write(angle);
  delay(200);
  float d = readUltrasonic();
  sendRadarReading(angle, d);
}

void updateRadarSweep() {
  if (millis() - lastSweepTime < (unsigned long)sweepDelay) return;
  currentAngle += sweepDirection * sweepStep;
  if (currentAngle >= 180) { currentAngle = 180; sweepDirection = -1; }
  else if (currentAngle <= 0) { currentAngle = 0; sweepDirection = 1; }
  radarServo.write(currentAngle);
  lastSweepTime = millis();
}

void sendRadarData() {
  if (millis() - lastRadarSend < (unsigned long)radarSendInterval) return;
  if (millis() - lastUltrasonicRead >= (unsigned long)ultrasonicReadInterval) {
    lastDistance = readUltrasonic();
    lastUltrasonicRead = millis();
  }
  sendRadarReading(currentAngle, lastDistance);
  lastRadarSend = millis();
}

void sendRadarReading(int angle, float distance) {
  if (client && client.connected()) {
    String msg = "RADAR:" + String(angle) + "," + String(distance,1);
    client.println(msg);
    static unsigned long lastDbg = 0;
    if (millis() - lastDbg > 500) {
      Serial.println("Sent: " + msg);
      lastDbg = millis();
    }
  }
}

// ————— Ultrasonic —————
float readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long dur = pulseIn(ECHO_PIN, HIGH, 30000);
  float dist = (dur * 0.034) / 2.0;
  if (dist <= 0 || dist > 400) dist = 400;
  return dist;
}

// ————— Tests —————
void testRadarSystem() {
  Serial.println("Testing radar sweep...");
  for (int a=0; a<=180; a+=30) {
    radarServo.write(a);
    delay(500);
    float d = readUltrasonic();
    Serial.printf("Angle %d°, Dist %.1fcm\n", a, d);
  }
  radarServo.write(90);
  currentAngle = 90;
  Serial.println("Radar test complete");
}

void testMotors() {
  Serial.println("Testing motors...");
  Serial.println("Forward");  moveForward();  delay(1000);
  Serial.println("Left");     turnLeft();     delay(1000);
  Serial.println("Right");    turnRight();    delay(1000);
  Serial.println("Backward"); moveBackward(); delay(1000);
  stopMotors();
  Serial.println("Motors test complete");
}

// ————— Movement —————
void moveForward() {
  digitalWrite(LEFT_MOTOR_IN1,  HIGH);
  digitalWrite(LEFT_MOTOR_IN2,  LOW);
  digitalWrite(RIGHT_MOTOR_IN1, HIGH);
  digitalWrite(RIGHT_MOTOR_IN2, LOW);
  delay(moveTime);
  stopMotors();
}

void moveBackward() {
  digitalWrite(LEFT_MOTOR_IN1,  LOW);
  digitalWrite(LEFT_MOTOR_IN2,  HIGH);
  digitalWrite(RIGHT_MOTOR_IN1, LOW);
  digitalWrite(RIGHT_MOTOR_IN2, HIGH);
  delay(moveTime);
  stopMotors();
}

void turnLeft() {
  digitalWrite(LEFT_MOTOR_IN1,  LOW);
  digitalWrite(LEFT_MOTOR_IN2,  HIGH);
  digitalWrite(RIGHT_MOTOR_IN1, HIGH);
  digitalWrite(RIGHT_MOTOR_IN2, LOW);
  delay(turnTime);
  stopMotors();
}

void turnRight() {
  digitalWrite(LEFT_MOTOR_IN1,  HIGH);
  digitalWrite(LEFT_MOTOR_IN2,  LOW);
  digitalWrite(RIGHT_MOTOR_IN1, LOW);
  digitalWrite(RIGHT_MOTOR_IN2, HIGH);
  delay(turnTime);
  stopMotors();
}

void stopMotors() {
  digitalWrite(LEFT_MOTOR_IN1,  LOW);
  digitalWrite(LEFT_MOTOR_IN2,  LOW);
  digitalWrite(RIGHT_MOTOR_IN1, LOW);
  digitalWrite(RIGHT_MOTOR_IN2, LOW);
}

// ————— Status —————
void sendStatus() {
  if (!client.connected()) return;
  client.println("STATUS:");
  client.println("Motor Speed: " + String(motorSpeed));
  client.println("Turn Speed: "  + String(turnSpeed));
  client.println("Move Time: "   + String(moveTime) + " ms");
  client.println("Turn Time: "   + String(turnTime) + " ms");
  client.println("Radar: "       + String(radarEnabled ? "ON" : "OFF"));
  client.println("Angle: "       + String(currentAngle) + "°");
  client.println("Sweep Delay: " + String(sweepDelay) + " ms");
  client.println("Last Dist: "   + String(lastDistance,1) + " cm");
  client.println("WiFi RSSI: "   + String(WiFi.RSSI()) + " dBm");
  client.println("Free Heap: "   + String(ESP.getFreeHeap()) + " bytes");
}

// ————— Emergency —————
void emergencyStop() {
  stopMotors();
  stopRadar();
  Serial.println("EMERGENCY STOP!");
}
