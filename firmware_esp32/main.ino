#include <WiFi.h>
#include <HTTPClient.h>
#include <Keypad.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ESP32Servo.h>
#include <WebServer.h>

WebServer server(80); // Khởi tạo server tại cổng 80

// ================== WIFI & SERVER ==================
const char* ssid     = "Nhan";
const char* password = "27112004";
const char* ip       = "192.168.25.18";

String serverFaceUrl;
String serverFingerprintUrl; 
String serverEnrollUrl;      

const char* deviceName = "DangNgocNhan";

// ================== HARDWARE SERIAL CHO VÂN TAY ==================
HardwareSerial FPSerial(2); 
const uint8_t FP_ADDR[4] = {0xFF, 0xFF, 0xFF, 0xFF};

// ================== LCD & I/O ==================
LiquidCrystal_I2C lcd(0x27, 16, 2);

const byte ROWS = 4;
const byte COLS = 4;
char hexaKeys[ROWS][COLS] = {
  {'1','2','3','A'},
  {'4','5','6','B'},
  {'7','8','9','C'},
  {'*','0','#','D'}
};
byte rowPins[ROWS] = {4, 5, 18, 19};
byte colPins[COLS] = {23, 25, 26, 27};
Keypad keypad_key = Keypad(makeKeymap(hexaKeys), rowPins, colPins, ROWS, COLS);

Servo doorServo;
const int servoPin         = 32;
const int servoLockedPos   = 180;  
const int servoUnlockedPos = 90;   
const int buzzerPin = 15; 

const int sr503Pin = 14;
const unsigned long BACKLIGHT_TIMEOUT = 10000;   
unsigned long lastActivityTime = 0; 
bool isBacklightOn = true;          

// ================== HELPER FUNCTIONS ==================
void beep(int ms = 100) {
  if (buzzerPin >= 0) {
    pinMode(buzzerPin, OUTPUT);
    digitalWrite(buzzerPin, HIGH); delay(ms); digitalWrite(buzzerPin, LOW);
  }
}
void beepLong() { beep(500); delay(100); beep(500); }
void beepError() { beep(200); delay(50); beep(200); delay(50); beep(200); }

void manageBacklight() {
  if (digitalRead(sr503Pin) == HIGH) lastActivityTime = millis();
  if (millis() - lastActivityTime < BACKLIGHT_TIMEOUT) {
    if (!isBacklightOn) { lcd.backlight(); isBacklightOn = true; }
  } else {
    if (isBacklightOn) { lcd.noBacklight(); isBacklightOn = false; }
  }
}

void wakeUpScreen() { lastActivityTime = millis(); manageBacklight(); }

void lcdMessage(const String &line1, const String &line2 = "") {
  wakeUpScreen(); lcd.clear();
  lcd.setCursor(0,0); lcd.print(line1);
  if (line2.length() > 0) { lcd.setCursor(0,1); lcd.print(line2); }
}

void openDoor(unsigned long unlockMs = 5000) {
  doorServo.write(servoUnlockedPos); beep();
  delay(unlockMs);
  doorServo.write(servoLockedPos); beep();
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  lcdMessage("Connecting WiFi"); 
  WiFi.begin(ssid, password);
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500); retries++;
  }
  if (WiFi.status() == WL_CONNECTED) lcdMessage("WiFi OK", WiFi.localIP().toString());
  else lcdMessage("WiFi FAILED");
}

// ================== PROTOCOL VÂN TAY ==================
const uint32_t IMG_BYTES = 36864; 
uint8_t imageBuf[IMG_BYTES];       

void sendSimpleCommand(uint8_t cmd) {
  uint8_t pkt[12];
  pkt[0] = 0xEF; pkt[1] = 0x01; 
  pkt[2] = FP_ADDR[0]; pkt[3] = FP_ADDR[1]; pkt[4] = FP_ADDR[2]; pkt[5] = FP_ADDR[3];
  pkt[6] = 0x01; pkt[7] = 0x00; pkt[8] = 0x03; pkt[9] = cmd;  
  uint16_t sum = pkt[6] + pkt[7] + pkt[8] + pkt[9];
  pkt[10] = (sum >> 8) & 0xFF; pkt[11] = sum & 0xFF; 
  FPSerial.write(pkt, sizeof(pkt));
}

bool readPacket(uint8_t &pid, uint16_t &length, uint8_t *buf, size_t bufSize, uint32_t timeoutMs) {
  uint32_t start = millis();
  int state = 0;
  while (millis() - start < timeoutMs) {
    if (FPSerial.available()) {
      uint8_t b = FPSerial.read();
      if (state == 0 && b == 0xEF) state = 1;
      else if (state == 1 && b == 0x01) { state = 2; break; }
      else state = 0;
    }
  }
  if (state != 2) return false; 

  while (FPSerial.available() < 4 && (millis() - start < timeoutMs)) {}
  if (FPSerial.available() < 4) return false;
  uint8_t addr[4]; FPSerial.readBytes(addr, 4); 

  while (FPSerial.available() < 3 && (millis() - start < timeoutMs)) {}
  if (FPSerial.available() < 3) return false;
  
  pid = FPSerial.read();
  uint8_t lenHi = FPSerial.read();
  uint8_t lenLo = FPSerial.read();
  length = ((uint16_t)lenHi << 8) | lenLo;

  uint16_t payloadLen = length - 2; 
  if (payloadLen > bufSize) return false;

  uint16_t got = 0;
  while (got < payloadLen && (millis() - start < timeoutMs)) {
    if (FPSerial.available()) buf[got++] = FPSerial.read();
  }
  while (FPSerial.available() < 2 && (millis() - start < timeoutMs)) {}
  if (FPSerial.available() >= 2) { FPSerial.read(); FPSerial.read(); }
  
  return (got == payloadLen);
}

bool readAck(uint8_t &confirmCode, uint32_t timeoutMs = 1000) {
  uint8_t pid; uint16_t length; uint8_t data[16]; 
  if (!readPacket(pid, length, data, sizeof(data), timeoutMs)) return false;
  if (pid != 0x07) return false; 
  confirmCode = data[0]; return true;
}

bool captureImageToBuffer() {
  uint8_t confirm;
  sendSimpleCommand(0x01); 
  if (!readAck(confirm, 2000) || confirm != 0x00) return false;
  sendSimpleCommand(0x0A); 
  if (!readAck(confirm, 2000) || confirm != 0x00) return false;

  while(FPSerial.available()) FPSerial.read();
  uint32_t offset = 0; 
  uint8_t pktBuf[256 + 10]; 
  unsigned long streamStart = millis();

  while (offset < IMG_BYTES && (millis() - streamStart < 8000)) {
    uint8_t pid; uint16_t length;
    if (readPacket(pid, length, pktBuf, sizeof(pktBuf), 1000)) {
      uint16_t dataLen = (length >= 2) ? (length - 2) : 0;
      if (dataLen > 0 && (offset + dataLen <= IMG_BYTES)) {
        memcpy(imageBuf + offset, pktBuf, dataLen);
        offset += dataLen;
      }
      if (pid == 0x08) break; 
    }
  }
  return (offset >= (IMG_BYTES - 512)); 
}

// ================== NETWORK LOGIC ==================
bool parseAllowAndTime(const String &resp, unsigned long &unlockMsOut) {
  String s = resp; s.replace(" ", "");
  bool allow = s.indexOf("\"allow\":true") != -1;
  unlockMsOut = 3000;
  int idx = s.indexOf("\"unlock_ms\":");
  if (idx != -1) {
    String sub = s.substring(idx + 12);
    unlockMsOut = sub.toInt(); if (unlockMsOut == 0) unlockMsOut = 3000;
  }
  return allow;
}

bool sendCheckFingerprint(unsigned long &unlockMsOut) {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  HTTPClient http;
  http.begin(serverFingerprintUrl); 
  http.addHeader("Content-Type", "application/octet-stream");
  int code = http.POST(imageBuf, IMG_BYTES);
  if (code > 0) {
      String resp = http.getString(); http.end();
      return parseAllowAndTime(resp, unlockMsOut);
  }
  http.end(); return false;
}

// SỬA LỖI: Thêm khai báo unlockMsOut trong hàm Enroll
bool sendEnrollRequest(String userId, int scanNum = 1) {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  HTTPClient http;
  unsigned long unlockMsOut = 0;
  
  // ⭐ THÊM scan_num vào URL
  String url = serverEnrollUrl + "?user_id=" + userId + "&scan_num=" + String(scanNum);
  
  Serial.println("Enroll URL: " + url); // Debug
  
  http.begin(url); 
  http.addHeader("Content-Type", "application/octet-stream");
  int code = http.POST(imageBuf, IMG_BYTES);
  if (code > 0) {
      String resp = http.getString(); 
      http.end();
      Serial.println("Response: " + resp); // Debug
      return parseAllowAndTime(resp, unlockMsOut);
  }
  http.end(); 
  return false;
}

bool requestAccessFace(unsigned long &unlockMsOut, String role = "USER") {
  connectWiFi(); 
  HTTPClient http;
  http.begin(serverFaceUrl); 
  http.addHeader("Content-Type", "application/json");

  http.setTimeout(25000);
  
  String json = "{\"device\":\"" + String(deviceName) + "\",\"method\":\"face\",\"role\":\"" + role + "\"}";
  int code = http.POST(json);
  if (code > 0) {
    String response = http.getString(); http.end();
    return parseAllowAndTime(response, unlockMsOut);
  }
  http.end(); return false;
}

// ================== MENU FUNCTIONS ==================
String inputUserId() {
  String id = "";
  lcdMessage("Nhap ID User:", "Xong an phim #");
  while (true) {
    manageBacklight();
    char k = keypad_key.getKey();
    if (k == 'D') return "CANCEL"; 
    if (k == '#') { 
      if (id.length() > 0) return id;
      else { beepError(); continue; }
    }
    if (k >= '0' && k <= '9' && id.length() < 10) {
       id += k; lcd.setCursor(0, 1); lcd.print(id); beep();
    }
  }
}

void modeUnlockFingerprint() {
  lcdMessage("Quet de MO CUA", "Huy: Bam D");
  unsigned long start = millis();
  unsigned long unlockMs = 0;
  while (millis() - start < 10000) {
    if (keypad_key.getKey() == 'D') return; 
    if (captureImageToBuffer()) {
       lcdMessage("Dang kiem tra...", "Cho Server...");
       if (sendCheckFingerprint(unlockMs)) {
         lcdMessage("OK: Mo Cua", String(unlockMs/1000)+"s");
         openDoor(unlockMs);
       } else {
         lcdMessage("TU CHOI", "Khong khop!"); beepError(); delay(1000);
       }
       return; 
    }
    delay(50);
  }
}

void modeEnrollNewUser() {
  unsigned long dummyMs = 0;
  lcdMessage("Verify ADMIN...", "Look at Camera");
  
  // Xác thực ADMIN bằng khuôn mặt
  if (!requestAccessFace(dummyMs, "ADMIN")) {
    lcdMessage("ACCESS DENIED", "Not an Admin!"); 
    beepError(); 
    delay(2000); 
    return;
  }
  
  beepLong();
  
  // Nhập ID người dùng
  String uid = inputUserId(); 
  if (uid == "CANCEL") return;
  
  // ⭐ THAY ĐỔI: Quét 3 lần
  const int REQUIRED_SCANS = 3;
  int successfulScans = 0;
  
  for (int scanNum = 1; scanNum <= REQUIRED_SCANS; scanNum++) {
    // Hiển thị hướng dẫn cho lần quét
    String scanMsg = "Lan quet " + String(scanNum) + "/" + String(REQUIRED_SCANS);
    String guideMsg = "ID: " + uid + " -> #";
    
    lcdMessage(scanMsg, guideMsg);
    
    // Đợi quét vân tay
    unsigned long start = millis();
    bool captured = false;
    
    while (millis() - start < 15000) { 
      char k = keypad_key.getKey();
      if (k == 'D') {
        lcdMessage("Da huy", "Quet lan " + String(scanNum));
        delay(1000);
        return;
      }
      
      if (k == '#') {
        // Bắt đầu quét khi nhấn #
        lcdMessage("Dang quet...", "Giu ngon tay");
        if (captureImageToBuffer()) {
          captured = true;
          break;
        } else {
          lcdMessage("Quet that bai", "Thu lai...");
          beepError();
          delay(1000);
          lcdMessage(scanMsg, guideMsg);
          start = millis(); // Reset timer
        }
      }
      
      manageBacklight();
      delay(50);
    }
    
    if (!captured) {
      lcdMessage("Loi: Het gio", "Lan " + String(scanNum));
      beepError();
      delay(1500);
      return;
    }
    
    // Gửi dữ liệu lên server
    lcdMessage("Gui data...", "Lan " + String(scanNum));
    
    if (sendEnrollRequest(uid, scanNum)) {
      successfulScans++;
      lcdMessage("Thanh cong!", "Da quet: " + String(successfulScans));
      beep();
      
      // Nếu chưa đủ 3 lần, chờ giữa các lần quét
      if (scanNum < REQUIRED_SCANS) {
        delay(1000);
        lcdMessage("Chuan bi lan", String(scanNum + 1) + "...");
        delay(1000);
      }
    } else {
      lcdMessage("That bai lan", String(scanNum));
      beepError();
      delay(1500);
      return;
    }
  }
  
  // Kết quả cuối cùng
  if (successfulScans == REQUIRED_SCANS) {
    lcdMessage("HOAN TAT!", "User: " + uid);
    beepLong(); beepLong();
  } else {
    lcdMessage("KHONG HOAN TAT", String(successfulScans) + "/" + String(REQUIRED_SCANS));
    beepError();
  }
  
  delay(2000);
}

void modeUnlockFace() {
  lcdMessage("Camera FaceID...", "Huy: Bam D");
  unsigned long unlockMs = 0;
  if (requestAccessFace(unlockMs, "USER")) { 
    lcdMessage("Face OK", "Mo cua..."); openDoor(unlockMs); 
  } else { 
    lcdMessage("Face Reject"); beepError(); delay(1500); 
  }
}

void showMainMenu() {
  lcd.clear(); 
  lcd.setCursor(0,0); lcd.print("A:Finger  B:Face"); 
  lcd.setCursor(0,1); lcd.print("C:Add Fg  D:Back");
}

void handleRemoteUnlock() {
  // Kiểm tra nếu có tham số unlock_ms gửi kèm (tùy chọn)
  unsigned long unlockMs = 5000; 
  if (server.hasArg("duration")) {
    unlockMs = server.arg("duration").toInt();
  }

  server.send(200, "text/plain", "OK: Door Opening"); // Phản hồi lại cho Python
  lcdMessage("Remote Unlock", "By Server...");
  openDoor(unlockMs);
  showMainMenu();
}

void setup() {
  Serial.begin(115200);
  pinMode(sr503Pin, INPUT);
  Wire.begin(21, 22); lcd.init(); lcd.backlight();
  doorServo.attach(servoPin); doorServo.write(servoLockedPos);
  FPSerial.setRxBufferSize(1024);
  FPSerial.begin(57600, SERIAL_8N1, 16, 17);
  serverFaceUrl        = "http://" + String(ip) + ":5000/api/upload_face";
  serverFingerprintUrl = "http://" + String(ip) + ":5000/api/upload_fingerprint_image";
  serverEnrollUrl      = "http://" + String(ip) + ":5000/api/enroll_fingerprint";
  connectWiFi();
  showMainMenu();

  server.on("/remote_unlock", HTTP_POST, handleRemoteUnlock);
  server.begin();
  Serial.println("HTTP Server started");
}

void loop() {
  server.handleClient();
  manageBacklight();
  char k = keypad_key.getKey();
  if (k) {
    wakeUpScreen(); beep();
    switch (k) {
      case 'A': modeUnlockFingerprint(); showMainMenu(); break;
      case 'B': modeUnlockFace(); showMainMenu(); break;
      case 'C': modeEnrollNewUser(); showMainMenu(); break;
      case 'D': showMainMenu(); break;
    }
  }
  delay(50);
}