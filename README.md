# HỆ THỐNG KHÓA CỬA THÔNG MINH TÍCH HỢP NHẬN DIỆN VÂN TAY VÀ KHUÔN MẶT

## 📖 Giới thiệu
Đây là đồ án với mục tiêu **Xây dựng hệ thống khóa cửa thông minh tích hợp nhận diện vân tay và khuôn mặt thông minh, có khả năng chống che giấu, chống deepfake.**.  
Hệ thống được xây dựng nhằm **Tìm hiểu về các kỹ thuật và công nghệ hiện đại trong lĩnh vực nhận diện vân tay và khuôn mặt thông minh, sử dụng các công nghệ và phương pháp hiện đại**.

Đồ án tập trung vào các chức năng chính:
- ✅ Xác nhận vân tay và khuôn mặt thông minh
- ✅ Chống che giấu, chống deepfake
- ✅ Chức năng quản lý người dùng
- ✅ Chức năng quản lý log
- ✅ Mở rộng mô hình trên thiết bị phần cứng Arduino ESP32
---

## 👥 Thông tin nhóm thực hiện trong dự án này 

| STT | Họ và tên | MSSV | Vai trò | Công việc đảm nhận trong hệ thống |
|-----|----------|------|--------|-------------------|
| 1 | [Trần Hồng Quang Lê] | [23110251] | [Trưởng nhóm] | [Nguyên cứu phát triển mô hình nhận diện khuôn mặt, phát hiện giả mạo] |
| 2 | [Đặng Ngọc Nhân] | [23110279] | [Thành viên] | [Triển khai hệ thống, lập trình phần mềm và phần cứng, viết readmi hướng dẫn] |
| 3 | [Cáp Thanh Nhàn] | [23110276] | [Thành viên] | [ Nguyên cứu phát triển mô hình nhận diện vân tay] |
| 4 | [Phạm Thị Tuyết Minh] | [23110268] | [Thành viên] | [Nguyên cứu phát triển mô hình nhận diện vân tay, Thiết kế giao diện phần mềm] |

---

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ lập trình:** Python
- **Framework / Thư viện:** Flask, FastAPI, PyTorch, OpenCV, Mediapipe, ONNX Runtime
- **Cơ sở dữ liệu:** MySQL
- **Công cụ hỗ trợ:** Git, VS Code
- **Môi trường chạy:** Windows

---

## 📂 Cấu trúc thư mục

```
ImageProcessing_BE/
├── backend/
│   ├── api/
│   │   ├── face_auth_with_anti_spoofing_service.py
│   │   ├── finger_auth_service.py
│   │   ├── main.py
│   │   └── opening_door.py
│   ├── core/
│   │   ├── database_mysql.py
│   │   ├── face_recognition_with_anti_spoofing.py
│   │   └── fingerPrint_recognition.py
│   ├── fingerprint_images/
│   ├── models/
│   ├── static/
│   └── ui/
│       ├── login_window.py
│       └── main_dashboard.py
├── database/
│   └── scriptMySQL.sql
├── firmware_esp32/
│   ├── circuit_Diagram.txt
│   └── main.ino
└── readmi.md
```
## ▶️ Hướng dẫn chạy phần mềm

### 1️⃣ Yêu cầu môi trường
- Hệ điều hành: [Windows]
- Ngôn ngữ / Nền tảng: [Python <3.11]
- Công cụ cần thiết:  
  - [VS Code]
  - [MySQL]
- Thư viện / package cần cài đặt:  
  - Mở terminal chạy lệnh: **pip install -r requirements.txt**
  - Lưu ý bạn có thể cài vào môi trường ảo hoặc trực tiếp vào môi trường hệ thống, nhưng mình khuyên bạn nên dùng anaconda để chạy môi trường ảo vì nó sẽ ổn định hơn
---

### 2️⃣ Thiết lập cơ sở dữ liệu
- Mở MySQL Workbench
- Chạy script MySQL từ file **scriptMySQL.sql** trong thư mục database
- Lưu ý nếu bạn đã có database có tên **smartKey_app** thì bạn cần xóa database đó trước khi chạy script, chỉnh lại cấu hình cơ sở dữ liệu trong file **database_mysql.py** trong thư mục backend/core

### 3️⃣ Tải mô hình trọng số vào thư mục models
- Tải checkpoint từ link drive: **https://drive.google.com/drive/folders/1EMVQCYNbNImrGV94Lv-PS0pYaVebJjVE** vào thư mục backend/models
- Lưu ý nếu bạn đã có checkpoint trong thư mục models thì bạn cần xóa checkpoint đó trước khi tải checkpoint mới.

### 4️⃣ Cài đặt phần mềm và chạy phần mềm
- Mở terminal chạy lệnh: **python backend/ui/login_window.py** vì dự án này có tích hợp phần cứng nên tôi chỉ hướng dẫn bạn chạy phần mềm để kiểm tra tính chính xác của mô hình nhận diện khuôn mặt kết hợp với các kỹ thuật chống giả mạo, và cách thực hiện thì bên phần báo cáo có hướng dẫn chi tiết.
- Mở terminal chạy lệnh: **python backend/api/main.py** cái này để chạy phần cứng nên bạn không cần chạy nó, chỉ có nhóm mình chạy được nó để demo phần cứng cho giảng viên


## Tác giả viết readmi hướng dẫn
- Đặng Ngọc Nhân