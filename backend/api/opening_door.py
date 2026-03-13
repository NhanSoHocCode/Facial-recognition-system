import requests

# Thay địa chỉ IP này bằng IP của ESP32 hiện thị trên LCD
ESP32_IP = "192.168.25.58" 

def unlock_door_remote(duration_ms=5000):
    url = f"http://{ESP32_IP}/remote_unlock"
    data = {'duration': duration_ms}
    try:
        response = requests.post(url, data=data, timeout=5)
        if response.status_code == 200:
            print("Lệnh mở cửa đã được gửi thành công!")
            print("Phản hồi từ ESP32:", response.text)
        else:
            print("Lỗi server:", response.status_code)
    except Exception as e:
        print("Không thể kết nối tới ESP32:", e)
