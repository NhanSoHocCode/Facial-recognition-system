from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

from face_auth_with_anti_spoofing_service import OptimizedAuthService 
from finger_auth_service import FingerprintAuthService

app = Flask(__name__)
ESP32_STREAM_URL = "http://192.168.25.158/"
# ip 192.168.25.18

# 1. KHỞI TẠO SERVICE
print("--- [INIT] DANG KHOI TAO SERVER VA AI MODELS ---")
try:
    face_auth_service = OptimizedAuthService()
    finger_auth_service = FingerprintAuthService()
    print("--- [INIT] AI READY! ---")
except Exception as e:
    print(f"--- [CRITICAL ERROR] Khong load duoc AI: {e}")
    face_auth_service = None
    finger_auth_service = None

def decode_image_from_raw(raw_image_data):
    try:
        width = 256
        height = 288
        expected_bytes_8bit = width * height      
        expected_bytes_4bit = width * height // 2 

        img_array = None

        if len(raw_image_data) == expected_bytes_4bit:
            packed_data = np.frombuffer(raw_image_data, dtype=np.uint8)
            high_nibbles = (packed_data >> 4) * 17 
            low_nibbles  = (packed_data & 0x0F) * 17
            full_img = np.empty(len(packed_data) * 2, dtype=np.uint8)
            full_img[0::2] = high_nibbles
            full_img[1::2] = low_nibbles
            img_array = full_img.reshape((height, width))

        elif len(raw_image_data) == expected_bytes_8bit:
            # Xử lý 8-bit
            img_array = np.frombuffer(raw_image_data, dtype=np.uint8).reshape((height, width))
        else:
            print(f"Loi kich thuoc anh: {len(raw_image_data)} bytes (Khong hop le)")
            return None

        if img_array is not None:
            return Image.fromarray(img_array).convert('L')
            
    except Exception as e:
        print(f"Loi giai ma anh: {e}")
    
    return None

# 3. LOGIC XÁC THỰC (AUTHENTICATION)
def logic_check_face(device_name, is_admin=False):
    if face_auth_service is None: return False
    if is_admin:
        face_auth_service.current_target_admin = True
    else: 
        face_auth_service.current_target_admin = False
    print(f"-> Checking Face ID from {device_name}")
    return face_auth_service.run_stream(ESP32_STREAM_URL, time_out=20)

def logic_process_fingerprint_auth(raw_image_data):
    print(f"-> Nhan duoc anh van tay de XAC THUC. Size: {len(raw_image_data)}")
    
    if raw_image_data:
        return finger_auth_service.authenticate(raw_image_data, threshold=0.7)
    return False

# 4. API ROUTES
@app.route('/api/upload_face', methods=['POST'])
def api_face():
    try:
        data = request.json
        device = data.get('device', 'Unknown')
        role_requested = data.get('role', 'USER')
        
        # Xác định có phải yêu cầu Admin hay không
        is_admin_flag = True if role_requested == 'ADMIN' else False
        
        if is_admin_flag:
            print(f"!!! [SECURITY] YEU CAU QUYEN ADMIN TU: {device} !!!")
        
        allow_access = face_auth_service.run_stream(ESP32_STREAM_URL, isAdmin=is_admin_flag, time_out=20)
        
        if allow_access:
            print(f">> [SUCCESS] Cho phep truy cap: {role_requested}")
            return jsonify({
                "allow": True, 
                "unlock_ms": 5000, 
                "message": f"Welcome {role_requested}"
            }), 200
        else:
            print(f">> [DENIED] Tu choi truy cap: {role_requested}")
            return jsonify({
                "allow": False, 
                "message": "Access Denied"
            }), 200

    except Exception as e:
        print(f"Error Face API: {e}")
        return jsonify({"allow": False, "message": str(e)}), 500

# --- API 2: NHẬN DIỆN VÂN TAY (ĐÃ THÊM CHECK SIZE) ---
@app.route('/api/upload_fingerprint_image', methods=['POST'])
def api_fingerprint_auth():
    print("\n--- NHAN DIEN VAN TAY QUA API ---")
    try:
        raw_data = request.data
        # Cảm biến AS608 gửi khoảng 36k bytes cho ảnh 4-bit
        if not raw_data or len(raw_data) < 30000: 
            print(">> [ERROR] Du lieu van tay qua nho hoac rong")
            return jsonify({"allow": False, "message": "Invalid data size"}), 400

        print(f"-> Processing Fingerprint... Size: {len(raw_data)} bytes")
        allow_access = finger_auth_service.authenticate(raw_data, threshold=0.7)

        return jsonify({
            "allow": allow_access,
            "unlock_ms": 5000, 
            "message": "Success" if allow_access else "Failed"
        })
    except Exception as e:
        print(f"Error Fingerprint API: {e}")
        return jsonify({"allow": False}), 500

# --- API 3: THÊM VÂN TAY MỚI (ENROLL - CHỨC NĂNG MỚI) ---
@app.route('/api/enroll_fingerprint', methods=['POST'])
def api_enroll_fingerprint():
    try:
        user_id = request.args.get('user_id')
        scan_num = request.args.get('scan_num', type=int, default=1)  # ⭐ THÊM DÒNG NÀY
        raw_data = request.data

        print(f"\n--- YEU CAU THEM VAN TAY MOI (LAN {scan_num}/3) ---")  # ⭐ SỬA DÒNG NÀY
        print(f"User ID: {user_id}")
        print(f"Scan số: {scan_num}/3")  # ⭐ THÊM
        print(f"Data Size: {len(raw_data)} bytes")

        if not user_id or not raw_data:
            return jsonify({"allow": False, "message": "Missing data"}), 400

        # ⭐ THÊM scan_num VÀO HÀM ENROLL
        result = finger_auth_service.enroll_finger(raw_data, user_id, scan_num)
        
        return jsonify({
            "allow": result.get("success", False),
            "completed": result.get("completed", False),  # ⭐ THÊM completed
            "message": f"Scan {scan_num}/3 processed"  # ⭐ SỬA MESSAGE
        })

    except Exception as e:
        print(f"Error Enroll API: {e}")
        return jsonify({"allow": False, "message": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)