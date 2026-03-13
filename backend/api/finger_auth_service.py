import numpy as np
from PIL import Image
from pathlib import Path
import datetime
import sys

current_file = Path(__file__).resolve()
backend_dir = current_file.parent.parent
sys.path.append(str(backend_dir))

from core.database_mysql import MySQLManager
from core.fingerPrint_recognition import FingerprintRecognizer

CKPT_FINGER = backend_dir / "models" / "best_arcface_fingerprint.pth"
SAVE_DIR = backend_dir / "fingerprint_images"

class FingerprintAuthService:
    def __init__(self):
        print(">> [Fingerprint Service] Initializing...")
        
        # 1. Tạo thư mục lưu ảnh
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 2. Khởi tạo Database
        try:
            self.db = MySQLManager()
            print("   -> Database Connected.")
        except Exception as e:
            print(f"   -> [ERROR] DB Connection failed: {e}")
            self.db = None

        # 3. Khởi tạo AI Model
        try:
            self.recognizer = FingerprintRecognizer(ckpt_path=str(CKPT_FINGER))
            print("   -> AI Model Loaded.")
        except Exception as e:
            print(f"   -> [ERROR] AI Model failed: {e}")
            self.recognizer = None
            
        print(">> [Fingerprint Service] Ready.")

    def _decode_image(self, raw_data, width=256, height=288):
        """
        Hàm nội bộ: Giải mã dữ liệu byte từ cảm biến thành ảnh Numpy
        """
        expected_bytes_8bit = width * height      
        expected_bytes_4bit = width * height // 2 

        if len(raw_data) == expected_bytes_4bit:
            # Giải nén 4-bit
            packed_data = np.frombuffer(raw_data, dtype=np.uint8)
            high_nibbles = (packed_data >> 4) * 17 
            low_nibbles  = (packed_data & 0x0F) * 17
            full_img = np.empty(len(packed_data) * 2, dtype=np.uint8)
            full_img[0::2] = high_nibbles
            full_img[1::2] = low_nibbles
            return full_img.reshape((height, width))

        elif len(raw_data) == expected_bytes_8bit:
            # Raw 8-bit
            return np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))
        
        else:
            print(f"   -> [ERROR] Kích thước dữ liệu không hợp lệ: {len(raw_data)} bytes")
            return None

    def authenticate(self, raw_data, threshold=0.6):
        """
        Hàm xử lý chính: Nhận raw data -> Trả về kết quả User
        """
        if self.recognizer is None or self.db is None:
            print("!!! Service chưa sẵn sàng (Lỗi Init)")
            return False

        try:
            # 1. Giải mã ảnh (Kết quả là Numpy Array)
            img_array = self._decode_image(raw_data)
            if img_array is None:
                return False

            # 2. Lưu ảnh để log
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"fp_{timestamp}.png"
            file_path = SAVE_DIR / filename
            
            img_pil = Image.fromarray(img_array).convert('L')
            img_pil.save(file_path)
            print(f"   -> Đã lưu ảnh: {filename}")

            # 3. Trích xuất đặc trưng (Embedding)
            emb_vector = self.recognizer.extract(img_array)

            # 4. Tìm kiếm trong Database
            found_user = self.db.find_user_by_face(emb_vector, threshold=0.6, isFace=False)

            if found_user:
                print(f"   => [SUCCESS] User: {found_user['name']} (Score: {found_user['similarity']:.4f})")
                return True
            else:
                print("   => [FAILED] Không tìm thấy vân tay phù hợp.")
                return False

        except Exception as e:
            print(f"   -> [ERROR] Lỗi xử lý vân tay: {e}")
            import traceback
            traceback.print_exc()
            return False
    def enroll_finger(self, raw_data, user_id, scan_num=1):
        """
        Hàm đăng ký vân tay mới - Lưu 3 embedding riêng biệt
        scan_num: số thứ tự lần quét (1, 2, 3)
        """
        if self.recognizer is None or self.db is None:
            print("!!! Service chưa sẵn sàng")
            return {"success": False, "completed": False}

        try:
            # 1. Giải mã ảnh
            img_array = self._decode_image(raw_data)
            if img_array is None:
                return {"success": False, "completed": False}

            # 2. Lưu ảnh với tên user_id.scan_num.png
            file_path = SAVE_DIR / f"{user_id}.{scan_num}.png"
            
            img_pil = Image.fromarray(img_array).convert('L')
            img_pil.save(file_path)
            print(f"   => Đã lưu ảnh vân tay lần {scan_num}: {file_path.name}")

            # 3. Trích xuất embedding và lưu vào DB NGAY
            emb_vector = self.recognizer.extract(img_array)
            
            # ⭐ LƯU EMBEDDING VÀO DB NGAY, không đợi scan 3
            self.db.add_embedding_recognition(user_id, emb_vector, isFace=False)
            print(f"   => Đã lưu embedding lần {scan_num} vào database")

            # 4. Kiểm tra đã đủ 3 lần chưa
            if scan_num == 3:
                print(f"   => [COMPLETED] Đã đăng ký đủ 3 embedding cho User {user_id}")
                return {"success": True, "completed": True}
            else:
                print(f"   => [PROGRESS] Đã quét {scan_num}/3 cho User {user_id}")
                return {"success": True, "completed": False}

        except Exception as e:
            print(f"   -> [ERROR] Lỗi đăng ký: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "completed": False}