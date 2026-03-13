import sys
import torch
import cv2
import time
import numpy as np
import threading
import urllib.request
from pathlib import Path
from queue import Queue

# ⭐ THÊM MEDIAPIPE
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- SETUP PATH ---
current_file = Path(__file__).resolve()
backend_path = current_file.parent.parent
sys.path.append(str(backend_path))

from core.database_mysql import MySQLManager
from core.face_recognition_with_anti_spoofing import RealtimeDeepFakeDetector_Optimized

backend_dir = Path(__file__).resolve().parent.parent 
RECOG_PATH = backend_dir / "models" / "faceRecognition_arcface_ckpt(2).pth"
YOLO_PATH = backend_dir / "models" / "yolov8s-face-lindevs.onnx"
DEEPFAKE_PATH = backend_dir / "models" / "deepfake_best5.pth"
ALIGNMENT_PATH = backend_dir / "models" / "face_landmarker.task"  # ⭐ Thêm path

# THREAD: CAMERA STREAM
class VideoStreamThread:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        return self

    def update(self):
        try:
            stream = urllib.request.urlopen(self.url, timeout=5)
            bytes_data = bytes()
            while not self.stopped:
                try:
                    bytes_data += stream.read(4096)
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            self.frame = img
                except Exception:
                    time.sleep(0.01)
        except Exception as e:
            print(f"Cam error: {e}")

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# ⭐ SERVICE CHÍNH: AUTH WITH DEEPFAKE CHECK + ALIGNMENT
class OptimizedAuthService(RealtimeDeepFakeDetector_Optimized):
    def __init__(self, use_alignment=True):
        # 1. KIỂM TRA FILE
        if not YOLO_PATH.exists(): raise FileNotFoundError(f"Thieu file: {YOLO_PATH}")
        if not DEEPFAKE_PATH.exists(): raise FileNotFoundError(f"Thieu file: {DEEPFAKE_PATH}")
        if not RECOG_PATH.exists(): raise FileNotFoundError(f"Thieu file: {RECOG_PATH}")
        
        # ⭐ Kiểm tra alignment model nếu bật
        if use_alignment and not ALIGNMENT_PATH.exists():
            print(f"⚠ Warning: Alignment model not found at {ALIGNMENT_PATH}")
            print("  Continuing without alignment...")
            use_alignment = False

        # 2. GỌI SUPER INIT (⭐ Thêm alignment params)
        super().__init__(
            yolo_path=str(YOLO_PATH),
            deepfake_model_path=str(DEEPFAKE_PATH),
            face_recognition_ckpt=str(RECOG_PATH),
            alignment_model_path=str(ALIGNMENT_PATH),  # ⭐ Thêm
            use_alignment=use_alignment  # ⭐ Thêm
        )

        # 3. Init Database
        self.db = MySQLManager()
        self.recognition_threshold = 0.85
        self.current_target_admin = False
        
        print(">> [System] Service Ready!")
        if self.use_alignment:
            print(">> [System] Face Alignment: ENABLED ✓")
        else:
            print(">> [System] Face Alignment: DISABLED")

    def inference_worker(self):
        """
        ⭐ Luồng xử lý AI chạy ngầm (với alignment support)
        """
        while self.running:
            try:
                data = self.inference_queue.get(timeout=0.1)
                if data is None: break

                frames_tensor = data 

                with torch.no_grad():
                    # --- DEEPFAKE DETECTION ---
                    B, T = frames_tensor.shape[:2]
                    feats = []
                    for t in range(T):
                        frame_t = frames_tensor[:, t]
                        ft = self.deepfake_model.backbone(frame_t)
                        feats.append(ft)
                    feats = torch.stack(feats, dim=1)
                    
                    x = self.deepfake_model.temporal_encoder(feats)
                    logits = self.deepfake_model.classifier(x)
                    prob = torch.sigmoid(logits).item()

                    self.stats['total_inferences'] += 1
                    self.score_history.append(prob)
                    smooth_prob = sum(self.score_history) / len(self.score_history)
                    print(f"Smooth prob: {smooth_prob}, threshold: {self.threshold}")
                    label = "FAKE" if smooth_prob > self.threshold else "REAL"

                    embedding = None
                    recognition_result = (False, None)

                    if label == "REAL":
                        last_frame_features = feats[:, -1, :, :, :]
                        embedding_tensor = self.face_part2(last_frame_features)
                        embedding = embedding_tensor.cpu().numpy().flatten().copy()

                        self.stats['face_recognition_runs'] += 1
                        self.stats['real_detected'] += 1

                        found_user = self.db.find_user_by_face(
                            embedding, 
                            threshold=self.recognition_threshold,
                            isAdmin=self.current_target_admin
                        )
                        if found_user:
                            recognition_result = (True, found_user)
                    else:
                        self.stats['fake_detected'] += 1
                    
                    self.result_queue.put((label, smooth_prob, embedding, recognition_result))

            except Exception: pass

    def run_stream(self, stream_url, isAdmin=False, time_out=20):
        """
        ⭐ HOÀN THIỆN VỚI FIX KHÔNG CÓ KHUÔN MẶT:
        - Nếu FAKE → đợi hết 20s, nếu không có REAL → đóng ngay
        - Nếu REAL + nhận diện đúng → mở ngay không cần đếm 20s
        - Nếu không có khuôn mặt quá 10s → timeout sớm
        """
        self.current_target_admin = isAdmin

        # Reset state
        self.score_history.clear()
        self.frame_buffer.clear()
        self.is_initial_fill = True
        self.last_label = "Initializing..."
        self.last_prob = 0.0
        self.last_embedding = None
        self.last_recognition_result = (False, None)

        # ⭐ Reset statistics
        self.stats['total_inferences'] = 0
        self.stats['fake_detected'] = 0
        self.stats['real_detected'] = 0
        self.stats['face_recognition_runs'] = 0
        self.stats['alignment_success'] = 0
        self.stats['alignment_fail'] = 0

        # Clear result queue
        try:
            while not self.result_queue.empty():
                self.result_queue.get_nowait()
        except Exception:
            pass

        print(f"\n>> [System] Connecting Camera... (Filter Admin: {isAdmin})")

        # 1) Khởi động Camera
        video_stream = VideoStreamThread(stream_url).start()

        check_cam_start = time.time()
        while video_stream.read() is None:
            if time.time() - check_cam_start > 5:
                video_stream.stop()
                print(">> [ERROR] Cannot connect to camera")
                return False
            time.sleep(0.1)

        # 2) Khởi động AI Thread
        if self.use_threading:
            self.running = True
            self.inference_thread = threading.Thread(
                target=self.inference_worker,
                daemon=True
            )
            self.inference_thread.start()

        print(f">> [System] Scanning started... (Time limit: {time_out}s)")
        if self.use_alignment:
            print(">> [System] Face Alignment: ACTIVE")

        start_time = time.time()
        auth_success = False
        
        # ⭐ TRACKING VARIABLES - THÊM BIẾN THEO DÕI KHÔNG CÓ MẶT
        last_fake_time = None  # Thời điểm phát hiện FAKE gần nhất
        has_real_detection = False  # Đã từng phát hiện REAL chưa
        consecutive_real_count = 0
        REQUIRED_FRAMES = 1  # Số frame REAL liên tiếp cần để xác thực
        
        # ⭐ THÊM: BIẾN THEO DÕI KHÔNG CÓ KHUÔN MẶT
        last_face_time = start_time  # Thời điểm cuối cùng có khuôn mặt
        no_face_start_time = None  # Thời điểm bắt đầu không có mặt
        MAX_NO_FACE_TIME = 10  # Timeout sau 10s không có mặt
        WARNING_NO_FACE_TIME = 5  # Cảnh báo sau 5s không có mặt
        
        # ⭐ THÊM: THỐNG KÊ FRAME
        total_frames = 0
        frames_with_face = 0
        frames_without_face = 0

        # ⭐ THÊM: BIẾN THEO DÕI THỜI GIAN
        time_since_last_face = 0  # Khởi tạo biến

        try:  # ⭐ THÊM TRY Ở ĐÂY
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                remaining_time = max(0, time_out - elapsed_time)
                total_frames += 1

                # ⭐ CHECK TIMEOUT TỔNG
                if elapsed_time > time_out:
                    print("\n>> [TIMEOUT] Access DENIED - Time limit exceeded")
                    break

                # ⭐ CHECK TIMEOUT KHÔNG CÓ MẶT
                time_since_last_face = current_time - last_face_time
                if time_since_last_face > MAX_NO_FACE_TIME:
                    print(f"\n>> [NO FACE TIMEOUT] No face detected for {int(time_since_last_face)}s")
                    break
                    
                # ⭐ CẢNH BÁO KHÔNG CÓ MẶT
                if time_since_last_face > WARNING_NO_FACE_TIME:
                    warning_seconds = int(time_since_last_face)
                    if warning_seconds % 2 == 0:  # Nhấp nháy cảnh báo
                        print(f"\r>> [WARNING] No face for {warning_seconds}s... {int(remaining_time)}s left", end="")

                frame = video_stream.read()
                if frame is None:
                    time.sleep(0.01)  # ⭐ Giảm CPU khi không có frame
                    continue

                # 3) LẤY KẾT QUẢ MỚI từ AI THREAD
                got_new_result = False

                if self.use_threading:
                    try:
                        res = None
                        # ⭐ CHỈ LẤY KẾT QUẢ MỚI NHẤT, BỎ QUA CÁI CŨ
                        while not self.result_queue.empty():
                            res = self.result_queue.get_nowait()
                            got_new_result = True
                    except Exception:
                        pass

                    if got_new_result and res is not None:
                        self.last_label, self.last_prob, self.last_embedding, self.last_recognition_result = res
                        if hasattr(self, 'slide_buffer'):
                            self.slide_buffer()

                # Process frame - THÊM TRY-CATCH ĐỂ TRÁNH CRASH
                bbox = None
                try:
                    bbox, _, _, _ = self.process_frame_fast(frame)
                except Exception as e:
                    print(f"\r>> [ERROR] Face detection error: {str(e)[:50]}", end="")
                    bbox = None
                
                # ⭐ CẬP NHẬT THỐNG KÊ CÓ/KHÔNG CÓ MẶT
                if bbox:
                    frames_with_face += 1
                    last_face_time = current_time
                    no_face_start_time = None
                else:
                    frames_without_face += 1
                    if no_face_start_time is None:
                        no_face_start_time = current_time

                label = self.last_label
                rec_result = self.last_recognition_result

                # ⭐ XỬ LÝ KẾT QUẢ MỚI
                if got_new_result and label != "Initializing...":
                    if label == "FAKE":
                        # ⭐ GHI NHẬN THỜI ĐIỂM FAKE
                        last_fake_time = current_time
                        has_real_detection = False
                        consecutive_real_count = 0
                        self.last_embedding = None
                        self.last_recognition_result = (False, None)
                        print(f"\r>> [FAKE DETECTED] Waiting for REAL detection... ({int(remaining_time)}s left)", end="")

                    elif label == "REAL":
                        # ⭐ ĐÃ CÓ REAL → RESET FLAG
                        has_real_detection = True
                        
                        # Kiểm tra xem có tìm thấy User trong DB không
                        if rec_result and rec_result[0] and rec_result[1] is not None:
                            consecutive_real_count += 1
                            user_name = rec_result[1].get('name', 'Unknown')
                            user_type = "Admin" if rec_result[1].get('is_admin', False) else "User"
                            
                            print(f"\r>> [VERIFYING] {user_type}: {user_name} ({consecutive_real_count}/{REQUIRED_FRAMES})", end="")

                            # ⭐ Đủ SỐ FRAME → MỞ NGAY
                            if consecutive_real_count >= REQUIRED_FRAMES:
                                print(f"\n>> [✓ SUCCESS] {user_type} Verified: {user_name}")
                                auth_success = True
                                break
                        else:
                            # REAL nhưng không nhận diện được (Unknown)
                            consecutive_real_count = 0
                            print(f"\r>> [SCANNING] Unknown Face - {int(remaining_time)}s left          ", end="")

                # ⭐ LOGIC ĐÓNG SỚM KHI FAKE
                # Nếu đã phát hiện FAKE và sau 20s không có REAL nào → đóng
                if last_fake_time is not None and not has_real_detection:
                    if elapsed_time >= time_out:
                        print("\n>> [✗ DENIED] FAKE detected - No valid REAL detection in time limit")
                        break

                # 5) VẼ UI
                h, w = frame.shape[:2]
                
                if bbox:
                    x1, y1, x2, y2, _ = bbox
                    
                    # ⭐ Xác định trạng thái để vẽ màu
                    if label == "FAKE":
                        color = (0, 0, 255)  # Đỏ
                        status = "⚠ FAKE DETECTED"
                    elif label == "REAL":
                        if rec_result and rec_result[0]:
                            color = (0, 255, 0)  # Xanh lá
                            status = f"✓ {rec_result[1].get('name', 'User')}"
                        else:
                            color = (0, 165, 255)  # Cam
                            status = "? UNKNOWN FACE"
                    else:
                        color = (255, 165, 0)  # Vàng
                        status = "⌛ INITIALIZING..."

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, status, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # ⭐ VẼ THÔNG BÁO KHI KHÔNG CÓ MẶT
                    no_face_text = "NO FACE DETECTED"
                    if no_face_start_time:
                        no_face_seconds = int(current_time - no_face_start_time)
                        no_face_text = f"NO FACE FOR {no_face_seconds}s"
                    
                    # Vẽ background cho text
                    text_size = cv2.getTextSize(no_face_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = h // 2
                    
                    # Màu sắc theo thời gian không có mặt
                    if no_face_start_time and (current_time - no_face_start_time) > 5:
                        text_color = (0, 0, 255)  # Đỏ
                        bg_color = (0, 0, 100)
                    else:
                        text_color = (0, 255, 255)  # Vàng
                        bg_color = (0, 100, 100)
                    
                    # Vẽ background
                    cv2.rectangle(frame, (text_x-10, text_y-35), 
                                (text_x + text_size[0] + 10, text_y + 10), 
                                bg_color, -1)
                    
                    # Vẽ text
                    cv2.putText(frame, no_face_text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                    
                    # Vẽ icon mặt với dấu hỏi
                    face_center_x, face_center_y = w // 2, h // 3
                    radius = 50
                    cv2.circle(frame, (face_center_x, face_center_y), radius, text_color, 3)
                    cv2.putText(frame, "?", (face_center_x-10, face_center_y+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)
                    
                    # Vẽ mũi tên hướng dẫn
                    arrow_start = (face_center_x, face_center_y + radius + 30)
                    arrow_end = (face_center_x, face_center_y + radius + 80)
                    cv2.arrowedLine(frame, arrow_start, arrow_end, text_color, 2, tipLength=0.3)
                    cv2.putText(frame, "Look here", (face_center_x-40, arrow_end[1]+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                # ⭐ Countdown bar CHÍNH
                bar_width = 200
                bar_height = 20
                cv2.rectangle(frame, (10, 10), (10 + bar_width, 10 + bar_height), (50, 50, 50), -1)
                
                # Màu bar: Xanh nếu có REAL, Đỏ nếu FAKE, Vàng nếu không có mặt
                if has_real_detection:
                    bar_color = (0, 165, 255)  # Cam
                elif label == "FAKE":
                    bar_color = (0, 0, 255)  # Đỏ
                else:
                    bar_color = (0, 255, 255)  # Vàng
                    
                fill_width = int((remaining_time/time_out) * bar_width)
                cv2.rectangle(frame, (10, 10), (10 + fill_width, 10 + bar_height), 
                            bar_color, -1)
                cv2.putText(frame, f"Time: {int(remaining_time)}s", (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # ⭐ THÊM: NO-FACE COUNTDOWN BAR
                if no_face_start_time:
                    no_face_bar_y = 40
                    no_face_bar_width = 150
                    no_face_elapsed = current_time - no_face_start_time
                    no_face_ratio = min(1.0, no_face_elapsed / MAX_NO_FACE_TIME)
                    
                    # Nền
                    cv2.rectangle(frame, (10, no_face_bar_y), 
                                (10 + no_face_bar_width, no_face_bar_y + 10), 
                                (50, 50, 50), -1)
                    
                    # Thanh tiến trình
                    fill_width = int(no_face_ratio * no_face_bar_width)
                    if no_face_ratio > 0.7:
                        bar_color = (0, 0, 255)  # Đỏ
                    elif no_face_ratio > 0.3:
                        bar_color = (0, 165, 255)  # Cam
                    else:
                        bar_color = (0, 255, 255)  # Vàng
                        
                    cv2.rectangle(frame, (10, no_face_bar_y), 
                                (10 + fill_width, no_face_bar_y + 10), 
                                bar_color, -1)
                    
                    cv2.putText(frame, "No-face timer", (10, no_face_bar_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # ⭐ Hiển thị thống kê alignment
                if self.use_alignment:
                    total_align = self.stats['alignment_success'] + self.stats['alignment_fail']
                    if total_align > 0:
                        success_rate = (self.stats['alignment_success'] / total_align) * 100
                        align_text = f"Aligned: {self.stats['alignment_success']}/{total_align} ({success_rate:.1f}%)"
                        cv2.putText(frame, align_text, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Hiển thị stats
                stats_y = 90
                cv2.putText(frame, f"Frames: {total_frames}", (10, stats_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Face: {frames_with_face} | NoFace: {frames_without_face}", 
                        (10, stats_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"REAL: {self.stats['real_detected']} | FAKE: {self.stats['fake_detected']}", 
                        (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # ⭐ THÊM: FRAME RATE
                if elapsed_time > 0:
                    fps = total_frames / elapsed_time
                    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Secure Face Auth with Alignment", frame)
                
                # ⭐ XỬ LÝ PHÍM NHANH HƠN
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n>> [USER QUIT] Manual exit")
                    break
                elif key == ord(' '):  # ⭐ THÊM: PAUSE/RESUME
                    print("\n>> [PAUSED] Press SPACE to continue...")
                    while True:
                        key2 = cv2.waitKey(0) & 0xFF
                        if key2 == ord(' '):
                            print(">> [RESUMED]")
                            break
                        elif key2 == ord('q'):
                            print("\n>> [USER QUIT] Manual exit")
                            break

        except KeyboardInterrupt:  # ⭐ ĐÂY LÀ PHẦN except ĐÚNG
            print("\n>> [INTERRUPT] Stopped by user")
            
        except Exception as e:  # ⭐ THÊM except cho lỗi khác
            print(f"\n>> [ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.running = False
            if self.use_threading and hasattr(self, 'inference_thread'):
                self.inference_queue.put(None)
                self.inference_thread.join(timeout=1)
            video_stream.stop()
            cv2.destroyAllWindows()

            # ⭐ In thống kê cuối CHI TIẾT HƠN
            print("\n" + "="*60)
            print("SESSION STATISTICS")
            print("="*60)
            print(f"Result: {'✓ SUCCESS' if auth_success else '✗ DENIED'}")
            print(f"Duration: {elapsed_time:.1f}s")
            print(f"Total frames: {total_frames}")
            if total_frames > 0:
                print(f"Frames with face: {frames_with_face} ({frames_with_face/total_frames*100:.1f}%)")
                print(f"Frames without face: {frames_without_face} ({frames_without_face/total_frames*100:.1f}%)")
            else:
                print(f"Frames with face: {frames_with_face} (0%)")
                print(f"Frames without face: {frames_without_face} (0%)")
            print(f"Total inferences: {self.stats['total_inferences']}")
            print(f"REAL detected: {self.stats['real_detected']}")
            print(f"FAKE detected: {self.stats['fake_detected']}")
            print(f"Face Recognition runs: {self.stats['face_recognition_runs']}")
            
            if self.use_alignment:
                total_align = self.stats['alignment_success'] + self.stats['alignment_fail']
                if total_align > 0:
                    success_rate = (self.stats['alignment_success'] / total_align) * 100
                    print(f"Alignment success: {self.stats['alignment_success']}/{total_align} ({success_rate:.1f}%)")
            
            if self.stats['total_inferences'] > 0:
                skip_rate = (self.stats['fake_detected'] / self.stats['total_inferences']) * 100
                print(f"Face Recognition skip rate: {skip_rate:.1f}%")
                print(f"→ Saved {self.stats['fake_detected']} unnecessary Part2 computations!")
            
            # ⭐ THÊM: LÝ DO THẤT BẠI
            if not auth_success:
                if time_since_last_face > MAX_NO_FACE_TIME:
                    print(f"Reason: No face detected for {int(time_since_last_face)}s")
                elif elapsed_time > time_out:
                    print(f"Reason: Timeout after {elapsed_time:.1f}s")
                elif label == "FAKE":
                    print(f"Reason: Fake face detected")
                else:
                    print(f"Reason: Unknown")
                    
            print("="*60)

        return auth_success

# ============================================================
# MAIN - DEMO USAGE
# ============================================================
if __name__ == "__main__":
    # ⭐ Khởi tạo service với Alignment
    auth_service = OptimizedAuthService(use_alignment=True)  # Set False để tắt alignment
    
    # URL Camera ESP32-CAM
    stream_url = "http://192.168.25.158/"
    
    # Test với Admin filter
    print("\n" + "="*60)
    print("TESTING ADMIN AUTHENTICATION WITH ALIGNMENT")
    print("="*60)
    result = auth_service.run_stream(stream_url, isAdmin=True, time_out=20)
    
    if result:
        print("\n✓ AUTHENTICATION SUCCESSFUL - ACCESS GRANTED")
    else:
        print("\n✗ AUTHENTICATION FAILED - ACCESS DENIED")