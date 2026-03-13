import customtkinter as ctk
from tkinter import messagebox
import sys
from pathlib import Path

# --- SETUP PATH ---
current_file = Path(__file__).resolve()
backend_path = current_file.parent.parent
sys.path.append(str(backend_path))
RECOG_PATH = backend_path / "models" / "faceRecognition_arcface_ckpt(2).pth"
YOLO_PATH = backend_path / "models" / "yolov8s-face-lindevs.onnx"
DEEPFAKE_PATH = backend_path / "models" / "deepfake_best5.pth"
from core.face_recognition_with_anti_spoofing import RealtimeDeepFakeDetector_Optimized
from core.database_mysql import MySQLManager

app = RealtimeDeepFakeDetector_Optimized(
            yolo_path=str(YOLO_PATH),
            deepfake_model_path=str(DEEPFAKE_PATH),
            face_recognition_ckpt=str(RECOG_PATH),
            num_frames=10,
            frame_skip=0,
            threshold=0.2,
            device="cuda",
            use_threading=True,
            use_alignment=False
        )
db = MySQLManager()

class LoginWindow(ctk.CTk):
    def __init__(self, app, db):
        super().__init__()

        self.title("Smart Door Lock - Login")
        self.geometry("580x620")
        ctk.set_appearance_mode("dark")

        # Header với thông tin trường
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(pady=10)
        
        ctk.CTkLabel(
            header_frame, 
            text="TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP. HỒ CHÍ MINH",
            font=("Segoe UI", 12, "bold"),
            text_color="#3498DB"
        ).pack()
        
        ctk.CTkLabel(
            header_frame, 
            text="KHOA CÔNG NGHỆ THÔNG TIN - MÔN HỌC XỬ LÝ ẢNH SỐ",
            font=("Segoe UI", 10),
            text_color="#95A5A6"
        ).pack()

        # Tiêu đề chính
        ctk.CTkLabel(
            self, 
            text="🔐 SMART DOOR LOCK", 
            font=("Segoe UI", 22, "bold"),
            text_color="#ECF0F1"
        ).pack(pady=10)

        # Đề tài
        project_frame = ctk.CTkFrame(self, fg_color="#2C3E50", corner_radius=10)
        project_frame.pack(pady=8, padx=20, fill="x")
        
        ctk.CTkLabel(
            project_frame,
            text="ĐỒ ÁN CUỐI KỲ",
            font=("Segoe UI", 11, "bold"),
            text_color="#E67E22"
        ).pack(pady=(8, 3))
        
        ctk.CTkLabel(
            project_frame,
            text="HỆ THỐNG KHÓA CỬA THÔNG MINH TÍCH HỢP NHẬN DIỆN VÂN TAY VÀ KHUÔN MẶT",
            font=("Segoe UI", 11, "bold"),
            text_color="#ECF0F1",
            wraplength=520
        ).pack(pady=(0, 8))

        # Thông tin nhóm - Chia 2 cột
        group_frame = ctk.CTkFrame(self, fg_color="#34495E", corner_radius=10)
        group_frame.pack(pady=8, padx=20, fill="x")
        
        # Hàng 1: GVHD và Mã lớp
        info_top = ctk.CTkFrame(group_frame, fg_color="transparent")
        info_top.pack(fill="x", padx=10, pady=(8, 4))
        
        ctk.CTkLabel(
            info_top,
            text="GVHD: PGS.TS. Hoàng Văn Dũng  |  MÃ LỚP: 25IDIPR430685_04",
            font=("Segoe UI", 10),
            text_color="#BDC3C7"
        ).pack()
        
        # Hàng 2: Sinh viên - 2 cột
        students_frame = ctk.CTkFrame(group_frame, fg_color="transparent")
        students_frame.pack(fill="x", padx=10, pady=(0, 8))
        
        left_col = ctk.CTkFrame(students_frame, fg_color="transparent")
        left_col.pack(side="left", fill="both", expand=True)
        
        right_col = ctk.CTkFrame(students_frame, fg_color="transparent")
        right_col.pack(side="right", fill="both", expand=True)
        
        ctk.CTkLabel(
            left_col,
            text="SVTH: Trần Hồng Quang Lê 23110251",
            font=("Segoe UI", 10),
            text_color="#1ABC9C",
            anchor="w"
        ).pack(fill="x")
        
        ctk.CTkLabel(
            left_col,
            text="           Phạm Thị Tuyết Minh 23110268",
            font=("Segoe UI", 10),
            text_color="#1ABC9C",
            anchor="w"
        ).pack(fill="x")
        
        ctk.CTkLabel(
            right_col,
            text="Cáp Thanh Nhàn 23110276",
            font=("Segoe UI", 10),
            text_color="#1ABC9C",
            anchor="w"
        ).pack(fill="x")
        
        ctk.CTkLabel(
            right_col,
            text="Đặng Ngọc Nhân 23110279",
            font=("Segoe UI", 10),
            text_color="#1ABC9C",
            anchor="w"
        ).pack(fill="x")

        # Form đăng nhập
        login_frame = ctk.CTkFrame(self, fg_color="transparent")
        login_frame.pack(pady=10, padx=20, fill="x")

        self.username = ctk.CTkEntry(
            login_frame, 
            placeholder_text="👤 Username",
            height=38,
            font=("Segoe UI", 12)
        )
        self.username.pack(pady=6, fill="x")
        self.user = None

        self.password = ctk.CTkEntry(
            login_frame, 
            placeholder_text="🔒 Password", 
            show="*",
            height=38,
            font=("Segoe UI", 12)
        )
        self.password.pack(pady=6, fill="x")

        self.app = app
        self.db = db

        ctk.CTkButton(
            login_frame, 
            text="🔑 Login by Account",
            command=self.login_account,
            height=38,
            font=("Segoe UI", 13, "bold"),
            fg_color="#3498DB",
            hover_color="#2980B9"
        ).pack(pady=8, fill="x")
        
        ctk.CTkButton(
            login_frame, 
            text="👁️ Login by Face",
            fg_color="#1ABC9C",
            hover_color="#16A085",
            command=self.login_face,
            height=38,
            font=("Segoe UI", 13, "bold")
        ).pack(pady=4, fill="x")

    def login_account(self):
        self.user = self.db.verify_admin_login(self.username.get(), self.password.get())
        if self.user != None:
            self.open_dashboard()
        else:
            messagebox.showerror("Login failed", "Sai tài khoản hoặc mật khẩu")

    def login_face(self):
        user_data = self.app.verify_admin(0)
        if user_data:
            # Map 'user_id' -> 'id' để Dashboard hoạt động đúng
            user_data['id'] = user_data['user_id']
            self.user = user_data
            self.open_dashboard()
        else:
            messagebox.showerror("Login failed", "Khuôn mặt không khớp hoặc không phải ADMIN")
    
    def open_dashboard(self):
        from ui.main_dashboard import Dashboard
        self.destroy()
        dashboard = Dashboard(self.user, self.app, self.db)
        dashboard.mainloop()


if __name__ == "__main__":
    app = LoginWindow(app, db)
    app.mainloop()