import customtkinter as ctk
from tkinter import messagebox, simpledialog
import sys
from pathlib import Path
import datetime

# --- SETUP PATH ---
current_file = Path(__file__).resolve()
backend_path = current_file.parent.parent
sys.path.append(str(backend_path))
from ui.login_window import LoginWindow
from core.database_mysql import MySQLManager
from api.opening_door import unlock_door_remote
from core.face_recognition_with_anti_spoofing import RealtimeDeepFakeDetector_Optimized

# db = MySQLManager()
backend_dir = Path(__file__).resolve().parent.parent 
RECOG_PATH = backend_dir / "models" / "faceRecognition_arcface_ckpt(2).pth"
YOLO_PATH = backend_dir / "models" / "yolov8s-face-lindevs.onnx"
DEEPFAKE_PATH = backend_dir / "models" / "deepfake_best5.pth"
# app = RealtimeDeepFakeDetector_Optimized(
#         yolo_path=str(YOLO_PATH),
#         deepfake_model_path=str(DEEPFAKE_PATH),
#         face_recognition_ckpt=str(RECOG_PATH),
#         num_frames=10,
#         frame_skip=0,
#         threshold=0.2,
#         device="cuda",
#         use_threading=True,
#         use_alignment=False
#     )
class Dashboard(ctk.CTk):
    def __init__(self, user, app, db):
        super().__init__()
        self.user = user
        self.app = app
        self.db = db
        
        self.title("SmartKey Dashboard")
        self.geometry("1200x800")
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.build_ui()
    
    def get_all_users(self):
        return self.db.fetch_all_users()

    def get_access_logs(self, limit=20):
        return self.db.fetch_access_logs(limit)
        
    def build_ui(self):
        """Xây dựng toàn bộ giao diện"""
        self.build_header()
        self.build_action_panel()
        self.build_main_content()
        
    def build_header(self):
        """Header với thông tin user và nút đăng xuất"""
        header = ctk.CTkFrame(self, height=70, corner_radius=0)
        header.pack(fill="x", padx=0, pady=0)

        # Thông tin user
        user_frame = ctk.CTkFrame(header, fg_color="transparent")
        user_frame.pack(side="left", padx=20, pady=15)
        
        ctk.CTkLabel(
            user_frame,
            text=f"👤 {self.user['name']}",
            font=("Segoe UI", 18, "bold")
        ).pack(side="left")
        
        role_color = "#E67E22" if self.user["role"] == "ADMIN" else "#3498DB"
        ctk.CTkLabel(
            user_frame,
            text=self.user['role'],
            font=("Segoe UI", 12),
            fg_color=role_color,
            corner_radius=5,
            padx=10,
            pady=5
        ).pack(side="left", padx=10)
        
        # Nút đăng xuất
        ctk.CTkButton(
            header,

            text="🚪 Đăng xuất",
            fg_color="#E74C3C",
            hover_color="#C0392B",
            command=self.on_logout,
            width=130,
            height=35,
            font=("Segoe UI", 13)
        ).pack(side="right", padx=20, pady=15)

    def build_action_panel(self):
        """Panel chứa các nút hành động chính"""
        if self.user["role"] != "ADMIN":
            return
            
        panel = ctk.CTkFrame(self, fg_color="transparent")
        panel.pack(pady=15, padx=20, fill="x")

        btn_frame = ctk.CTkFrame(panel, fg_color="transparent")
        btn_frame.pack()
        
        # Nút mở cửa thủ công
        ctk.CTkButton(
            btn_frame, 
            text="🔓 MỞ CỬA THỦ CÔNG",
            fg_color="#C0392B",
            hover_color="#A93226",
            command=self.on_manual_unlock,
            width=280,
            height=45,
            font=("Segoe UI", 14, "bold"),
            corner_radius=8
        ).pack(side="left", padx=8)

        # Nút đăng ký khuôn mặt
        ctk.CTkButton(
            btn_frame, 
            text="📷 ĐĂNG KÝ KHUÔN MẶT",
            fg_color="#2980B9",
            hover_color="#21618C",
            width=280,
            height=45,
            font=("Segoe UI", 14, "bold"),
            command=self.on_enroll_face,
            corner_radius=8
        ).pack(side="left", padx=8)
        
        # Nút làm mới
        ctk.CTkButton(
            btn_frame, 
            text="🔄 LÀM MỚI",
            fg_color="#27AE60",
            hover_color="#229954",
            width=200,
            height=45,
            font=("Segoe UI", 14, "bold"),
            command=self.refresh,
            corner_radius=8
        ).pack(side="left", padx=8)

    def build_main_content(self):
        """Nội dung chính: bảng user và access logs"""
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Bảng quản lý user (chiếm 60% chiều cao)
        self.build_user_table(main_container)
        
        # Bảng lịch sử truy cập (chiếm 40% chiều cao)
        self.build_access_log_table(main_container)

    def build_user_table(self, parent):
        """Bảng quản lý tài khoản"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, pady=(0, 10))

        # Header
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(15, 10))

        ctk.CTkLabel(
            header, 
            text="👥 QUẢN LÝ TÀI KHOẢN",
            font=("Segoe UI", 17, "bold")
        ).pack(side="left")

        if self.user["role"] == "ADMIN":
            ctk.CTkButton(
                header, 
                text="➕ Thêm User",
                fg_color="#27AE60",
                hover_color="#229954",
                command=self.on_add_user,
                width=140,
                height=35,
                font=("Segoe UI", 13, "bold"),
                corner_radius=6
            ).pack(side="right")

        # Table header
        table_header = ctk.CTkFrame(frame, fg_color="#34495E", height=40)
        table_header.pack(fill="x", padx=15, pady=(0, 5))
        
        headers = [
            ("ID", 60),
            ("Tên", 200),
            ("Username", 150),
            ("Role", 100),
            ("Trạng thái", 110),
            ("Ngày tạo", 160)
        ]
        
        if self.user["role"] == "ADMIN":
            headers.append(("Hành động", 150))
        
        for text, width in headers:
            ctk.CTkLabel(
                table_header, 
                text=text, 
                width=width, 
                font=("Segoe UI", 12, "bold"),
                anchor="w"
            ).pack(side="left", padx=8, pady=8)

        # Scrollable table
        table = ctk.CTkScrollableFrame(frame, fg_color="#2C3E50")
        table.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        users = self.get_all_users()

        for u in users:
            row = ctk.CTkFrame(table, fg_color="#34495E", height=50)
            row.pack(fill="x", pady=3, padx=3)

            # ID
            ctk.CTkLabel(
                row, 
                text=str(u["id"]), 
                width=60,
                anchor="w",
                font=("Segoe UI", 11)
            ).pack(side="left", padx=8)
            
            # Name
            ctk.CTkLabel(
                row, 
                text=u["name"], 
                width=200, 
                anchor="w",
                font=("Segoe UI", 11)
            ).pack(side="left", padx=8)
            
            # Username
            username = u.get("username", "N/A")
            ctk.CTkLabel(
                row, 
                text=username, 
                width=150, 
                anchor="w",
                font=("Segoe UI", 11)
            ).pack(side="left", padx=8)
            
            # Role
            role_color = "#E67E22" if u["role"] == "ADMIN" else "#3498DB"
            ctk.CTkLabel(
                row, 
                text=u["role"], 
                width=100, 
                fg_color=role_color, 
                corner_radius=5,
                font=("Segoe UI", 10, "bold")
            ).pack(side="left", padx=8, pady=6)
            
            # Status
            status_color = "#27AE60" if u["status"] == "ACTIVE" else "#E74C3C"
            ctk.CTkLabel(
                row, 
                text=u["status"], 
                width=110, 
                fg_color=status_color, 
                corner_radius=5,
                font=("Segoe UI", 10, "bold")
            ).pack(side="left", padx=8, pady=6)
            
            # Created at
            created_at = str(u.get("created_at", "N/A"))[:16]  # Cắt bớt timestamp
            ctk.CTkLabel(
                row, 
                text=created_at, 
                width=160, 
                anchor="w",
                font=("Segoe UI", 10)
            ).pack(side="left", padx=8)

            # Actions (chỉ admin)
            if self.user["role"] == "ADMIN":
                action_frame = ctk.CTkFrame(row, fg_color="transparent")
                action_frame.pack(side="left", padx=8)
                
                ctk.CTkButton(
                    action_frame, 
                    text="✏️",
                    width=45,
                    height=32,
                    fg_color="#3498DB",
                    hover_color="#2E86C1",
                    command=lambda x=u: self.on_edit_user(x),
                    font=("Segoe UI", 14),
                    corner_radius=5
                ).pack(side="left", padx=3)

                if u["id"] != self.user["id"]:
                    ctk.CTkButton(
                        action_frame, 
                        text="🗑",
                        fg_color="#E74C3C",
                        hover_color="#CB4335",
                        width=45,
                        height=32,
                        command=lambda x=u: self.on_delete_user(x),
                        font=("Segoe UI", 14),
                        corner_radius=5
                    ).pack(side="left", padx=3)

    def build_access_log_table(self, parent):
        """Bảng lịch sử truy cập"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True)

        # Header
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(15, 10))
        
        ctk.CTkLabel(
            header, 
            text="📜 LỊCH SỬ TRUY CẬP (20 GẦN NHẤT)",
            font=("Segoe UI", 17, "bold")
        ).pack(side="left")

        # Table header
        log_header = ctk.CTkFrame(frame, fg_color="#34495E", height=40)
        log_header.pack(fill="x", padx=15, pady=(0, 5))
        
        headers = [
            ("ID", 60),
            ("Tên người dùng", 220),
            ("Phương thức", 130),
            ("Độ tin cậy", 120),
            ("Thời gian", 200)
        ]
        
        for text, width in headers:
            ctk.CTkLabel(
                log_header, 
                text=text, 
                width=width, 
                font=("Segoe UI", 12, "bold"),
                anchor="w"
            ).pack(side="left", padx=8, pady=8)

        # Scrollable table
        table = ctk.CTkScrollableFrame(frame, fg_color="#2C3E50")
        table.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        logs = self.get_access_logs(limit=20)

        for log in logs:
            row = ctk.CTkFrame(table, fg_color="#34495E", height=45)
            row.pack(fill="x", pady=3, padx=3)

            # ID
            ctk.CTkLabel(
                row, 
                text=str(log["id"]), 
                width=60,
                anchor="w",
                font=("Segoe UI", 11)
            ).pack(side="left", padx=8)

            # Name
            name = log["name"] if log["name"] else "Unknown"
            ctk.CTkLabel(
                row, 
                text=name, 
                width=220, 
                anchor="w",
                font=("Segoe UI", 11)
            ).pack(side="left", padx=8)
            
            # Method
            method_color = "#9B59B6" if log["method"] == "FACE" else "#F39C12"
            ctk.CTkLabel(
                row, 
                text=log["method"], 
                width=130, 
                fg_color=method_color, 
                corner_radius=5,
                font=("Segoe UI", 10, "bold")
            ).pack(side="left", padx=8, pady=6)
            
            # Confidence score
            score = log.get("confidence_score", 0)
            if score:
                score_text = f"{score:.4f}"
                if score >= 0.7:
                    score_color = "#27AE60"
                elif score >= 0.5:
                    score_color = "#E67E22"
                else:
                    score_color = "#E74C3C"
            else:
                score_text = "N/A"
                score_color = "#95A5A6"
                
            ctk.CTkLabel(
                row, 
                text=score_text, 
                width=120, 
                fg_color=score_color, 
                corner_radius=5,
                font=("Segoe UI", 10, "bold")
            ).pack(side="left", padx=8, pady=6)
            
            # Created at
            time_str = str(log["created_at"]) if log.get("created_at") else "N/A"
            ctk.CTkLabel(
                row, 
                text=time_str, 
                width=200, 
                anchor="w",
                font=("Segoe UI", 10)
            ).pack(side="left", padx=8)

    # ================= DATABASE FUNCTIONS =================
    def get_all_users(self):
        """Lấy danh sách tất cả user từ database"""
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            sql = """
                SELECT id, name, username, role, status, created_at 
                FROM users 
                ORDER BY id DESC
            """
            cursor.execute(sql)
            return cursor.fetchall()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lấy danh sách users: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def get_access_logs(self, limit=20):
        """Lấy lịch sử truy cập"""
        conn = self.db.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            sql = """
                SELECT 
                    al.id, 
                    u.name, 
                    al.method, 
                    al.confidence_score, 
                    al.created_at 
                FROM access_logs al
                LEFT JOIN users u ON al.user_id = u.id
                ORDER BY al.created_at DESC
                LIMIT %s
            """
            cursor.execute(sql, (limit,))
            return cursor.fetchall()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lấy access logs: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()

    # ================= ACTION HANDLERS =================
    def on_manual_unlock(self):
        result = unlock_door_remote()
        messagebox.showinfo(f"Thông báo! {result}")

    def on_add_user(self):
        """Dialog thêm user mới với đầy đủ trường"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Thêm User Mới")
        dialog.geometry("500x550")
        dialog.resizable(False, False)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (dialog.winfo_screenheight() // 2) - (550 // 2)
        dialog.geometry(f"+{x}+{y}")
        
        ctk.CTkLabel(
            dialog,
            text="➕ Thêm User Mới",
            font=("Segoe UI", 18, "bold")
        ).pack(pady=20)
        
        # Form frame
        form = ctk.CTkFrame(dialog, fg_color="transparent")
        form.pack(padx=40, fill="both", expand=True)
        
        # Name
        ctk.CTkLabel(form, text="Tên hiển thị:", font=("Segoe UI", 12)).pack(anchor="w", pady=(10, 5))
        name_var = ctk.StringVar()
        ctk.CTkEntry(
            form, 
            textvariable=name_var,
            height=40,
            font=("Segoe UI", 12),
            placeholder_text="Ví dụ: Nguyễn Văn A"
        ).pack(fill="x", pady=(0, 10))
        
        # Username
        ctk.CTkLabel(form, text="Username (chỉ dành cho ADMIN):", font=("Segoe UI", 12)).pack(anchor="w", pady=(10, 5))
        username_var = ctk.StringVar()
        username_entry = ctk.CTkEntry(
            form, 
            textvariable=username_var,
            height=40,
            font=("Segoe UI", 12),
            placeholder_text="Ví dụ: nguyenvana"
        )
        username_entry.pack(fill="x", pady=(0, 10))
        
        # Password
        ctk.CTkLabel(form, text="Password (chỉ dành cho ADMIN):", font=("Segoe UI", 12)).pack(anchor="w", pady=(10, 5))
        password_var = ctk.StringVar()
        password_entry = ctk.CTkEntry(
            form, 
            textvariable=password_var,
            height=40,
            font=("Segoe UI", 12),
            show="*",
            placeholder_text="Nhập mật khẩu"
        )
        password_entry.pack(fill="x", pady=(0, 10))
        
        # Role
        ctk.CTkLabel(form, text="Role:", font=("Segoe UI", 12)).pack(anchor="w", pady=(10, 5))
        role_var = ctk.StringVar(value="MEMBER")
        
        def on_role_change(choice):
            if choice == "MEMBER":
                username_entry.configure(state="disabled")
                password_entry.configure(state="disabled")
                username_var.set("")
                password_var.set("")
            else:
                username_entry.configure(state="normal")
                password_entry.configure(state="normal")
        
        role_menu = ctk.CTkOptionMenu(
            form,
            variable=role_var,
            values=["ADMIN", "MEMBER"],
            height=40,
            font=("Segoe UI", 12),
            command=on_role_change
        )
        role_menu.pack(fill="x", pady=(0, 10))
        
        # Khóa ban đầu vì default là MEMBER
        username_entry.configure(state="disabled")
        password_entry.configure(state="disabled")
        
        def save_user():
            name = name_var.get().strip()
            username = username_var.get().strip()
            password = password_var.get().strip()
            role = role_var.get()
            
            if not all([name, username, password]) and role_var=="ADMIN":
                messagebox.showwarning("Cảnh báo", "Vui lòng điền đầy đủ thông tin!")
                return
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
            try:
                if role_var == "ADMIN":
                    sql = """
                        INSERT INTO users (name, username, password, role, status, created_at) 
                        VALUES (%s, %s, %s, %s, 'ACTIVE', %s)
                    """
                    cursor.execute(sql, (name, username, password, role, datetime.datetime.now()))
                    conn.commit()
                else:
                    sql = """
                        INSERT INTO users (name, role, status, created_at) 
                        VALUES (%s, %s, 'ACTIVE', %s)
                    """
                    cursor.execute(sql, (name, role, datetime.datetime.now()))
                    conn.commit()

                dialog.destroy()
                messagebox.showinfo("Thành công", f"Đã thêm user: {name}")
                self.refresh()
            except Exception as e:
                conn.rollback()
                messagebox.showerror("Lỗi", f"Không thể thêm user: {str(e)}")
            finally:
                cursor.close()
                conn.close()
        
        # Buttons
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(
            btn_frame,
            text="💾 Lưu",
            command=save_user,
            width=150,
            height=40,
            fg_color="#27AE60",
            hover_color="#229954",
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="✗ Hủy",
            command=dialog.destroy,
            width=150,
            height=40,
            fg_color="#E74C3C",
            hover_color="#C0392B",
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)
    
    def on_enroll_face(self):
        """Giao diện chọn User từ Combobox để đăng ký khuôn mặt"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Đăng ký khuôn mặt")
        dialog.geometry("450x300")
        dialog.grab_set()
        
        ctk.CTkLabel(dialog, text="📷 Đăng ký khuôn mặt", font=("Segoe UI", 18, "bold")).pack(pady=15)
        
        # Lấy danh sách từ DB
        users = self.db.get_user_list_for_enrollment()
        user_options = [f"ID: {u['id']} - {u['name']}" for u in users]
        
        if not user_options:
            ctk.CTkLabel(dialog, text="Không có user khả dụng").pack()
            return

        combo_var = ctk.StringVar(value=user_options[0])
        ctk.CTkLabel(dialog, text="Chọn người dùng:").pack(pady=5)
        combobox = ctk.CTkComboBox(dialog, values=user_options, variable=combo_var, width=300, height=40)
        combobox.pack(pady=10)

        def confirm():
            user_id = int(combo_var.get().split(" - ")[0].replace("ID: ", ""))
            dialog.destroy()
            self.app.run_camera_registration(0, user_id)
            messagebox.showinfo("Thông báo", "Đăng ký thành công!")

        ctk.CTkButton(dialog, text="Bắt đầu", command=confirm, fg_color="#2980B9").pack(pady=20)

    def on_edit_user(self, user):
        """Dialog chỉnh sửa user với logic khóa/mở field theo Role"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Chỉnh sửa User")
        dialog.geometry("500x600")
        dialog.grab_set()

        ctk.CTkLabel(dialog, text=f"✏️ Chỉnh sửa User #{user['id']}", font=("Segoe UI", 18, "bold")).pack(pady=20)
        form = ctk.CTkFrame(dialog, fg_color="transparent")
        form.pack(padx=40, fill="both", expand=True)

        # Name
        ctk.CTkLabel(form, text="Tên hiển thị:").pack(anchor="w")
        name_var = ctk.StringVar(value=user["name"])
        ctk.CTkEntry(form, textvariable=name_var, height=40).pack(fill="x", pady=(0, 10))

        # Username field
        ctk.CTkLabel(form, text="Username:").pack(anchor="w")
        username_var = ctk.StringVar(value=user.get("username", "") if user.get("username") else "")
        username_entry = ctk.CTkEntry(form, textvariable=username_var, height=40)
        username_entry.pack(fill="x", pady=(0, 10))

        # Password field
        ctk.CTkLabel(form, text="Password mới (để trống nếu không đổi):").pack(anchor="w")
        password_var = ctk.StringVar()
        password_entry = ctk.CTkEntry(form, textvariable=password_var, height=40, show="*")
        password_entry.pack(fill="x", pady=(0, 10))

        # Role & Status
        ctk.CTkLabel(form, text="Role:").pack(anchor="w")
        role_var = ctk.StringVar(value=user["role"])
        
        # Hàm xử lý khóa/mở khi thay đổi Role
        def toggle_fields(choice):
            if choice == "MEMBER":
                username_entry.configure(state="disabled", fg_color="#34495E")
                password_entry.configure(state="disabled", fg_color="#34495E")
                username_var.set("")
                password_var.set("")
                # ⭐ MEMBER: Cho phép chỉnh sửa status
                status_menu.configure(state="normal", fg_color="#1D1E1E")
            else:
                username_entry.configure(state="normal", fg_color="#1D1E1E")
                password_entry.configure(state="normal", fg_color="#1D1E1E")
                # ⭐ ADMIN: KHÔNG cho phép chỉnh sửa status
                status_menu.configure(state="disabled", fg_color="#34495E")

        role_menu = ctk.CTkOptionMenu(
            form, 
            variable=role_var, 
            values=["ADMIN", "MEMBER"], 
            command=toggle_fields, 
            height=40
        )
        role_menu.pack(fill="x", pady=(0, 10))

        # ⭐ STATUS FIELD VỚI KIỂM SOÁT
        ctk.CTkLabel(form, text="Trạng thái:").pack(anchor="w")
        status_var = ctk.StringVar(value=user["status"])
        
        # Tạo status_menu
        status_menu = ctk.CTkOptionMenu(
            form, 
            variable=status_var, 
            values=["ACTIVE", "INACTIVE"], 
            height=40
        )
        status_menu.pack(fill="x", pady=(0, 10))
        
        # ⭐ KIỂM TRA NGAY: Nếu user đang chỉnh sửa là ADMIN -> disable status
        if user["role"] == "ADMIN":
            status_menu.configure(state="disabled", fg_color="#34495E")
        
        # ⭐ THÊM LABEL HIỂN THỊ THÔNG BÁO
        if user["role"] == "ADMIN":
            ctk.CTkLabel(
                form, 
                text="⚠️ ADMIN luôn có trạng thái ACTIVE", 
                font=("Segoe UI", 11, "italic"),
                text_color="#E67E22"
            ).pack(pady=(0, 10))
        
        # Chạy kiểm tra lần đầu để set trạng thái các ô nhập liệu
        toggle_fields(user["role"])

        def save_changes():
            # Kiểm tra tên
            if not name_var.get().strip():
                messagebox.showwarning("Cảnh báo", "Tên không được để trống!")
                return
            
            # Xử lý username theo role
            if role_var.get() == "ADMIN":
                final_username = username_var.get().strip()
                if not final_username:
                    messagebox.showwarning("Cảnh báo", "Admin phải có username!")
                    return
                # ⭐ ADMIN luôn có status = "ACTIVE" (không thể đổi)
                final_status = "ACTIVE"
            else:
                final_username = None
                final_status = status_var.get()  # Lấy từ field
            
            # Xử lý password
            new_password = password_var.get().strip()
            if not new_password:
                new_password = None
            
            success = self.db.update_user(
                user["id"], 
                name_var.get().strip(), 
                final_username,
                new_password,
                role_var.get(), 
                final_status  # Dùng status đã xử lý
            )
            
            if success:
                dialog.destroy()
                messagebox.showinfo("Thành công", "Đã cập nhật!")
                self.refresh()
            else:
                messagebox.showerror("Lỗi", "Không thể cập nhật user.")
                
        # Buttons
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(
            btn_frame,
            text="💾 Lưu thay đổi",
            command=save_changes,
            width=160,
            height=40,
            fg_color="#3498DB",
            hover_color="#2E86C1",
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="✗ Hủy",
            command=dialog.destroy,
            width=160,
            height=40,
            fg_color="#E74C3C",
            hover_color="#C0392B",
            font=("Segoe UI", 13, "bold")
        ).pack(side="left", padx=5)
        
    def on_delete_user(self, user):
        """Xóa user"""
        if user["id"] == self.user["id"]:
            messagebox.showerror("Lỗi", "Không thể xóa chính mình!")
            return

        result = messagebox.askyesno(
            "⚠️ Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa user:\n\n"
            f"ID: {user['id']}\n"
            f"Tên: {user['name']}\n"
            f"Username: {user.get('username', 'N/A')}\n\n"
            f"⚠️ Hành động này không thể hoàn tác!",
            icon='warning'
        )
        
        if result:
            success = self.db.delete_user(user["id"])
            if success:
                messagebox.showinfo("Thành công", f"Đã xóa user: {user['name']}")
                self.refresh()
            else:
                messagebox.showerror("Lỗi", "Không thể xóa user")

    def refresh(self):
        """Làm mới giao diện"""
        # Clear all widgets
        for widget in self.winfo_children():
            widget.destroy()
        
        # Rebuild UI
        self.build_ui()

    def on_logout(self):
        """Đăng xuất và quay về màn hình Login"""
        print(f"[LOGOUT] Bắt đầu đăng xuất user: {self.user['name']}")
        
        # Đóng cửa sổ Dashboard hiện tại
        self.destroy()
        
        # Tạo và chạy cửa sổ Login mới
        try:
            login_window = LoginWindow(self.app, self.db)
            login_window.mainloop()
        except Exception as e:
            print(f"[LOGOUT ERROR] Không thể mở LoginWindow: {e}")
            # Fallback: thoát chương trình
            sys.exit(0)
    
    def on_login_callback(self, user):
        """Callback sau khi đăng nhập thành công"""
        Dashboard(user).mainloop()

