import mysql.connector
import numpy as np
import torch
import torch.nn.functional as F
import datetime
class MySQLManager:
    def __init__(self):
        self.config = {
            'user': 'root',
            'password': 'root',
            'host': 'localhost',
            'port': 3306,
            'database': 'smartKey_app',
            'raise_on_warnings': True,
            'charset': 'utf8mb4',
        }

    def get_connection(self):
        return mysql.connector.connect(**self.config)
    
    def compute_cosine_similarity(self, embed1, embed2) -> float:
        """
        Tính cosine similarity giữa 2 vector embedding.
        """
        if isinstance(embed1, np.ndarray): embed1 = torch.from_numpy(embed1)
        if isinstance(embed2, np.ndarray): embed2 = torch.from_numpy(embed2)

        embed1 = embed1.float()
        embed2 = embed2.float()

        if embed1.dim() == 1: embed1 = embed1.unsqueeze(0)
        if embed2.dim() == 1: embed2 = embed2.unsqueeze(0)

        similarity = F.cosine_similarity(embed1, embed2, dim=1)
        return similarity.item()
    
    # Write query
    def fetch_all_users(self):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT id, name, username, role, status, created_at FROM users ORDER BY id DESC")
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()

    def fetch_access_logs(self, limit=20):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            sql = """
                SELECT al.id, u.name, al.method, al.confidence_score, al.created_at 
                FROM access_logs al
                LEFT JOIN users u ON al.user_id = u.id
                ORDER BY al.created_at DESC LIMIT %s
            """
            cursor.execute(sql, (limit,))
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()

    def get_user_list_for_enrollment(self):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT id, name FROM users WHERE status = 'ACTIVE'")
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
    def delete_user(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            conn.commit()
            return True
        finally:
            cursor.close()
            conn.close()

    def add_user(self, name, username, password, role):
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if role == "ADMIN":
                sql = "INSERT INTO users (name, username, password, role, status, created_at) VALUES (%s, %s, %s, %s, 'ACTIVE', NOW())"
                cursor.execute(sql, (name, username, password, role))
            else:
                sql = "INSERT INTO users (name, role, status, created_at) VALUES (%s, %s, 'ACTIVE', NOW())"
                cursor.execute(sql, (name, role))
            conn.commit()
            return True
        finally:
            cursor.close()
            conn.close()

    def update_user(self, user_id, name, username, password, role, status):
        """
        Cập nhật thông tin user.
        - ADMIN: phải có username, password có thể None (không đổi)
        - MEMBER: username có thể None, password luôn None
        """
        print(f"[DB UPDATE] user_id={user_id}, name='{name}', username={username}, "
            f"pwd={'[CÓ]' if password else '[None]'}, role='{role}', status='{status}'")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if role == "ADMIN":
                # Kiểm tra ADMIN phải có username
                if not username:
                    print("!!! Lỗi: ADMIN phải có username")
                    return False
                    
                if password:  # Có password mới
                    sql = "UPDATE users SET name=%s, username=%s, password=%s, role=%s, status=%s WHERE id=%s"
                    cursor.execute(sql, (name, username, password, role, status, user_id))
                else:  # Không đổi password
                    sql = "UPDATE users SET name=%s, username=%s, role=%s, status=%s WHERE id=%s"
                    cursor.execute(sql, (name, username, role, status, user_id))
            else:  # MEMBER
                # MEMBER: username có thể là None (NULL trong database)
                # Không cập nhật password cho member
                sql = "UPDATE users SET name=%s, username=%s, role=%s, status=%s WHERE id=%s"
                cursor.execute(sql, (name, username, role, status, user_id))
                
            conn.commit()
            affected_rows = cursor.rowcount
            print(f"[DB UPDATE] Thành công: {affected_rows} dòng bị ảnh hưởng")
            return affected_rows > 0
            
        except mysql.connector.Error as err:
            conn.rollback()
            print(f"!!! Lỗi SQL khi update user: {err}")
            print(f"!!! SQL State: {err.sqlstate}, Error Code: {err.errno}")
            # In thêm thông tin để debug
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            conn.rollback()
            print(f"!!! Lỗi hệ thống: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            cursor.close()
            conn.close()
        
    def get_access_logs(self, limit=20):
        conn = self.get_connection()
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
            logs = cursor.fetchall()
            
            print(f"{'ID':<5} | {'Name':<20} | {'Method':<12} | {'Score':<8} | {'Time'}")
            print("-" * 70)
            for log in logs:
                name = log['name'] if log['name'] else "Unknown"
                print(f"{log['id']:<5} | {name:<20} | {log['method']:<12} | {log['confidence_score']:<8.4f} | {log['created_at']}")
            
            return logs
        except mysql.connector.Error as err:
            print(f"!!! Lỗi: {err}")
            return []
        finally:
            cursor.close()
            conn.close()

    def add_embedding_recognition(self, user_id, face_embedding, isFace = True):
        if hasattr(face_embedding, 'detach'):
            face_embedding = face_embedding.detach().cpu().numpy()
    
        face_embedding = face_embedding.astype(np.float32)
        embedding_bytes = face_embedding.tobytes()

        conn = self.get_connection()
        cursor = conn.cursor(prepared=True)
        
        if isFace:
            type = "FACE"
        else:
            type = "FINGERPRINT"

        try:
            sql_user = b"INSERT INTO user_credentials (user_id, type, data, created_at) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql_user, (user_id, type, embedding_bytes, datetime.datetime.now()))
            conn.commit()
            print(f"=> Đã thêm thành công: (User_Id: {user_id}) - Type: {type}")

        except mysql.connector.Error as err:
            conn.rollback()
            print(f"!!! Lỗi SQL: {err.errno} / {err.sqlstate} / {err.msg}")
        except Exception as e:
            conn.rollback()
            print(f"!!! Lỗi hệ thống: {e}")
        finally:
            cursor.close()
            conn.close()

    def get_all_users(self):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            sql = "SELECT id, name, role, status FROM users ORDER BY id DESC"
            cursor.execute(sql)
            return cursor.fetchall()
        except Exception as e:
            print(f"Lỗi: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def verify_admin_login(self, username, password):
        """
        Xác minh đăng nhập dành riêng cho ADMIN hoặc OWNER.
        Trả về thông tin user nếu thành công, None nếu thất bại.
        """
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            sql = """
                SELECT id, name, username, password, role, status 
                FROM users 
                WHERE username = %s AND status = 'ACTIVE' 
                AND role IN ('ADMIN')
            """
            cursor.execute(sql, (username,))
            user = cursor.fetchone()

            if user:
                if user['password'] == password:
                    print(f"==> Đăng nhập ADMIN thành công: {user['name']}")
                    return user
            print("!!! Đăng nhập thất bại: Sai tài khoản, mật khẩu hoặc không có quyền Admin.")
            return None

        except mysql.connector.Error as err:
            print(f"!!! Lỗi SQL: {err}")
            return None
        finally:
            cursor.close()
            conn.close()


    def add_access_logs(self, user_id, credentital_id, method, confidence_score):
        conn = self.get_connection()
        cursor = conn.cursor(prepared=True)

        try:
            sql_user = b"INSERT INTO access_logs (user_id, credential_id, method, confidence_score, created_at) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(sql_user, (user_id, credentital_id, method, confidence_score, datetime.datetime.now()))
            id = cursor.lastrowid
            if not id:
                raise Exception("Không lấy được Access-logs ID sau khi insert log.")
            conn.commit()
            print(f"=> Đã thêm access logs thành công")
            return id

        except mysql.connector.Error as err:
            conn.rollback()
            print(f"!!! Lỗi SQL: {err.errno} / {err.sqlstate} / {err.msg}")
            return None
        except Exception as e:
            conn.rollback()
            print(f"!!! Lỗi hệ thống: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
    
    def find_user_by_face(self, face_embedding, threshold=0.4, isFace=True, isAdmin=False):
        if hasattr(face_embedding, 'detach'):
            face_embedding = face_embedding.detach().cpu().numpy()
    
        face_embedding = face_embedding.astype(np.float32)

        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            if isAdmin:
                print("!!! YEU CAU QUYEN ADMIN !")
                if(isFace):
                    auth_type = "FACE"
                else:
                    auth_type = "FINGERPRINT"
                sql = """
                    SELECT 
                        uc.id AS credential_id,
                        uc.data AS embedding_blob,
                        u.id AS user_id,
                        u.name AS name,
                        u.role AS role,
                        u.status AS status
                    FROM user_credentials uc
                    JOIN users u ON uc.user_id = u.id
                    WHERE uc.type = %s
                    AND u.role = 'ADMIN'
                    AND uc.is_active = TRUE
                    AND u.status = 'ACTIVE'
                """
                cursor.execute(sql, (auth_type,))
                rows = cursor.fetchall()
                if not rows:
                    print("!!! Chưa có admin nào trong DB.")
                    return None
            else:
                print("=== TÌM KIẾM USER THƯỜNG ===")
                if(isFace):
                    auth_type = "FACE"
                else:
                    auth_type = "FINGERPRINT"
                sql = """
                    SELECT 
                        uc.id AS credential_id,
                        uc.data AS embedding_blob,
                        u.id AS user_id,
                        u.name AS name,
                        u.role AS role,
                        u.status AS status
                    FROM user_credentials uc
                    JOIN users u ON uc.user_id = u.id
                    WHERE uc.type = %s
                    AND uc.is_active = TRUE
                    AND u.status = 'ACTIVE'
                """
                cursor.execute(sql, (auth_type,))
                rows = cursor.fetchall()
                if not rows:
                    print("!!! Chưa có user nào trong DB.")
                    return None

            best_match = None
            best_score = -1.0

            for row in rows:
                blob = row["embedding_blob"]  # dạng bytes (LONGBLOB)

                emb_db = np.frombuffer(blob, dtype=np.float32)

                if emb_db.shape[0] != face_embedding.shape[0]:
                    print(f"!!! Bỏ qua credential {row['credential_id']} do kích thước không khớp: {emb_db.shape}")
                    continue

                score = self.compute_cosine_similarity(face_embedding, emb_db)

                if score > best_score:
                    best_score = score
                    best_match = row

            print(f"Best similarity: {best_score:.4f}")

            # 4. So sánh với threshold để quyết định mở cửa
            if best_match is not None and best_score >= threshold:
                result = {
                    "user_id": best_match["user_id"],
                    "name": best_match["name"],
                    "role": best_match["role"],
                    "credential_id": best_match["credential_id"],
                    "similarity": best_score
                }
                print(f"=> Nhận diện thành công: {result['name']} (ID: {result['user_id']}), score = {best_score:.4f}")
                self.add_access_logs(result['user_id'], result['credential_id'], "FACE", best_score)
                return result
            else:
                print("!!! Không tìm thấy user phù hợp.")
                return None

        except mysql.connector.Error as err:
            print(f"!!! Lỗi SQL: {err}")
            return None
        finally:
            cursor.close()
            conn.close()

