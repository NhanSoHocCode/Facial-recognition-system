CREATE DATABASE IF NOT EXISTS smartKey_app
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE smartKey_app;

-- Bảng users (Đã thêm username và password)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    username VARCHAR(50) NOT NULL UNIQUE, -- Tên đăng nhập duy nhất
    password VARCHAR(255) NOT NULL,       -- Lưu hash password (ví dụ: bcrypt)
    role ENUM('ADMIN', 'MEMBER') DEFAULT 'MEMBER',
    status ENUM('ACTIVE', 'INACTIVE') DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Bảng user_credentials: lưu embedding khuôn mặt / template vân tay
CREATE TABLE IF NOT EXISTS user_credentials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    type ENUM('FACE', 'FINGERPRINT') NOT NULL,
    data LONGBLOB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_credentials_user_type (user_id, type)
) ENGINE=InnoDB;

-- Bảng log ra/vào
CREATE TABLE IF NOT EXISTS access_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NULL,
    credential_id INT NULL,
    method ENUM('FACE','FINGERPRINT') NOT NULL,
    snapshot_path VARCHAR(255),
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (credential_id) REFERENCES user_credentials(id) ON DELETE SET NULL,
    INDEX idx_access_logs_time (created_at)
) ENGINE=InnoDB;

-- Trigger cập nhật username và password khi role thay đổi
CREATE TRIGGER before_user_update_role
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    IF NEW.role = 'MEMBER' THEN
        SET NEW.username = NULL;
        SET NEW.password = NULL;
    END IF;
END;

-- Trigger xóa user_credentials khi user bị xóa (cascade)
CREATE TRIGGER before_user_delete_cleanup
BEFORE DELETE ON users
FOR EACH ROW
BEGIN
    DELETE FROM access_logs WHERE user_id = OLD.id;
END;

-- 3. Trigger xử lý khi xóa credential (xóa log liên quan)
CREATE TRIGGER before_credential_delete_cleanup
BEFORE DELETE ON user_credentials
FOR EACH ROW
BEGIN
    DELETE FROM access_logs WHERE credential_id = OLD.id;
END;
