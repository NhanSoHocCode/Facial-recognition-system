import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from typing import Optional, Union
import numpy as np
import cv2


# ==========================================
# PHẦN 1: CẤU TRÚC MẠNG AI
# ==========================================
class FingerprintEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        # Sửa lớp đầu tiên cho ảnh 1 kênh (Grayscale)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, old_conv.out_channels,
                                        kernel_size=old_conv.kernel_size,
                                        stride=old_conv.stride,
                                        padding=old_conv.padding, bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.embedding(feat)
        return F.normalize(emb, p=2, dim=1)


# ==========================================
# PHẦN 2: CLASS XỬ LÝ CHÍNH
# ==========================================
class FingerprintRecognizer:
    def __init__(self, ckpt_path: str, device: Optional[str] = None, image_size: int = 224):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Không tìm thấy model tại: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.embedding_dim = ckpt.get("config", {}).get("embedding_dim", 128)

        self.model = FingerprintEmbeddingNet(embedding_dim=self.embedding_dim)
        self.model.load_state_dict(ckpt["backbone"] if "backbone" in ckpt else ckpt)
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def preprocess_fingerprint_method3(self, img: np.ndarray,
                                       top_rows=21, padding=15,
                                       block_size=16, variance_threshold=300,
                                       noise_kernel_size=3):
        if img is None or len(img.shape) == 0:
            return np.full((224, 224), 255, dtype=np.uint8)

        # Convert sang grayscale nếu là ảnh màu
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_original = img.copy()

        # 1. Xóa N hàng đầu
        h, w = img_original.shape
        img_clean = img_original[top_rows:h, 0:w]

        # 2. Enhance contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_clean)

        # 3. Denoise
        img_denoised = cv2.fastNlMeansDenoising(img_enhanced, None,
                                                h=10,
                                                templateWindowSize=7,
                                                searchWindowSize=21)

        # 4. Tạo variance mask
        h, w = img_denoised.shape
        variance_mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = img_denoised[i:i + block_size, j:j + block_size]
                if np.var(block) > variance_threshold:
                    variance_mask[i:i + block_size, j:j + block_size] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        variance_mask = cv2.morphologyEx(variance_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        variance_mask = cv2.morphologyEx(variance_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 5. Binarize (adaptive threshold)
        img_binary = cv2.adaptiveThreshold(img_denoised, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

        # 6. Clean binary noise
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (noise_kernel_size, noise_kernel_size))
        img_binary_clean = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN,
                                            kernel_noise, iterations=1)

        # 7. Áp variance mask lên ảnh binary ⭐ BƯỚC QUAN TRỌNG
        img_masked_processed = np.ones_like(img_binary_clean) * 255
        img_masked_processed[variance_mask > 0] = img_binary_clean[variance_mask > 0]

        # 8. Tạo final mask từ ảnh đã xử lý ⭐ BƯỚC QUAN TRỌNG
        _, final_mask = cv2.threshold(img_masked_processed, 250, 255, cv2.THRESH_BINARY_INV)
        kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_mask, iterations=2)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_mask, iterations=1)

        # 9. Áp mask ngược lên ảnh gốc (sau xóa hàng)
        img_final_masked = img_clean.copy()
        img_final_masked[final_mask == 0] = 255

        # 10. Crop theo mask với padding
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return img_final_masked

        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        if not valid_contours:
            return img_final_masked

        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        h_img, w_img = img_final_masked.shape
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)

        img_final = img_final_masked[y1:y2, x1:x2]

        return img_final

    def pre_process(self, image):
        """Hàm bọc tiền xử lý và resize cơ bản"""
        # Gọi Method 3 đã fix
        processed = self.preprocess_fingerprint_method3(image)

        # Đảm bảo trả về ảnh không rỗng
        if processed.size == 0:
            return np.full((self.image_size, self.image_size), 255, dtype=np.uint8)
        return processed

    @torch.no_grad()
    def extract(self, image_data: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        # 1. Chuyển đổi mọi đầu vào về Numpy Grayscale
        if isinstance(image_data, str):
            img_np = cv2.imread(image_data, 0)
        elif isinstance(image_data, np.ndarray):
            img_np = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) if len(image_data.shape) == 3 else image_data
        elif isinstance(image_data, Image.Image):
            img_np = np.array(image_data.convert("L"))

        if img_np is None: raise ValueError("Ảnh không hợp lệ")

        # 2. Tiền xử lý (Cắt, lọc, xóa nền)
        img_processed = self.pre_process(img_np)

        # 3. Chuyển sang Tensor để AI xử lý
        pil_img = Image.fromarray(img_processed).convert("L")
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 4. Extract
        embedding = self.model(img_tensor)
        return embedding.squeeze(0).cpu().numpy()

    def compute_similarity(self, embed1, embed2):
        return np.dot(embed1, embed2) / (np.linalg.norm(embed1) * np.linalg.norm(embed2) + 1e-10)
