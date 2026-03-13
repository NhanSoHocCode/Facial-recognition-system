import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
import time
import threading
from queue import Queue
from pathlib import Path
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- SETUP PATH ---
current_file = Path(__file__).resolve()
backend_path = current_file.parent.parent
sys.path.append(str(backend_path))

from core.database_mysql import MySQLManager


# ============================================================
# FACE ALIGNMENT MODULE
# ============================================================
class FaceAligner:
    """Face alignment using MediaPipe landmarks"""

    def __init__(self, model_path="face_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print("✓ Face alignment model loaded")

    def align_face(self, img, output_size=224):
        """
        Căn chỉnh khuôn mặt dựa trên landmarks
        Args:
            img: BGR image từ OpenCV
            output_size: kích thước output (224x224 cho model)
        Returns:
            aligned face hoặc None nếu không detect được
        """
        h, w = img.shape[:2]

        # Convert to MediaPipe format
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )

        res = self.landmarker.detect(mp_img)
        if not res.face_landmarks:
            return None

        lm = res.face_landmarks[0]

        # Eye landmarks indices
        LEFT_EYE = [33, 133]
        RIGHT_EYE = [362, 263]

        # Tính trung điểm mắt
        left_eye = np.mean([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE], axis=0)
        right_eye = np.mean([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE], axis=0)

        # Tính góc xoay
        dx, dy = right_eye - left_eye
        angle = np.degrees(np.arctan2(dy, dx))

        # Tâm xoay (giữa 2 mắt)
        center_x = int((left_eye[0] + right_eye[0]) / 2)
        center_y = int((left_eye[1] + right_eye[1]) / 2)
        eyes_center = (center_x, center_y)

        # Xoay ảnh
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))

        # Crop face bbox
        xs = np.array([p.x * w for p in lm])
        ys = np.array([p.y * h for p in lm])

        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())

        # Đảm bảo không vượt biên
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        face = rotated[y1:y2, x1:x2]

        if face.size == 0:
            return None

        return cv2.resize(face, (output_size, output_size))


# ============================================================
# YOLO FACE DETECTOR
# ============================================================
class YOLOFaceDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.conf_threshold = conf_threshold
        self.input_size = 640

    def detect_faces(self, image: np.ndarray):
        """Detect faces từ ảnh numpy (BGR format)"""
        img_height, img_width = image.shape[:2]

        # Preprocess
        img_resized = cv2.resize(image, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)

        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: img_batch})
        predictions = outputs[0]

        # Post-process
        faces = []
        if len(predictions.shape) == 3:
            predictions = predictions[0].T

            for pred in predictions:
                conf = pred[4]
                if conf > self.conf_threshold:
                    x_center, y_center, w, h = pred[:4]

                    x_center = x_center * img_width / self.input_size
                    y_center = y_center * img_height / self.input_size
                    w = w * img_width / self.input_size
                    h = h * img_height / self.input_size

                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)

                    faces.append((x1, y1, x2, y2, float(conf)))

        return faces


# ============================================================
# SPLIT FACE RECOGNITION MODEL
# ============================================================
class FaceRecognitionBackbone_Part1(nn.Module):
    """Part1: conv1 -> layer2 → Output features để Transformer sử dụng"""

    def __init__(self, pretrained_ckpt=None):
        super().__init__()
        base = models.wide_resnet101_2(weights='IMAGENET1K_V2')

        if pretrained_ckpt:
            print(f"Loading face recognition checkpoint: {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)

            if 'model' in checkpoint:
                face_state_dict = checkpoint['model']
                backbone_state_dict = {}

                for key, value in face_state_dict.items():
                    if key.startswith('backbone.'):
                        new_key = key.replace('backbone.', '')
                        backbone_state_dict[new_key] = value

                base.load_state_dict(backbone_state_dict, strict=False)
                print("✓ Loaded face recognition backbone weights")

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class FaceRecognitionBackbone_Part2(nn.Module):
    """Part2: layer3, layer4 + embedding → CHỈ CHẠY KHI DETECT REAL"""

    def __init__(self, pretrained_ckpt=None):
        super().__init__()
        base = models.wide_resnet101_2(weights='IMAGENET1K_V2')

        if pretrained_ckpt:
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)

            if 'model' in checkpoint:
                face_state_dict = checkpoint['model']
                backbone_state_dict = {}

                for key, value in face_state_dict.items():
                    if key.startswith('backbone.'):
                        new_key = key.replace('backbone.', '')
                        backbone_state_dict[new_key] = value

                base.load_state_dict(backbone_state_dict, strict=False)

        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.embed = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        if pretrained_ckpt and 'model' in checkpoint:
            embed_state = {}
            for key, value in checkpoint['model'].items():
                if key.startswith('embed.'):
                    embed_state[key.replace('embed.', '')] = value

            if embed_state:
                self.embed.load_state_dict(embed_state, strict=False)
                print("✓ Loaded face recognition embedding weights")

    def forward(self, x):
        """x: Output từ Part1 (B, 512, 28, 28)"""
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        embedding = self.embed(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


# ============================================================
# TRANSFORMER ENCODER
# ============================================================
class Transformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, depth=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos_embedding = nn.Parameter(torch.randn(1, 20, embed_dim))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.mean(dim=[3, 4])  # Global average pooling
        x = x + self.pos_embedding[:, :T, :]
        x = self.encoder(x)
        x = x.mean(dim=1)  # Temporal pooling
        return x


# ============================================================
# DEEPFAKE MODEL
# ============================================================
class DeepFakeModel_WithSplitModel(nn.Module):
    """DeepFake model sử dụng Part1 của Face Recognition"""

    def __init__(self, face_recognition_ckpt=None, freeze_backbone=True):
        super().__init__()

        self.backbone = FaceRecognitionBackbone_Part1(pretrained_ckpt=face_recognition_ckpt)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("✓ Face recognition backbone frozen")

        self.temporal_encoder = Transformer(
            embed_dim=512,
            num_heads=8,
            depth=4,
            mlp_ratio=2.0,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, frames, labels=None):
        """frames: (B, T, 3, 224, 224)"""
        B, T = frames.shape[:2]

        feats = []
        for t in range(T):
            ft = self.backbone(frames[:, t])
            feats.append(ft)

        feats = torch.stack(feats, dim=1)
        x = self.temporal_encoder(feats)
        logits = self.classifier(x)

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.view(-1),
                labels.float()
            )
            return logits, loss

        return logits


# ============================================================
# OPTIMIZED DETECTOR WITH FACE ALIGNMENT
# ============================================================
class RealtimeDeepFakeDetector_Optimized:
    """
    Pipeline tối ưu với Face Alignment:
    1. YOLO detect face
    2. MediaPipe align face (mới thêm)
    3. Part1 features → Transformer → DeepFake Classification
    4. IF REAL → Part2 → Face Recognition
    5. IF FAKE → Skip Face Recognition
    """

    def __init__(
            self,
            yolo_path,
            deepfake_model_path,
            face_recognition_ckpt,
            alignment_model_path="face_landmarker.task",
            num_frames=10,
            frame_skip=0,
            threshold=0.2,
            device="cuda",
            use_threading=True,
            use_alignment=True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Config cho việc register
        self.is_registering = True
        self.reg_count = 0
        self.max_reg = 3
        self.last_reg_time = 0
        self.reg_interval = 5
        
        # Load YOLO
        print("Loading YOLO face detector...")
        self.yolo = YOLOFaceDetector(yolo_path, conf_threshold=0.5)

        # Load Face Aligner
        self.use_alignment = use_alignment
        if self.use_alignment:
            try:
                print("Loading face alignment model...")
                self.aligner = FaceAligner(alignment_model_path)
            except Exception as e:
                print(f"⚠ Warning: Could not load face aligner: {e}")
                print("  Continuing without alignment...")
                self.use_alignment = False

        # Load DeepFake model
        print("Loading DeepFake model...")
        self.deepfake_model = DeepFakeModel_WithSplitModel(
            face_recognition_ckpt=face_recognition_ckpt,
            freeze_backbone=True
        ).to(self.device)

        if deepfake_model_path:
            checkpoint = torch.load(deepfake_model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.deepfake_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ DeepFake model loaded (AUC: {checkpoint.get('auc', 'N/A'):.4f})")
            else:
                self.deepfake_model.load_state_dict(checkpoint)
                print("✓ DeepFake model loaded")

        self.deepfake_model.eval()

        # Load Face Recognition Part2
        print("Loading Face Recognition Part2...")
        self.face_part2 = FaceRecognitionBackbone_Part2(
            pretrained_ckpt=face_recognition_ckpt
        ).to(self.device)
        self.face_part2.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Config
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.threshold = threshold
        self.use_threading = use_threading

        # Buffers
        self.frame_buffer = deque(maxlen=num_frames)
        self.overlap_frames = 3
        self.score_history = deque(maxlen=3)
        self.frame_counter = 0
        self.is_initial_fill = True

        # Cache
        self.last_bbox = None
        self.last_label = "Initializing..."
        self.last_prob = None
        self.last_embedding = None
        self.last_recognition_result = None

        # Threading
        if use_threading:
            self.inference_queue = Queue(maxsize=2)
            self.result_queue = Queue(maxsize=2)
            self.inference_thread = None
            self.running = False

        # Reference
        self.reference_embedding = None
        self.reference_name = None
        self.recognition_threshold = 0.6
        self.db = MySQLManager()

        self.stats = {
            'total_inferences': 0,
            'fake_detected': 0,
            'real_detected': 0,
            'face_recognition_runs': 0,
            'alignment_success': 0,
            'alignment_fail': 0
        }

        print("✓ Optimized detector ready")
        if self.use_alignment:
            print("✓ Face alignment enabled\n")
        else:
            print("⚠ Face alignment disabled\n")

    def load_reference_image(self, image_path, name="Reference"):
        """Load ảnh reference và tính embedding (có alignment)"""
        print(f"Loading reference image: {image_path}")

        img = Image.open(image_path).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        faces = self.yolo.detect_faces(img_cv)

        if len(faces) > 0:
            x1, y1, x2, y2, conf = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
            face_crop = img_cv[y1:y2, x1:x2]

            if self.use_alignment:
                aligned_face = self.aligner.align_face(face_crop, output_size=224)
                if aligned_face is not None:
                    face_crop = aligned_face
                    print(f"  ✓ Face aligned")

            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(face_rgb)
            print(f"  ✓ Detected face (conf: {conf:.3f})")
        else:
            print(f"  ⚠ No face detected, using full image")

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.deepfake_model.backbone(img_tensor)
            embedding = self.face_part2(feat)

        self.reference_embedding = embedding
        self.reference_name = name

        print(f"✓ Reference embedding loaded for '{name}'")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding norm: {torch.norm(embedding).item():.4f}\n")

    def compare_with_reference(self, current_embedding):
        """So sánh embedding với reference"""
        if self.reference_embedding is None:
            return False, 0.0

        similarity = F.cosine_similarity(
            current_embedding,
            self.reference_embedding
        ).item()

        is_same = similarity > self.recognition_threshold
        return is_same, similarity

    def detect_and_crop_face(self, frame, expand_ratio=0.01):
        """
        UPDATED: Detect, crop, và align face
        """
        faces = self.yolo.detect_faces(frame)
        if len(faces) == 0:
            return None, None

        img_h, img_w = frame.shape[:2]
        img_area = img_h * img_w

        MIN_AREA_RATIO = 0.03

        valid_faces = []
        for (x1, y1, x2, y2, conf) in faces:
            area = (x2 - x1) * (y2 - y1)
            if area / img_area >= MIN_AREA_RATIO:
                valid_faces.append((x1, y1, x2, y2, conf))

        if len(valid_faces) == 0:
            return None, None

        x1, y1, x2, y2, conf = max(
            valid_faces,
            key=lambda f: (f[2]-f[0]) * (f[3]-f[1])
        )

        w, h = x2 - x1, y2 - y1
        pad = int(expand_ratio * max(w, h))

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_w, x2 + pad)
        y2 = min(img_h, y2 + pad)

        face = frame[y1:y2, x1:x2]

        if self.use_alignment and face.size > 0:
            aligned_face = self.aligner.align_face(face, output_size=224)
            if aligned_face is not None:
                face = aligned_face
                self.stats['alignment_success'] += 1
            else:
                self.stats['alignment_fail'] += 1
                # Fallback: resize normally
                face = cv2.resize(face, (224, 224))

        return face, (x1, y1, x2, y2, conf)

    def inference_worker(self):
        """Inference worker with alignment support"""
        while self.running:
            try:
                data = self.inference_queue.get(timeout=0.1)
                if data is None:
                    break

                frames_tensor = data

                with torch.no_grad():
                    B, T = frames_tensor.shape[:2]

                    feats = []
                    for t in range(T):
                        ft = self.deepfake_model.backbone(frames_tensor[:, t])
                        feats.append(ft)

                    feats = torch.stack(feats, dim=1)

                    x = self.deepfake_model.temporal_encoder(feats)
                    logits = self.deepfake_model.classifier(x)
                    prob = torch.sigmoid(logits).item()

                    self.stats['total_inferences'] += 1

                self.score_history.append(prob)
                smooth_prob = sum(self.score_history) / len(self.score_history)
                label = "FAKE" if smooth_prob > self.threshold else "REAL"

                embedding = None
                recognition_result = None

                if label == "REAL":
                    with torch.no_grad():
                        last_frame_features = feats[:, -1, :, :, :]
                        embedding = self.face_part2(last_frame_features)

                    self.stats['face_recognition_runs'] += 1
                    self.stats['real_detected'] += 1

                    if self.reference_embedding is not None:
                        is_same, similarity = self.compare_with_reference(embedding)
                        recognition_result = (is_same, similarity)
                else:
                    self.stats['fake_detected'] += 1

                self.result_queue.put((label, smooth_prob, embedding, recognition_result))

            except Exception as e:
                pass

    def process_frame_fast(self, frame):
        """Process frame with alignment"""
        face, bbox = self.detect_and_crop_face(frame)

        if face is None:
            return None, "No face", None, None

        self.last_bbox = bbox
        
        if self.is_initial_fill:
            should_process = True
        else:
            should_process = (self.frame_counter % (self.frame_skip + 1)) == 0

        if should_process:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = self.transform(face_pil)
            self.frame_buffer.append(face_tensor)

            if self.is_initial_fill and len(self.frame_buffer) >= self.num_frames:
                self.is_initial_fill = False

            if len(self.frame_buffer) >= self.num_frames:
                frames_tensor = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)

                if self.use_threading:
                    if not self.inference_queue.full():
                        self.inference_queue.put(frames_tensor)
                        self.slide_buffer()
                else:
                    with torch.no_grad():
                        B, T = frames_tensor.shape[:2]

                        feats = []
                        for t in range(T):
                            ft = self.deepfake_model.backbone(frames_tensor[:, t])
                            feats.append(ft)

                        feats = torch.stack(feats, dim=1)

                        x = self.deepfake_model.temporal_encoder(feats)
                        logits = self.deepfake_model.classifier(x)
                        prob = torch.sigmoid(logits).item()

                        self.stats['total_inferences'] += 1

                    self.score_history.append(prob)
                    smooth_prob = sum(self.score_history) / len(self.score_history)
                    label = "FAKE" if smooth_prob > self.threshold else "REAL"

                    self.last_label = label
                    self.last_prob = smooth_prob

                    if label == "REAL":
                        with torch.no_grad():
                            last_frame_features = feats[:, -1, :, :, :]
                            self.last_embedding = self.face_part2(last_frame_features)

                        self.stats['face_recognition_runs'] += 1
                        self.stats['real_detected'] += 1

                        if self.reference_embedding is not None:
                            is_same, similarity = self.compare_with_reference(self.last_embedding)
                            self.last_recognition_result = (is_same, similarity)
                    else:
                        self.stats['fake_detected'] += 1
                        self.last_embedding = None
                        self.last_recognition_result = None

                    self.slide_buffer()

        return bbox, self.last_label, self.last_prob, self.last_recognition_result

    def slide_buffer(self):
        """Sliding window: Xóa frames cũ nhất, giữ lại overlap frames"""
        if len(self.frame_buffer) < self.num_frames:
            return

        frames_to_keep = list(self.frame_buffer)[-self.overlap_frames:]
        self.frame_buffer.clear()
        for frame in frames_to_keep:
            self.frame_buffer.append(frame)

    def run_camera_registration(self, camera_id=0, user_id=None):
        """
        Hàm run_camera cho đăng ký với Face Alignment
        """
        if user_id is None:
            print("LỖI: user_id không được để trống!")
            return
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("ERROR: Không thể mở camera!")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_registering = True
        self.reg_count = 0
        self.last_reg_time = time.time()
        
        if self.use_threading:
            self.running = True
            self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
            self.inference_thread.start()

        print(f"BẮT ĐẦU ĐĂNG KÝ CHO USER: {user_id}")
        if self.use_alignment:
            print("✓ Face Alignment: ENABLED")

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                # Update Deepfake Result
                if self.use_threading:
                    try:
                        label, prob, _, _ = self.result_queue.get_nowait()
                        self.last_label = label
                        self.last_prob = prob
                        self.slide_buffer()
                    except: pass

                # Detect Face
                bbox, label, prob, _ = self.process_frame_fast(frame)
                
                # Hiển thị UI
                if bbox is not None:
                    x1, y1, x2, y2, _ = bbox
                    color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Anti-Fake: {label}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Logic Đăng ký tự động
                    if self.is_registering and label == "REAL":
                        current_time = time.time()
                        elapsed = current_time - self.last_reg_time
                        countdown = self.reg_interval - elapsed

                        if countdown <= 0:
                            # Chụp ảnh
                            face_img = frame[y1:y2, x1:x2]
                            if face_img.size > 0:
                                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                face_pil = Image.fromarray(face_rgb)
                                face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

                                with torch.no_grad():
                                    feat_p1 = self.deepfake_model.backbone(face_tensor)
                                    embedding = self.face_part2(feat_p1)
                                    vector_np = embedding.cpu().numpy().flatten()
                                
                                # Lưu DB
                                self.db.add_embedding_recognition(user_id, vector_np)
                                
                                self.reg_count += 1
                                self.last_reg_time = current_time
                                print(f"✓ Đã lưu lần {self.reg_count}/3")
                        else:
                            cv2.putText(frame, f"Chuan bi chup {self.reg_count+1}/3 trong: {int(countdown)+1}s", 
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    elif label != "REAL":
                        cv2.putText(frame, "CANH BAO GIA MAO - TAM DUNG", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if self.use_alignment:
                    total_align = self.stats['alignment_success'] + self.stats['alignment_fail']
                    if total_align > 0:
                        success_rate = (self.stats['alignment_success'] / total_align) * 100
                        align_text = f"Aligned: {self.stats['alignment_success']}/{total_align} ({success_rate:.1f}%)"
                        cv2.putText(frame, align_text, (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.imshow("Registration with Alignment", frame)
                
                if (cv2.waitKey(1) & 0xFF == ord('q')) or (self.reg_count >= self.max_reg):
                    break

        finally:
            self.running = False
            if self.use_threading and hasattr(self, 'inference_thread'):
                self.inference_queue.put(None)
            cap.release()
            cv2.destroyAllWindows()
            
            # In thống kê cuối
            print("\n" + "="*50)
            print("REGISTRATION COMPLETED")
            print("="*50)
            print(f"Total registrations: {self.reg_count}")
            if self.use_alignment:
                total_align = self.stats['alignment_success'] + self.stats['alignment_fail']
                if total_align > 0:
                    success_rate = (self.stats['alignment_success'] / total_align) * 100
                    print(f"Alignment success: {self.stats['alignment_success']}/{total_align} ({success_rate:.1f}%)")
            print("="*50)

    def verify_admin(self, camera_id=0):
        """
        Xác minh Admin tự động với Face Alignment
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("ERROR: Không thể mở camera!")
            return False

        if self.use_threading:
            self.running = True
            self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
            self.inference_thread.start()

        found_user = None
        start_verify_process = time.time()
        last_db_check_time = 0
        verify_interval = 1.0
        timeout_limit = 20.0

        print("\n" + "="*50)
        print(f"ADMIN VERIFICATION WITH ALIGNMENT (Timeout: {timeout_limit}s)")
        if self.use_alignment:
            print("✓ Face Alignment: ENABLED")
        print("="*50)

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                current_time = time.time()
                elapsed_time = current_time - start_verify_process
                remaining_time = max(0, timeout_limit - elapsed_time)

                # Timeout check
                if elapsed_time > timeout_limit:
                    print("! [TIMEOUT] Không thể xác minh Admin trong thời gian quy định.")
                    found_user = None
                    break

                # Update Deepfake Result
                if self.use_threading:
                    try:
                        label, prob, _, _ = self.result_queue.get_nowait()
                        self.last_label = label
                        self.last_prob = prob
                        self.slide_buffer()
                    except: pass

                # Process Frame
                bbox, label, prob, _ = self.process_frame_fast(frame)
                
                if bbox is not None:
                    x1, y1, x2, y2, _ = bbox
                    
                    color = (0, 255, 0) if self.last_label == "REAL" else (0, 0, 255)
                    if self.last_label == "Initializing...": color = (255, 165, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Anti-Spoofing: {self.last_label}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Logic xác minh chính
                    if self.last_label == "REAL":
                        if current_time - last_db_check_time >= verify_interval:
                            face_img = frame[y1:y2, x1:x2]
                            if face_img.size > 0:
                                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                face_pil = Image.fromarray(face_rgb)
                                face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

                                with torch.no_grad():
                                    feat_p1 = self.deepfake_model.backbone(face_tensor)
                                    embedding = self.face_part2(feat_p1)
                                    vector_np = embedding.cpu().numpy().flatten()
                                
                                result = self.db.find_user_by_face(vector_np, isFace=True, isAdmin=True)
                                last_db_check_time = current_time

                                if result:
                                    found_user = result
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    cv2.putText(frame, "ADMIN VERIFIED!", (x1, y1 - 40), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        if not found_user:
                            cv2.putText(frame, "Matching Face...", (x1, y2 + 25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    
                    elif self.last_label == "FAKE":
                        cv2.putText(frame, "FAKE DETECTED - BLOCKED", (10, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if self.use_alignment:
                    total_align = self.stats['alignment_success'] + self.stats['alignment_fail']
                    if total_align > 0:
                        success_rate = (self.stats['alignment_success'] / total_align) * 100
                        align_text = f"Aligned: {self.stats['alignment_success']}/{total_align} ({success_rate:.1f}%)"
                        cv2.putText(frame, align_text, (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.rectangle(frame, (10, 10), (210, 30), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 10), (10 + int(remaining_time * 10), 30), (0, 165, 255), -1)
                cv2.putText(frame, f"Time left: {int(remaining_time)}s", (15, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Admin Verification System", frame)

                if (cv2.waitKey(1) & 0xFF == ord('q')) or found_user:
                    if found_user: cv2.waitKey(1000)
                    break

        except Exception as e:
            print(f"Lỗi hệ thống: {e}")
            found_user = None
            
        finally:
            self.running = False
            if self.use_threading and hasattr(self, 'inference_thread'):
                self.inference_queue.put(None)
                self.inference_thread.join(timeout=1)
            
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*50)
            print("VERIFICATION COMPLETED")
            print("="*50)
            print(f"Result: {'SUCCESS' if found_user else 'FAILED'}")
            print(f"Total inferences: {self.stats['total_inferences']}")
            print(f"REAL detected: {self.stats['real_detected']}")
            print(f"FAKE detected: {self.stats['fake_detected']}")
            if self.use_alignment:
                total_align = self.stats['alignment_success'] + self.stats['alignment_fail']
                if total_align > 0:
                    success_rate = (self.stats['alignment_success'] / total_align) * 100
                    print(f"Alignment success: {self.stats['alignment_success']}/{total_align} ({success_rate:.1f}%)")
            print("="*50)
            
            return found_user


# ============================================================
# MAIN
# ============================================================
backend_dir = Path(__file__).resolve().parent.parent 
RECOG_PATH = backend_dir / "models" / "faceRecognition_arcface_ckpt(2).pth"
YOLO_PATH = backend_dir / "models" / "yolov8s-face-lindevs.onnx"
DEEPFAKE_PATH = backend_dir / "models" / "deepfake_best5.pth"

if __name__ == "__main__":
    camera_id = 0 
    user_id = 2
    
    # ⭐ Khởi tạo với Face Alignment
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
    
    mode = "register"
    
    if mode == "register":
        app.run_camera_registration(camera_id=camera_id, user_id=user_id)
    else:
        result = app.verify_admin(camera_id=camera_id)
        print(f"\nFinal Result: {'VERIFIED ✓' if result else 'FAILED ✗'}")