import torch
import os

# Đường dẫn thư mục gốc
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")

# Cấu hình huấn luyện cơ bản
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = True

# Cấu hình mô hình
IMAGE_SIZE = 416 # Resize ảnh về 416x416
NUM_CLASSES = 5
CLASSES = ["person", "car", "dog", "cat", "chair"]

# Anchors (Khởi tạo default của YOLOv3, sẽ dùng KMeans để tính lại sau)
ANCHORS = [
    [(116, 90), (156, 198), (373, 326)], # Scale 13x13 (Stride 32)
    [(30, 61), (62, 45), (59, 119)],     # Scale 26x26 (Stride 16)
    [(10, 13), (16, 30), (33, 23)],      # Scale 52x52 (Stride 8)
]

# Tỉ lệ grid so với input size (Scale ratios)
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

# Ngưỡng cho NMS & loss
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45
IGNORE_IOU_THRESH = 0.5 # Ngưỡng bỏ qua target cho prediction box nếu IoU > threshold
