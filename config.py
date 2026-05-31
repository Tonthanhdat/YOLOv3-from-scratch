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
NUM_EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = True

# Cấu hình mô hình
IMAGE_SIZE = 416 # Resize ảnh về 416x416
NUM_CLASSES = 5
CLASSES = ["person", "car", "dog", "cat", "chair"]

# Anchors (Khởi tạo default của YOLOv3, sẽ dùng KMeans để tính lại sau)
ANCHORS = [
    [(199, 146), (146, 237), (292, 276)], # Scale 13x13 (Stride 32)
    [(50, 106), (96, 77), (91, 162)],     # Scale 26x26 (Stride 16)
    [(18, 24), (27, 62), (51, 37)],       # Scale 52x52 (Stride 8)
]

# Tỉ lệ grid so với input size (Scale ratios)
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

# Ngưỡng cho NMS & loss
CONF_THRESHOLD = 0.1  # Giảm xuống 0.1 để giữ recall khi tính mAP
NMS_THRESHOLD = 0.45  # Trả về mặc định 0.45 để không xóa nhầm box
IGNORE_IOU_THRESH = 0.5 # Ngưỡng bỏ qua target cho prediction box nếu IoU > threshold
