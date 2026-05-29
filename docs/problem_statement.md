# Yêu Cầu Đồ Án: Cài đặt một mô hình phát hiện đối tượng (object detection) từ đầu

Sinh viên cần tự xây dựng quy trình huấn luyện và suy luận cho bài toán phát hiện đối tượng, bao gồm:

- Đọc dữ liệu và tiền xử lý ảnh/nhãn.
- Tăng cường dữ liệu.
- Mạng trích xuất đặc trưng, ví dụ mạng tích chập CNN.
- Đầu dự đoán phát hiện đối tượng.
- Hàm mất mát.
- Suy luận, ngưỡng độ tin cậy và khử trùng hộp bao bằng NMS (Non-Maximum Suppression).

Không được sử dụng các bộ phát hiện đối tượng hoàn chỉnh như YOLOv5/v8, Detectron2, MMDetection, hoặc Faster R-CNN/SSD có sẵn trong torchvision. Được phép dùng PyTorch, các lớp mạng cơ bản, và mạng trích xuất đặc trưng đã huấn luyện trước nếu giảng viên cho phép.

## Bộ Dữ Liệu
Bộ dữ liệu gồm ảnh tự nhiên có đối tượng thuộc 5 lớp:
1. `person`
2. `car`
3. `dog`
4. `cat`
5. `chair`

Cấu trúc thư mục:
```text
public/
├── classes.json
├── train/
│   └── images/
├── val/
│   └── images/
├── annotations/
│   ├── train.json
│   └── val.json
└── tools/
    └── evaluate_predictions.py
```
Trong thư mục `public/` chỉ có tập huấn luyện và tập kiểm định. Khi chấm tự động, hệ thống mới cung cấp thư mục ảnh kiểm tra ẩn cho `predict.py`; nhãn của tập kiểm tra ẩn được giữ riêng và không công bố.

## Định Dạng Nhãn
Tệp `train.json` và `val.json` có dạng:

```json
{
  "classes": ["person", "car", "dog", "cat", "chair"],
  "images": [
    {
      "id": "img_a13f42c9d8b0.jpg",
      "file_name": "train/images/img_a13f42c9d8b0.jpg",
      "width": 500,
      "height": 375
    }
  ],
  "annotations": [
    {
      "image_id": "img_a13f42c9d8b0.jpg",
      "class": "person",
      "bbox": [48, 72, 210, 356]
    }
  ]
}
```
Quy ước hộp bao:
`bbox = [xmin, ymin, xmax, ymax]`
Tọa độ tính theo điểm ảnh trên ảnh gốc.

## Yêu Cầu Kĩ Thuật

1. **Quy Trình Dữ Liệu**:
   - Bộ đọc dữ liệu.
   - Thay đổi kích thước ảnh và chuẩn hóa giá trị điểm ảnh.
   - Xử lý nhiều đối tượng trong cùng một ảnh.
   - Tăng cường dữ liệu, tối thiểu gồm lật ngang ảnh. Khuyến khích cắt ngẫu nhiên, thay đổi màu sắc, và huấn luyện với nhiều kích thước ảnh.

2. **Mô Hình Phát Hiện Đối Tượng**:
   - Dự đoán: Hộp bao, Nhãn lớp, Điểm độ tin cậy.
   - Các hướng: anchor-based, anchor-free, mạng lưới kiểu YOLO, hoặc kiểu SSD tự cài đặt.

3. **Hàm Mất Mát**:
   - Mất mát phân lớp.
   - Mất mát định vị hộp bao.
   - Mất mát độ tin cậy.
   - Khuyến khích dùng Cross Entropy, BCE, Smooth L1, IoU/GIoU/DIoU.

4. **Suy Luận**:
   - Ngưỡng độ tin cậy.
   - NMS theo từng lớp.
   - Chuyển hộp bao về tọa độ ảnh gốc.

## Yêu Cầu Nộp Bài
Cấu trúc thư mục mã nguồn:
```text
<my_submission>/
├── public/ 
├── models/
├── utils/
├── train.py
├── predict.py
├── README.md
└── requirements.txt
```

**Lệnh suy luận bắt buộc:**
```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

**Lệnh huấn luyện bắt buộc:**
```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```
Lưu mô hình tốt nhất vào `./models/best.pth`.

**Tệp README.md cần nêu rõ:**
- Cách cài đặt môi trường.
- Cách huấn luyện.
- Cách chạy suy luận.
- Vị trí đặt mô hình.

## Định Dạng Kết Quả Dự Đoán
`predictions.json` phải là một mảng JSON:
```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```
Ảnh không phát hiện đối tượng nào vẫn cần xuất `"boxes": []`.

## Chấm Tự Động & Thang Điểm
- Quy trình dữ liệu: 20đ
- Kiến trúc mô hình: 20đ
- Hàm mất mát và huấn luyện: 20đ
- Suy luận và NMS: 20đ
- Kết quả trên tập kiểm tra ẩn: 20đ (Dựa vào mAP@0.5)

**Thang điểm mAP@0.5:**
- `< 0.30`: 0đ
- `0.30 - < 0.45`: 5đ
- `0.45 - < 0.60`: 10đ
- `0.60 - < 0.75`: 15đ
- `>= 0.75`: 20đ
