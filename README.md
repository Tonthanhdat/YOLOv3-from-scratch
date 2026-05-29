# YOLOv3 Object Detection từ đầu bằng PyTorch

Đây là bộ code cài đặt mô hình phát hiện đối tượng YOLOv3 từ đầu, không sử dụng các framework có sẵn (như YOLOv5, Detectron2,...).
Mô hình tự xây dựng kiến trúc mạng nguyên bản, cấu hình Anchor, xử lý hàm mất mát (Loss) và Non-Maximum Suppression (NMS) phục vụ cho đồ án phân tích ảnh. Toàn bộ kiến trúc và các chi tiết thuật toán được ghi lại ở `docs/technical_details.md`.

## 1. Cài đặt môi trường
Bạn có thể thiết lập mô hình dễ dàng trên các môi trường có hỗ trợ GPU như Kaggle, Google Colab hoặc máy cá nhân.

```bash
pip install -r requirements.txt
```

## 2. Chuẩn bị dữ liệu
Dữ liệu cần được giải nén vào thư mục `public/` theo đúng cấu trúc đề bài:
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

**(Tùy chọn) Tính toán Anchor Boxes:**
Để có độ chính xác tốt nhất, bạn nên chạy file K-means để tìm cấu hình Anchor (hộp neo) tối ưu nhất với dữ liệu, sau đó thay thế kết quả hiển thị vào biến `ANCHORS` trong file `config.py`.
```bash
python utils/kmeans_anchor.py
```

## 3. Huấn luyện (Train)
Chạy lệnh sau để tiến hành huấn luyện mô hình. Tham số `--checkpoint_dir` sẽ quyết định nơi lưu model. Trong quá trình học, weights mô hình tốt nhất sẽ được lưu dưới dạng `./models/best.pth`.
```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

## 4. Suy luận (Predict)
Để chạy nhận diện (inference), sử dụng lệnh dưới. Kết quả nhận diện sẽ được lưu dưới dạng file JSON đúng cấu trúc đầu ra với các tọa độ được chuyển đổi về tỉ lệ ảnh gốc.
```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json
```

## 5. Đánh giá tự động (Evaluate)
Sau khi có `predictions.json`, sử dụng công cụ có sẵn trong tệp dữ liệu gốc để kiểm tra mAP:
```bash
python public/tools/evaluate_predictions.py \
  --ground_truth ./public/annotations/val.json \
  --predictions predictions.json \
  --output val_score.json
```
