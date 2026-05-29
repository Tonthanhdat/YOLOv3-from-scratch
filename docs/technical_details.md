# Thông tin kỹ thuật & Thiết kế kiến trúc YOLOv3

File này đóng vai trò là "Single Source of Truth" (Nguồn chân lý duy nhất) cho toàn bộ các thiết kế kỹ thuật, thuật toán và công thức toán học được áp dụng trong quá trình triển khai mô hình YOLOv3 từ đầu cho đồ án này.

## 1. Kiến trúc mô hình (Architecture)
YOLOv3 là mô hình một giai đoạn (one-stage object detector) dự đoán bounding box dựa trên grid cell ở 3 tỉ lệ khác nhau (multi-scale).

### 1.1. Backbone (Mạng trích xuất đặc trưng)
- **Chuẩn YOLOv3**: Sử dụng `Darknet-53` gồm 53 lớp tích chập với các residual block.
- **Tùy chọn thay thế khả thi**: `ResNet-50` (nếu giảng viên cho phép dùng pre-trained) để hội tụ nhanh hơn.
- Đầu ra của Backbone cung cấp 3 feature maps ở 3 scale khác nhau (ví dụ nếu ảnh đầu vào là 416x416 thì feature maps sẽ có kích thước: 52x52, 26x26, 13x13).

### 1.2. Neck (Cổ mạng)
- Sử dụng **FPN (Feature Pyramid Network)**: Upsample các feature map ở độ phân giải thấp và nối (concatenate) với feature map ở độ phân giải cao hơn từ backbone để tạo ra các feature giàu ngữ nghĩa và chi tiết cho từng scale.

### 1.3. Head (Đầu dự đoán)
Tại mỗi scale (ví dụ 13x13), mạng chia ảnh thành grid `13x13`.
Tại mỗi grid cell, dự đoán `B = 3` anchor boxes.
Với mỗi anchor box, mạng dự đoán `4 (tọa độ)` + `1 (objectness)` + `C (số lượng class, ở đây C=5)` giá trị.
- Vậy output channels tại mỗi scale là: `3 * (4 + 1 + 5) = 30` channels.

## 2. Anchor Boxes (Hộp neo)
YOLOv3 sử dụng K-Means clustering trên tập dữ liệu train để tìm ra kích thước tối ưu của các Anchor boxes.
Cần có 9 anchors tổng cộng chia cho 3 scales (mỗi scale 3 anchors).
- **Scale 1 (13x13 - Lưới to, phát hiện vật lớn)**: Dùng 3 anchors lớn nhất.
- **Scale 2 (26x26 - Lưới vừa)**: Dùng 3 anchors trung bình.
- **Scale 3 (52x52 - Lưới nhỏ, phát hiện vật nhỏ)**: Dùng 3 anchors nhỏ nhất.

## 3. Công thức tính Bounding Box
Mạng không dự đoán trực tiếp tọa độ hộp bao, mà dự đoán các phần bù (offsets):
- `t_x, t_y`: Độ dịch chuyển tâm.
- `t_w, t_h`: Tỉ lệ chiều rộng, cao.
- `p_w, p_h`: Kích thước gốc của Anchor box ứng với grid cell đó.
- `c_x, c_y`: Tọa độ góc trên bên trái của grid cell.

Công thức giải mã để lấy tâm (`b_x, b_y`) và kích thước (`b_w, b_h`) trên kích thước của feature map:
- `b_x = sigmoid(t_x) + c_x`
- `b_y = sigmoid(t_y) + c_y`
- `b_w = p_w * exp(t_w)`
- `b_h = p_h * exp(t_h)`

## 4. Hàm Mất Mát (Loss Function)
Loss = Loss_Localization + Loss_Objectness + Loss_Classification

- **Localization Loss (Bbox Loss)**: Có thể dùng `MSE` trên tọa độ (như YOLO gốc) hoặc tiên tiến hơn là `GIoU / CIoU Loss` để bao hàm tốt hình học.
- **Objectness Loss (Confidence)**: Sử dụng `Binary Cross Entropy (BCE)`. Tính theo hai phần:
  - Lưới chứa đối tượng (Loss_obj).
  - Lưới KHÔNG chứa đối tượng (Loss_noobj). Thường có trọng số nhỏ hơn để tránh lấn át do quá nhiều negative samples.
- **Classification Loss**: Sử dụng `BCE` hoặc `CrossEntropy` để phạt lỗi phân loại ở các ô có chứa đối tượng.

## 5. Quy trình huấn luyện & Tiền xử lý (Pipeline)
- **Resize**: Cần resize ảnh với kĩ thuật `Letterbox` (thêm padding đen) để giữ nguyên tỉ lệ khung hình (aspect ratio) của ảnh gốc.
- **Augmentation**:
  - Horizontal Flip (lật ngang).
  - Color Jitter (thay đổi độ sáng, độ tương phản).
  - Mosaic / Mixup (nếu cần cải thiện hiệu năng).
- **Target building**: Chuyển các bounding box từ định dạng JSON/tọa độ thật thành ma trận label phù hợp với kích thước `[batch_size, 3, grid_size, grid_size, 10]`.

## 6. Suy luận & NMS (Inference & Non-Maximum Suppression)
- Sau khi có các boxes dự đoán, chuyển ngược `b_x, b_y, b_w, b_h` về hệ tọa độ điểm ảnh của hình gốc (đảo ngược quá trình Letterbox).
- Lọc bỏ các box có độ tin cậy `< conf_threshold`.
- Áp dụng **NMS** theo từng class: Lọc và loại bỏ các box bị chồng lấn cao hơn ngưỡng `nms_threshold` so với box tự tin nhất của cùng một vật thể.
