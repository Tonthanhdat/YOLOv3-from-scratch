import torch

def iou_width_height(box1, boxes2):
    """
    Tính IoU dựa trên width và height của box so với anchor box (chung tâm).
    box1: tensor [w, h]
    boxes2: tensor [N, 2] [w, h]
    """
    intersection = torch.min(box1[0], boxes2[:, 0]) * torch.min(box1[1], boxes2[:, 1])
    union = (box1[0] * box1[1]) + (boxes2[:, 0] * boxes2[:, 1]) - intersection
    return intersection / (union + 1e-6)

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Tính IoU giữa dự đoán và label.
    box_format: "midpoint" [x_mid, y_mid, w, h] hoặc "corners" [x1, y1, x2, y2]
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) để khi không có giao nhau thì trả về 0 thay vì số âm
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Thực hiện Non Max Suppression.
    bboxes: list [[class_pred, prob_score, x1, y1, x2, y2], ...]
    """
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes_after_nms.append(chosen_box)
        
        # Chỉ giữ lại các box có class khác với chosen box HOẶC có iou < threshold
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            ).item() < iou_threshold
        ]

    return bboxes_after_nms

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Chuyển đổi tensor dự đoán (hoặc target) dạng Grid-scale về dạng box tương đối [0, 1] 
    trên toàn ảnh: [batch_size, num_anchors * S * S, 6] (6 là [class, score, x, y, w, h])
    Format của predictions/target: [x, y, w, h, objectness, class_0, class_1, ...]
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    
    box_predictions = predictions[..., 0:4].clone()
    
    if is_preds:
        anchors = anchors.reshape(1, num_anchors, 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:4] = torch.exp(box_predictions[..., 2:4]) * anchors
        
        obj_scores = torch.sigmoid(predictions[..., 4:5])
        class_probs = torch.softmax(predictions[..., 5:], dim=-1)
        class_scores, best_class = torch.max(class_probs, dim=-1, keepdim=True)
        
        scores = obj_scores * class_scores
    else:
        scores = predictions[..., 4:5]
        best_class = torch.argmax(predictions[..., 5:], dim=-1, keepdim=True)

    cell_indices = (
        torch.arange(S)
        .repeat(BATCH_SIZE, num_anchors, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    
    x = (box_predictions[..., 0:1] + cell_indices) / S
    y = (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)) / S
    w_h = box_predictions[..., 2:4] / S
    
    converted_bboxes = torch.cat((best_class.float(), scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()
