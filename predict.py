import torch
import argparse
import os
import json
from PIL import Image
from tqdm import tqdm

import config
from models.yolov3 import YOLOv3
from utils.transforms import YoloTransforms
from utils.boxes import cells_to_bboxes, non_max_suppression

def main(args):
    device = config.DEVICE
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(device)
    
    # Load model. Đề bài yêu cầu best.pth lưu ở ./models/
    # Truyền vào --checkpoint_dir='./models/' khi train. Ở predict sẽ tự trỏ vào models/best.pth
    model_path = os.path.join(os.path.dirname(__file__), "models", "best.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Warning: No weights found at {model_path}. Using random weights.")
    
    model.eval()
    
    transform = YoloTransforms(image_size=config.IMAGE_SIZE, is_train=False)
    
    predictions_list = []
    
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(args.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        img_tensor, _ = transform(image, [])
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(img_tensor)
            bboxes = []
            for i in range(3):
                S = out[i].shape[2]
                anchor = scaled_anchors[i]
                boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
                for idx, (box) in enumerate(boxes_scale_i[0]):
                    bboxes.append(box)

            nms_boxes = non_max_suppression(
                bboxes, iou_threshold=config.NMS_THRESHOLD, threshold=config.CONF_THRESHOLD, box_format="midpoint"
            )
            
            # Reverse letterbox scale
            scale = min(config.IMAGE_SIZE / orig_w, config.IMAGE_SIZE / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            pad_w = (config.IMAGE_SIZE - new_w) // 2
            pad_h = (config.IMAGE_SIZE - new_h) // 2
            
            img_boxes = []
            for box in nms_boxes:
                class_id = int(box[0])
                confidence = float(box[1])
                x_mid, y_mid, w_rel, h_rel = box[2], box[3], box[4], box[5]
                
                abs_x_mid = x_mid * config.IMAGE_SIZE
                abs_y_mid = y_mid * config.IMAGE_SIZE
                abs_w = w_rel * config.IMAGE_SIZE
                abs_h = h_rel * config.IMAGE_SIZE
                
                abs_x_mid -= pad_w
                abs_y_mid -= pad_h
                
                orig_x_mid = abs_x_mid / scale
                orig_y_mid = abs_y_mid / scale
                orig_box_w = abs_w / scale
                orig_box_h = abs_h / scale
                
                xmin = orig_x_mid - orig_box_w / 2
                ymin = orig_y_mid - orig_box_h / 2
                xmax = orig_x_mid + orig_box_w / 2
                ymax = orig_y_mid + orig_box_h / 2
                
                xmin = max(0.0, float(xmin))
                ymin = max(0.0, float(ymin))
                xmax = min(float(orig_w), float(xmax))
                ymax = min(float(orig_h), float(ymax))
                
                img_boxes.append({
                    "class": config.CLASSES[class_id],
                    "confidence": round(confidence, 4),
                    "bbox": [xmin, ymin, xmax, ymax]
                })
                
        predictions_list.append({
            "image_id": img_name,
            "boxes": img_boxes
        })
        
    with open(args.output, 'w') as f:
        json.dump(predictions_list, f, indent=2)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images")
    parser.add_argument("--output", type=str, required=True, help="Path to output predictions.json")
    args = parser.parse_args()

    main(args)
