import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import sys

# Thêm đường dẫn gốc để import từ thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.boxes import iou_width_height

class YOLODataset(Dataset):
    def __init__(self, annotation_file, img_dir, anchors, image_size=416, S=[13, 26, 52], num_classes=5, transform=None):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        self.img_dir = img_dir
        self.transform = transform
        self.image_size = image_size
        self.S = S
        
        # anchors parameter is passed from config.ANCHORS
        # shape: (3 scales, 3 anchors per scale, 2 values w/h)
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # [9, 2]
        self.num_anchors_per_scale = self.anchors.shape[0] // 3 # 3
        
        self.num_classes = num_classes
        self.ignore_iou_thresh = config.IGNORE_IOU_THRESH
        
        self.images = data['images']
        self.annotations = data['annotations']
        
        self.img_to_anns = {img['id']: [] for img in self.images}
        for ann in self.annotations:
            self.img_to_anns[ann['image_id']].append(ann)
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        img_info = self.images[index]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'].split('/')[-1])
        
        image = Image.open(img_path).convert("RGB")
        
        bboxes = []
        for ann in self.img_to_anns[img_id]:
            class_id = config.CLASSES.index(ann['class'])
            bbox = ann['bbox'] # [xmin, ymin, xmax, ymax] absolute coords
            bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], class_id])
            
        if self.transform:
            image, bboxes = self.transform(image, bboxes)
            
        # 3 targets (1 cho mỗi scale prediction)
        # target_shape = [num_anchors, S, S, 6] -> 6: [x, y, w, h, obj_conf, class_id_one_hot...]
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 5 + self.num_classes)) for s in self.S]
        
        for box in bboxes:
            if len(box) == 0: continue
            # Convert xmin, ymin, xmax, ymax sang tọa độ tâm (x,y) và (w,h)
            x = (box[0] + box[2]) / 2.0
            y = (box[1] + box[3]) / 2.0
            w = box[2] - box[0]
            h = box[3] - box[1]
            
            class_id = int(box[4])
            
            # Tính IOU theo width-height để xem anchor nào tốt nhất (0-8)
            iou_anchors = iou_width_height(torch.tensor([w, h]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            
            x_cell, y_cell = x / self.image_size, y / self.image_size
            has_anchor = [False, False, False]
            
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                s = self.S[scale_idx]
                
                i, j = int(s * y_cell), int(s * x_cell)
                if i >= s: i = s - 1
                if j >= s: j = s - 1
                
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 4]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 4] = 1 # obj conf = 1
                    
                    x_cell_grid, y_cell_grid = s * x_cell - j, s * y_cell - i
                    width_cell = (w / self.image_size) * s
                    height_cell = (h / self.image_size) * s
                    
                    targets[scale_idx][anchor_on_scale, i, j, 0:4] = torch.tensor([x_cell_grid, y_cell_grid, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 5 + class_id] = 1 # one-hot class
                    
                    has_anchor[scale_idx] = True
                    
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Ignore prediction (obj = -1) => Loss function bỏ qua ô này để không phạt mạng
                    targets[scale_idx][anchor_on_scale, i, j, 4] = -1 
                    
        return image, tuple(targets)
