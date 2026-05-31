import torch
import random
from torchvision import transforms
from PIL import Image, ImageOps

class YoloTransforms:
    def __init__(self, image_size=416, is_train=True):
        self.image_size = image_size
        self.is_train = is_train

    def __call__(self, image, bboxes):
        """
        image: PIL Image
        bboxes: list of [xmin, ymin, xmax, ymax, class_id]
        """
        w, h = image.size
        # Letterbox resize (thêm viền xám để giữ tỷ lệ khung hình)
        scale = min(self.image_size / w, self.image_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Pad to image_size
        pad_w = (self.image_size - new_w) // 2
        pad_h = (self.image_size - new_h) // 2
        
        padding = (pad_w, pad_h, self.image_size - new_w - pad_w, self.image_size - new_h - pad_h)
        image = ImageOps.expand(image, padding, fill=(128, 128, 128))
        
        new_bboxes = []
        for box in bboxes:
            xmin = box[0] * scale + pad_w
            ymin = box[1] * scale + pad_h
            xmax = box[2] * scale + pad_w
            ymax = box[3] * scale + pad_h
            new_bboxes.append([xmin, ymin, xmax, ymax, box[4]])
            
        # Data Augmentation (Chỉ áp dụng khi train)
        if self.is_train:
            # Lật ngang ngẫu nhiên
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                for i in range(len(new_bboxes)):
                    # Lật tọa độ x
                    xmin, xmax = new_bboxes[i][0], new_bboxes[i][2]
                    new_bboxes[i][0] = self.image_size - xmax
                    new_bboxes[i][2] = self.image_size - xmin
        
        # Chuyển thành Tensor và áp dụng Color Jitter
        image = transforms.ToTensor()(image)
        if self.is_train:
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            image = color_jitter(image)
            
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
            
        return image, torch.tensor(new_bboxes)
