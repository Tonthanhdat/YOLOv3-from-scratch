import torch
import torch.nn as nn
from utils.boxes import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Trọng số của các thành phần loss theo YOLOv3 paper
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        """
        predictions: [batch_size, 3, S, S, 5+num_classes] (chưa qua activation function nào)
        target: [batch_size, 3, S, S, 5+num_classes]
        anchors: tensor [3, 2] (width, height tương đối so với toàn bộ ảnh)
        """
        # Xác định obj và noobj (theo giá trị obj_conf trong target)
        obj = target[..., 4] == 1
        noobj = target[..., 4] == 0
        
        # 1. No Object Loss
        no_object_loss = self.bce(
            predictions[..., 4:5][noobj], target[..., 4:5][noobj]
        )

        # 2. Object Loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 0:2]), torch.exp(predictions[..., 2:4]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 0:4][obj]).detach()
        # Mục tiêu BCE ở đây sử dụng ious làm trọng số ground truth
        object_loss = self.bce((predictions[..., 4:5][obj]), (ious * target[..., 4:5][obj]))

        # 3. Box Coordinates Loss
        predictions[..., 0:2] = self.sigmoid(predictions[..., 0:2])
        # scale w,h targets
        target[..., 2:4] = torch.log(1e-16 + target[..., 2:4] / anchors)
        box_loss = self.mse(predictions[..., 0:4][obj], target[..., 0:4][obj])

        # 4. Class Loss
        class_loss = self.entropy(
            predictions[..., 5:][obj], target[..., 5:][obj].argmax(-1)
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
