import torch
import torch.nn as nn

from utils.boxes import intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_class = 1
        self.lambda_noobj = 5
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        obj = target[..., 4] == 1
        noobj = target[..., 4] == 0

        no_object_loss = self.bce(
            predictions[..., 4:5][noobj], target[..., 4:5][noobj]
        )

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        if obj.sum() == 0:
            return self.lambda_noobj * no_object_loss

        pred_box = predictions[..., 0:4].clone()
        pred_box[..., 0:2] = self.sigmoid(pred_box[..., 0:2])
        pred_box[..., 2:4] = torch.exp(pred_box[..., 2:4]) * anchors

        ious = intersection_over_union(pred_box[obj], target[..., 0:4][obj]).detach()
        object_loss = self.bce(
            predictions[..., 4:5][obj],
            ious * target[..., 4:5][obj],
        )

        ciou = intersection_over_union(
            pred_box[obj], target[..., 0:4][obj], return_ciou=True
        )
        box_loss = torch.mean(1 - ciou)

        class_loss = self.entropy(
            predictions[..., 5:][obj], target[..., 5:][obj].argmax(-1)
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
