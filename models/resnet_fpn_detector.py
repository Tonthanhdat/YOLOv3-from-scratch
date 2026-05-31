import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1),
            # Mỗi anchor sẽ đoán: 1(obj) + 4(bbox) + num_classes
            nn.Conv2d(in_channels * 2, 3 * (num_classes + 5), kernel_size=1, bias=True),
        )
        self.num_classes = num_classes

    def forward(self, x):
        # x: [batch_size, channels, S, S]
        # Đầu ra: [batch_size, 3, S, S, 5+num_classes]
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class ResNetFPNDetector(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        # 1. Load Pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        
        # Trích xuất các stage
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1 # /4
        self.layer2 = backbone.layer2 # /8  (C3) -> 52x52
        self.layer3 = backbone.layer3 # /16 (C4) -> 26x26
        self.layer4 = backbone.layer4 # /32 (C5) -> 13x13
        
        # 2. Xây dựng FPN (Feature Pyramid Network)
        # Các kênh output của ResNet50: C3=512, C4=1024, C5=2048
        fpn_out = 256
        
        # Lateral convolutions
        self.lat_c5 = nn.Conv2d(2048, fpn_out, kernel_size=1)
        self.lat_c4 = nn.Conv2d(1024, fpn_out, kernel_size=1)
        self.lat_c3 = nn.Conv2d(512, fpn_out, kernel_size=1)
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Smoothing convolutions sau khi cộng
        self.smooth_p4 = nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)
        
        # 3. Yolo Detection Heads
        self.head_p5 = ScalePrediction(fpn_out, num_classes) # Cho Scale 1 (13x13)
        self.head_p4 = ScalePrediction(fpn_out, num_classes) # Cho Scale 2 (26x26)
        self.head_p3 = ScalePrediction(fpn_out, num_classes) # Cho Scale 3 (52x52)
        
    def forward(self, x):
        # Bottom-up (ResNet)
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Top-down (FPN)
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + self.upsample(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.lat_c3(c3) + self.upsample(p4)
        p3 = self.smooth_p3(p3)
        
        # YOLO Predictions (Theo thứ tự scale từ nhỏ đến lớn: 13x13, 26x26, 52x52)
        out_scale1 = self.head_p5(p5)
        out_scale2 = self.head_p4(p4)
        out_scale3 = self.head_p3(p3)
        
        return [out_scale1, out_scale2, out_scale3]

if __name__ == "__main__":
    num_classes = 5
    IMAGE_SIZE = 416
    model = ResNetFPNDetector(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    outputs = model(x)
    print("Test forwarding:")
    assert outputs[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert outputs[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert outputs[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success! Output shapes matched expected.")
