import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn_act else nn.Identity()
        self.leaky = nn.LeakyReLU(0.1) if bn_act else nn.Identity()

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            )
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels * 2, kernel_size=3, padding=1),
            # Mỗi anchor sẽ đoán: 1(obj) + 4(bbox) + num_classes
            CNNBlock(in_channels * 2, 3 * (num_classes + 5), bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        # x: [batch_size, channels, S, S]
        # Đầu ra: [batch_size, 3, S, S, 5+num_classes]
        # Đầu ra gốc của pred(x): [batch_size, 3*(5+num_classes), S, S]
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

# Cấu trúc của mạng Darknet-53
# List chứa Tuple(channels, kernel_size, stride), "B" cho ResidualBlock, "S" cho ScalePrediction, "U" cho Upsampling
config_darknet = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # Đến đây là backbone xong, Scale 13x13 (Layer 52)
    (512, 1, 1),
    (1024, 3, 1),
    "S", # Bắn ra output cho scale 1 (13x13)
    (256, 1, 1),
    "U", # Upsample
    (256, 1, 1),
    (512, 3, 1),
    "S", # Bắn ra output cho scale 2 (26x26)
    (128, 1, 1),
    "U", # Upsample
    (128, 1, 1),
    (256, 3, 1),
    "S", # Bắn ra output cho scale 3 (52x52)
]

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # Lưu 3 outputs
        route_connections = [] # Lưu các route layer (skip connections từ backbone sang FPN neck)

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            # Các bước lấy route connections ở sau residual block 8 lần (scale 52x52, 26x26)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                # Concat với route connections
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config_darknet:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
                    
        return layers

if __name__ == "__main__":
    num_classes = 5
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    print("Test forwarding:")
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success! Output shapes matched expected.")
