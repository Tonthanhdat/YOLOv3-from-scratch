import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import json

import config
from models.resnet_fpn_detector import ResNetFPNDetector
from utils.dataset import YOLODataset
from utils.transforms import YoloTransforms
from models.loss import YOLOLoss

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.amp.autocast(device_type=config.DEVICE) if config.DEVICE == 'cuda' else torch.autocast(device_type='cpu', enabled=False):
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        
        if config.DEVICE == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # Cần unscale trước khi clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        loop.set_postfix(loss=sum(losses)/len(losses))

def set_freeze_state(model, epoch):
    # Luôn freeze stem, layer1, layer2
    for param in model.stem.parameters(): param.requires_grad = False
    for param in model.layer1.parameters(): param.requires_grad = False
    for param in model.layer2.parameters(): param.requires_grad = False

    if epoch < 5:
        # Freeze nốt layer3 và layer4 trong 5 epoch đầu
        for param in model.layer3.parameters(): param.requires_grad = False
        for param in model.layer4.parameters(): param.requires_grad = False
    else:
        # Unfreeze layer3 and layer4
        for param in model.layer3.parameters(): param.requires_grad = True
        for param in model.layer4.parameters(): param.requires_grad = True

def main(args):
    print(f"Training on device: {config.DEVICE}")
    model = ResNetFPNDetector(num_classes=config.NUM_CLASSES, pretrained=True).to(config.DEVICE)
    
    # Param groups cho optimizer
    optimizer = optim.AdamW([
        {"params": model.stem.parameters(), "lr": 1e-5},
        {"params": model.layer1.parameters(), "lr": 1e-5},
        {"params": model.layer2.parameters(), "lr": 1e-5},
        {"params": model.layer3.parameters(), "lr": 1e-5},
        {"params": model.layer4.parameters(), "lr": 1e-5},
        
        {"params": model.lat_c5.parameters(), "lr": 1e-4},
        {"params": model.lat_c4.parameters(), "lr": 1e-4},
        {"params": model.lat_c3.parameters(), "lr": 1e-4},
        {"params": model.smooth_p4.parameters(), "lr": 1e-4},
        {"params": model.smooth_p3.parameters(), "lr": 1e-4},
        
        {"params": model.head_p5.parameters(), "lr": 1e-4},
        {"params": model.head_p4.parameters(), "lr": 1e-4},
        {"params": model.head_p3.parameters(), "lr": 1e-4},
    ], weight_decay=config.WEIGHT_DECAY)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_map = 0.0
    start_epoch = 0

    checkpoint_file = args.resume if args.resume else os.path.join(args.checkpoint_dir, "last.pth")
    if os.path.exists(checkpoint_file):
        print(f"=> Tìm thấy checkpoint: {checkpoint_file}. Đang tải để tiếp tục huấn luyện...")
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "best_map" in checkpoint:
            best_map = checkpoint["best_map"]
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
    else:
        print(f"=> Không tìm thấy checkpoint để tiếp tục. Sẽ huấn luyện từ đầu (Train from scratch).")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler() if config.DEVICE == 'cuda' else None

    train_transform = YoloTransforms(image_size=config.IMAGE_SIZE, is_train=True)
    train_dataset = YOLODataset(
        annotation_file=args.train_data,
        img_dir=args.image_dir,
        anchors=config.ANCHORS,
        transform=train_transform,
        S=config.S,
        image_size=config.IMAGE_SIZE
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    scaled_anchors = (
        (torch.tensor(config.ANCHORS) / config.IMAGE_SIZE)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} (LR: {optimizer.param_groups[-1]['lr']:.6f})")
        
        # Áp dụng chiến thuật đóng băng (Freeze/Unfreeze)
        set_freeze_state(model, epoch)
        
        model.train()
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        
        # Luôn lưu checkpoint cuối cùng
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_map": best_map,
            "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pth"))
        
        # Chấm điểm mAP mỗi 5 epoch
        if (epoch + 1) % 5 == 0 or epoch == config.NUM_EPOCHS - 1:
            print("Đang đánh giá mAP trên tập Validation...")
            model.eval()
            
            # Lưu tạm trọng số để predict
            temp_weights = os.path.join(args.checkpoint_dir, "temp_eval.pth")
            torch.save(checkpoint, temp_weights)
            
            # 1. Chạy Predict.py
            pred_json = "temp_predictions.json"
            os.system(f"python predict.py --image_dir {args.val_image_dir} --output {pred_json} --weights {temp_weights}")
            
            # 2. Chạy Evaluate_predictions.py
            score_json = "temp_score.json"
            os.system(f"python public/tools/evaluate_predictions.py --ground_truth {args.val_data} --predictions {pred_json} --output {score_json}")
            
            # 3. Đọc kết quả
            if os.path.exists(score_json):
                with open(score_json, "r") as f:
                    score_data = json.load(f)
                    map50 = score_data.get("mAP@0.5", 0.0)
                
                print(f"-> Validation mAP@0.5: {map50:.4f}")
                
                if map50 > best_map:
                    best_map = map50
                    torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pth"))
                    print(f"*** Mới lưu Best Model (mAP@0.5 tăng lên {best_map:.4f}) ***")
            else:
                print("Lỗi: Không đánh giá được mAP.")
                
        # Cập nhật LR Scheduler
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to train JSON")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to train images")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Path to val images")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to save models")
    parser.add_argument("--resume", type=str, default="", help="Path to specific checkpoint to resume from (e.g. models/best.pth)")
    args = parser.parse_args()

    main(args)
