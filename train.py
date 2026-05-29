import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

import config
from models.yolov3 import YOLOv3
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
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loop.set_postfix(loss=sum(losses)/len(losses))

def main(args):
    print(f"Training on device: {config.DEVICE}")
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler() if config.DEVICE == 'cuda' else None

    train_transform = YoloTransforms(image_size=config.IMAGE_SIZE, is_train=True)
    val_transform = YoloTransforms(image_size=config.IMAGE_SIZE, is_train=False)

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

    val_dataset = YOLODataset(
        annotation_file=args.val_data,
        img_dir=args.val_image_dir,
        anchors=config.ANCHORS,
        transform=val_transform,
        S=config.S,
        image_size=config.IMAGE_SIZE
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        model.train()
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        
        # Validation loop
        model.eval()
        val_losses = []
        print("Evaluating validation loss...")
        with torch.no_grad():
            for x, y in val_loader:
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
                val_losses.append(loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"-> Validation Loss: {avg_val_loss:.4f}")
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        
        # Lưu mô hình tốt nhất dựa trên val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"*** Mới lưu Best Model (Val Loss giảm xuống {best_val_loss:.4f}) ***")
            
        # Luôn lưu checkpoint cuối cùng
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to train JSON")
    parser.add_argument("--val_data", type=str, required=True, help="Path to val JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to train images")
    parser.add_argument("--val_image_dir", type=str, required=True, help="Path to val images")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to save models")
    args = parser.parse_args()

    main(args)
