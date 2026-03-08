import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_setup import create_dataloaders
from src.models.vit import create_vit_model, create_vit_small


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None and device.type == 'cuda'

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(val_loader, desc="Validating"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if use_amp and device.type == 'cuda':
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Mixed precision setup
    use_amp = args.amp and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled")

    train_loader, val_loader, num_classes = create_dataloaders(
        csv_path=args.csv_path,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        mode='single',
        num_workers=args.num_workers
    )
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model selection: ViT-Small (faster) or ViT-Base
    if args.model == 'vit_small':
        print("Using ViT-Small (faster)")
        model = create_vit_small(num_classes=num_classes, pretrained=args.pretrained)
    else:
        print("Using ViT-Base")
        model = create_vit_model(
            num_classes=num_classes,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone
        )
    model = model.to(device)

    # torch.compile for PyTorch 2.0+ (significant speedup)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()")
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    start_epoch = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'model_type': args.model,
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model with val acc: {val_acc:.2f}%")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'num_classes': num_classes,
            'model_type': args.model,
        }, os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth'))

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT for Mushroom Classification")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of images")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    # Speed optimizations
    parser.add_argument("--model", type=str, default="vit_base", choices=["vit_base", "vit_small"],
                        help="Model variant: vit_base (slower, more accurate) or vit_small (faster)")
    parser.add_argument("--amp", action="store_true", default=True, help="Use mixed precision training")
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--compile", action="store_true", default=False, help="Use torch.compile (PyTorch 2.0+)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    main(args)