import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_setup import create_dataloaders
from src.models.vit import create_vit_model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
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
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(val_loader, desc="Validating"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

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

    # Create dataloaders
    train_loader, val_loader, num_classes = create_dataloaders(
        csv_path=args.csv_path,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        mode='single'
    )
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = create_vit_model(
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model with val acc: {val_acc:.2f}%")

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'num_classes': num_classes,
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

    args = parser.parse_args()
    main(args)