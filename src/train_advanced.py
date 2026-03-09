"""
Advanced training script for mushroom classification.
Inspired by Danish Fungi Dataset training approach.

Features:
- FocalLoss / SeeSawLoss for class imbalance
- Mixup / CutMix augmentation
- Gradient clipping
- EMA (Exponential Moving Average)
- Gradient accumulation
- Strong augmentations
"""

import os
import sys
import argparse
import math
from copy import deepcopy
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
try:
    from torch.amp import GradScaler
    from torch.amp import autocast as _autocast
    def amp_autocast(device_type):
        return _autocast(device_type)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast as _autocast
    def amp_autocast(device_type):
        return _autocast()
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_setup import create_dataloaders
from src.models.vit import create_vit_model, create_vit_small
from src.losses import FocalLoss, SeeSawLoss, LabelSmoothingCrossEntropy


class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Keeps a running average of the model parameters for better generalization.
    """
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)


def mixup_data(x, y, alpha=0.8):
    """
    Mixup: Creates mixed inputs and targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: Cuts and pastes patches among training images.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)

    # Get bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda based on actual area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for mixup/cutmix.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_class_counts(dataloader):
    """
    Count samples per class for SeeSawLoss.
    """
    counts = Counter()
    for batch in dataloader:
        labels = batch["label"].numpy()
        counts.update(labels)

    num_classes = max(counts.keys()) + 1
    class_counts = [counts.get(i, 1) for i in range(num_classes)]
    return class_counts


def train_one_epoch(
    model, train_loader, criterion, optimizer, device,
    scaler=None, ema=None,
    mixup_alpha=0.0, cutmix_alpha=0.0, mixup_prob=0.5,
    clip_grad=None, accumulation_steps=1
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None and device.type == 'cuda'
    use_mixup = mixup_alpha > 0 or cutmix_alpha > 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Apply Mixup or CutMix
        if use_mixup and np.random.random() < mixup_prob:
            if cutmix_alpha > 0 and np.random.random() > 0.5:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha)
            else:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            mixed = True
        else:
            mixed = False

        if use_amp:
            with amp_autocast('cuda'):
                outputs = model(images)
                if mixed:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update(model)
        else:
            outputs = model(images)
            if mixed:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()

                if ema is not None:
                    ema.update(model)

        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        if not mixed:
            correct += predicted.eq(labels).sum().item()
        else:
            # For mixed samples, use primary label for tracking
            correct += predicted.eq(labels_a).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
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
            with amp_autocast('cuda'):
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
    print(f"Training configuration:")
    print(f"  - Loss: {args.loss}")
    print(f"  - Mixup alpha: {args.mixup_alpha}")
    print(f"  - CutMix alpha: {args.cutmix_alpha}")
    print(f"  - EMA: {args.ema}")
    print(f"  - Gradient clipping: {args.clip_grad}")
    print(f"  - Accumulation steps: {args.accumulation_steps}")
    print(f"  - Augmentation: {args.augmentation}")

    # Mixed precision setup
    use_amp = args.amp and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print("  - Mixed precision (AMP) enabled")

    train_loader, val_loader, num_classes = create_dataloaders(
        csv_path=args.csv_path,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        mode='single',
        num_workers=args.num_workers,
        augmentation_strength=args.augmentation
    )
    print(f"\nNumber of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model selection
    if args.model == 'vit_small':
        print("Using ViT-Small")
        model = create_vit_small(num_classes=num_classes, pretrained=args.pretrained)
    else:
        print("Using ViT-Base")
        model = create_vit_model(
            num_classes=num_classes,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone
        )
    model = model.to(device)

    # torch.compile
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()")
        model = torch.compile(model)

    # EMA setup
    ema = None
    if args.ema:
        ema = ModelEMA(model, decay=args.ema_decay, device=device)
        print(f"EMA enabled with decay={args.ema_decay}")

    # Loss function selection
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=1.0, gamma=args.focal_gamma)
        print(f"Using FocalLoss (gamma={args.focal_gamma})")
    elif args.loss == 'seesaw':
        print("Computing class counts for SeeSawLoss...")
        class_counts = get_class_counts(train_loader)
        criterion = SeeSawLoss(class_counts)
        print("Using SeeSawLoss")
    elif args.loss == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        print(f"Using LabelSmoothingCrossEntropy (smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == 'onecycle':
        steps_per_epoch = len(train_loader) // args.accumulation_steps
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1
        )
    else:
        scheduler = None

    best_val_acc = 0.0
    start_epoch = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        saved_model_type = checkpoint.get('model_type', 'vit_base')
        if saved_model_type != args.model:
            print(f"WARNING: Checkpoint is from {saved_model_type}, but you're using {args.model}")
            print("Starting fresh instead of resuming")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if ema and 'ema_state_dict' in checkpoint:
                ema.load_state_dict(checkpoint['ema_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, ema=ema,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_prob=args.mixup_prob,
            clip_grad=args.clip_grad,
            accumulation_steps=args.accumulation_steps
        )

        # Validate with EMA model if available
        if ema is not None:
            val_loss, val_acc = validate(ema.ema, val_loader, criterion, device, use_amp)
            print(f"(EMA) Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            # Also validate regular model
            val_loss_reg, val_acc_reg = validate(model, val_loader, criterion, device, use_amp)
            print(f"(Reg) Val Loss: {val_loss_reg:.4f}, Val Acc: {val_acc_reg:.2f}%")
        else:
            val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)

        if scheduler:
            if args.scheduler != 'onecycle':
                scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'model_type': args.model,
            }
            if ema:
                save_dict['ema_state_dict'] = ema.state_dict()
            if scheduler:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(save_dict, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model with val acc: {val_acc:.2f}%")

        # Save latest checkpoint
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'num_classes': num_classes,
            'model_type': args.model,
        }
        if ema:
            save_dict['ema_state_dict'] = ema.state_dict()
        if scheduler:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(save_dict, os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth'))

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced ViT Training for Mushroom Classification")

    # Data arguments
    parser.add_argument("--csv_path", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of images")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model arguments
    parser.add_argument("--model", type=str, default="vit_base", choices=["vit_base", "vit_small"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--freeze_backbone", action="store_true", default=False)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "onecycle", "none"])

    # Loss function
    parser.add_argument("--loss", type=str, default="focal",
                        choices=["ce", "focal", "seesaw", "label_smoothing"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # Augmentation
    parser.add_argument("--augmentation", type=str, default="strong",
                        choices=["light", "standard", "strong"])
    parser.add_argument("--mixup_alpha", type=float, default=0.8, help="Mixup alpha (0 to disable)")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0, help="CutMix alpha (0 to disable)")
    parser.add_argument("--mixup_prob", type=float, default=0.5, help="Probability of applying mixup/cutmix")

    # Regularization
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--ema", action="store_true", default=True, help="Use EMA")
    parser.add_argument("--no_ema", action="store_false", dest="ema")
    parser.add_argument("--ema_decay", type=float, default=0.9999)

    # Efficiency
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--compile", action="store_true", default=False)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    main(args)