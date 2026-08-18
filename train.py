import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SemiconductorDataset
from losses import ssim
from model import RestorationNetV2


DATASET = "semicon_new"
GT_DIR = os.path.join(DATASET, "train", "train", "GT")
NOISY_DIR = os.path.join(DATASET, "train", "train", "NoisyLR")
SPLIT_FILE = "split.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain V2 with augmentation, EMA, and a smoother LR schedule."
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--num-features", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=12)
    parser.add_argument("--ssim-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-path", default="best_model_v2_retrain_improved.pth")
    parser.add_argument("--history-path", default="v2_retrain_improved_history.npz")
    parser.add_argument(
        "--use-validation-for-training",
        action="store_true",
        help="Final retrain mode: train on train+validation and save the last EMA model.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        error = prediction - target
        return torch.mean(torch.sqrt(error * error + self.epsilon * self.epsilon))


def combined_loss(prediction, target, pixel_loss, ssim_weight):
    l_pixel = pixel_loss(prediction, target)
    l_ssim = 1.0 - ssim(prediction, target)
    return l_pixel + ssim_weight * l_ssim, l_pixel, l_ssim


def calculate_batch_psnr(prediction, target):
    prediction = torch.clamp(prediction, 0.0, 1.0)
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item()


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)

        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def validate(model, val_loader, device):
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for noisy, gt in val_loader:
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            prediction = model(noisy)

            batch_size = noisy.size(0)
            total_psnr += calculate_batch_psnr(prediction, gt) * batch_size
            total_ssim += ssim(prediction, gt).item() * batch_size
            count += batch_size

    return total_psnr / count, total_ssim / count


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    with open(SPLIT_FILE, "r") as file:
        split_data = json.load(file)

    train_files = split_data["train"]
    val_files = split_data["validation"]

    if args.use_validation_for_training:
        train_files = train_files + val_files
        val_files = []

    train_dataset = SemiconductorDataset(
        GT_DIR,
        NOISY_DIR,
        train_files,
        augment=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None

    if val_files:
        val_dataset = SemiconductorDataset(
            GT_DIR,
            NOISY_DIR,
            val_files,
            augment=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = RestorationNetV2(
        num_features=args.num_features,
        num_blocks=args.num_blocks,
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_files))
    print("Model parameters:", parameters)
    print("Model parameters (M):", parameters / 1e6)

    pixel_loss = CharbonnierLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    history = {
        "train_loss": [],
        "train_psnr": [],
        "val_psnr": [],
        "val_ssim": [],
        "learning_rate": [],
    }

    best_psnr = -float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_psnr = 0.0
        sample_count = 0

        for noisy, gt in train_loader:
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda",
            ):
                prediction = model(noisy)
                loss, _, _ = combined_loss(
                    prediction,
                    gt,
                    pixel_loss,
                    args.ssim_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            update_ema(model, ema_model, args.ema_decay)

            batch_size = noisy.size(0)
            total_loss += loss.item() * batch_size
            total_psnr += calculate_batch_psnr(prediction.detach(), gt) * batch_size
            sample_count += batch_size

        scheduler.step()

        train_loss = total_loss / sample_count
        train_psnr = total_psnr / sample_count
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_psnr"].append(train_psnr)
        history["learning_rate"].append(current_lr)

        if val_loader is not None:
            val_psnr, val_ssim = validate(ema_model, val_loader, device)
            history["val_psnr"].append(val_psnr)
            history["val_ssim"].append(val_ssim)

            print(
                f"Epoch [{epoch:03d}/{args.epochs}] "
                f"Loss: {train_loss:.6f} "
                f"Train PSNR: {train_psnr:.4f} dB "
                f"Val PSNR: {val_psnr:.4f} dB "
                f"Val SSIM: {val_ssim:.6f} "
                f"LR: {current_lr:.2e}"
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": ema_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "args": vars(args),
                    },
                    args.model_path,
                )

                print(f"  Best EMA model saved ({val_psnr:.4f} dB)")

        else:
            print(
                f"Epoch [{epoch:03d}/{args.epochs}] "
                f"Loss: {train_loss:.6f} "
                f"Train PSNR: {train_psnr:.4f} dB "
                f"LR: {current_lr:.2e}"
            )

    if val_loader is None:
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            args.model_path,
        )
        print("Final EMA model saved:", args.model_path)

    np.savez(
        args.history_path,
        train_loss=np.array(history["train_loss"]),
        train_psnr=np.array(history["train_psnr"]),
        val_psnr=np.array(history["val_psnr"]),
        val_ssim=np.array(history["val_ssim"]),
        learning_rate=np.array(history["learning_rate"]),
    )

    print("Training history saved:", args.history_path)


if __name__ == "__main__":
    main()
