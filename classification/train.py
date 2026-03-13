import argparse
import datetime
import glob
import json
import os
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
import torch.optim as optim
from adan_pytorch import Adan
# from resnet50 import resnet50
from ConvNext import convnext_base
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from train_utils import (create_lr_scheduler, evaluate, get_params_groups,
                         train_one_epoch)


def find_latest_checkpoint():
    """Returns latest checkpoint file if exists, else None"""
    checkpoints = glob.glob("checkpoint.pt")
    if not checkpoints:
        return None
    latest = max(checkpoints, key=os.path.getctime)
    return latest


def main(args):
    # Set device
    device = torch.device(torch.cuda.current_device()
                          if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # TensorBoard writer
    tb_writer = SummaryWriter()

    # Data transforms
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 360)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # Load datasets
    train_dataset = datasets.ImageFolder(root=join(args.data_path, "train"),
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=join(args.data_path, "test"),
                                       transform=data_transform["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0)

    # Initialize model
    model = convnext_base(num_classes=args.num_classes)
    model.to(device)

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=10)

    # Best metrics
    best_acc = 0.0
    best_train_acc = 0.0
    best_kappa = 0.0
    start_epoch = 0

    # AUTO-RESUME: load latest checkpoint if exists
    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
        print(f"Loading checkpoint {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        best_train_acc = ckpt.get("best_train_acc", 0.0)
        best_kappa = ckpt.get("best_kappa", 0.0)
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc, train_kappa1, train_truee, train_predd = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler
        )
        print(f"Epoch {epoch} | train_kappa = {train_kappa1:.4f}")

        # Train confusion matrix
        if train_acc > best_train_acc:
            df = pd.DataFrame({"True": train_truee, "Pred": train_predd})
            confmtpd = pd.crosstab(df['True'], df['Pred'], dropna=False)
            cfmfig = plt.figure()
            sn.heatmap(confmtpd, annot=True, cmap='Greens', fmt='d')
            best_train_acc = train_acc

        # Validate
        val_loss, val_acc, val_kappa1, val_truee, val_predd = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )
        print(f"Epoch {epoch} | val_kappa = {val_kappa1:.4f}")

        # Log metrics to JSON (append mode)
        log_data = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_kappa": float(train_kappa1),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_kappa": float(val_kappa1),
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open("log.json", "w") as f:
            f.write(json.dumps(log_data) + "\n")

        # TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc",
                "learning_rate", "train_kappa", "val_kappa"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], train_kappa1, epoch)
        tb_writer.add_scalar(tags[6], val_kappa1, epoch)

        # SAVE checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_acc": best_acc,
            "best_train_acc": best_train_acc,
            "best_kappa": best_kappa
        }, f"checkpoint.pt")

        # Save best model
        if val_acc > best_acc:
            PATH = join(args.data_path, "train")
            os.makedirs(PATH, exist_ok=True)
            torch.save(model.state_dict(), join(PATH, "cancer_resnet50.pth"))
            best_acc = val_acc

        # Update best kappa
        if val_kappa1 > best_kappa:
            best_kappa = val_kappa1

        print(
            f"best_train_acc = {best_train_acc:.4f} | best_val_acc = {best_acc:.4f} | best_val_kappa = {best_kappa:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=0.05)
    parser.add_argument('--data-path', type=str,
                        default=r"./classification/data/8422229/BMP_classification/BMP_classification/")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)