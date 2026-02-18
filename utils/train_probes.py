import copy
import random
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import torch.nn as nn


def get_current_order(epoch, num_epochs, min_order, max_order):
    """Helper for progressive training mode"""
    progress = epoch / num_epochs
    order = int(min_order + progress * (max_order - min_order))
    return min(order, max_order)


def binary_accuracy(preds, labels):
    """Compute binary classification accuracy from logits."""
    probs = torch.sigmoid(preds)
    preds_bin = (probs > 0.5).long()
    return (preds_bin == labels.long()).float().mean().item()


def train(
    classifier,
    train_loader,
    val_loader,
    num_epochs,
    train_mode,
    min_order=1,
    max_order=5,
    metric=binary_accuracy,
    num_seeds=1,
    lr=1e-4,
    weight_decay=0.1,
    d_prob=0.0,
    device=torch.device("cuda"),
    use_mixed_precision=True,
    early_stopping_patience=None,
    metric_min_delta=1e-4,
):
    """
    Train a probe or classifier with flexible modes:
      - train_all: train all polynomial terms together
      - progressive: gradually unfreeze higher-order terms
      - random: sample random order per epoch
    Returns: (cls_all, num_params, last_epoch, final_val_acc)
    """

    cls_all = []
    num_params_all = []
    best_val_acc_all = []
    best_epoch_all = []

    if train_mode == "progressive":
        num_epochs = num_epochs * (max_order - min_order + 1)

    for seed in tqdm(range(num_seeds), desc="Training seeds"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cls = classifier(train_mode=train_mode, d_prob=d_prob).to(device)
        optimizer = torch.optim.AdamW(cls.parameters(), lr=1e-4, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        criterion = torch.nn.BCEWithLogitsLoss()
        scaler = GradScaler(enabled=use_mixed_precision)

        num_params = sum(p.numel() for p in cls.parameters())
        if seed == 0:
            print(f"Num params: {num_params:,}")
            print(f"Train mode: {train_mode}\n")

        prev_order = min_order
        best_val_acc = 0.0
        best_epoch = 0
        epochs_without_improve = 0
        best_state_dict = copy.deepcopy(cls.state_dict())

        for epoch in range(1, 100 + 1):
            cls.train()
            running_loss = 0.0
            train_preds, train_labels = [], []

            for xb_cpu, yb_cpu in train_loader:
                xb = xb_cpu.to(device, non_blocking=True)
                yb = yb_cpu.to(device, non_blocking=True)

                # ---- choose current order ----
                if train_mode == "progressive":
                    current_order = get_current_order(epoch, num_epochs, min_order, max_order)
                    if current_order > prev_order:
                        # freeze previous terms (lower orders)
                        for name, param in cls.named_parameters():
                            for o in range(prev_order):
                                if f"HO.{o}" in name or f"lam.{o}" in name:
                                    param.requires_grad = False
                        prev_order = current_order
                elif train_mode == "random":
                    current_order = random.randint(min_order, max_order)
                elif train_mode in ["train_all", "max"]:
                    current_order = max_order
                else:
                    raise ValueError(f"Unknown train mode: {train_mode}")

                # ---- forward + backward ----
                optimizer.zero_grad()
                with autocast(device_type="cuda", enabled=use_mixed_precision):
                    logits = cls(xb, test_time_order=current_order)
                    if isinstance(logits, list):
                        logits = logits[-1]
                    logits = logits.view(-1)
                    yb = yb.view(-1)
                    loss = criterion(logits, yb)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(cls.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.detach().item() * xb.size(0)
                train_preds.append(logits.detach().cpu())
                train_labels.append(yb.cpu())

            # ---- epoch-level training metrics ----
            train_preds = torch.cat(train_preds)
            train_labels = torch.cat(train_labels)
            train_acc = metric(train_preds, train_labels)
            epoch_loss = running_loss / len(train_loader.dataset)

            # ---- validation ----
            cls.eval()
            val_preds, val_labels = [], []
            running_val_loss = 0.0

            with torch.no_grad():
                for xb_cpu, yb_cpu in val_loader:
                    xb = xb_cpu.to(device, non_blocking=True)
                    yb = yb_cpu.to(device, non_blocking=True)
                    logits = cls(xb, test_time_order=max_order)
                    if isinstance(logits, list):
                        logits = logits[-1]

                    # Ensure shapes match
                    logits = logits.view(-1)
                    yb = yb.view(-1)

                    loss_val = criterion(logits, yb)
                    running_val_loss += loss_val.item() * xb.size(0)

                    val_preds.append(logits.detach().cpu())
                    val_labels.append(yb.cpu())

            val_preds = torch.cat(val_preds)
            val_labels = torch.cat(val_labels)
            val_acc = metric(val_preds, val_labels)
            epoch_loss_val = running_val_loss / len(val_loader.dataset)
            scheduler.step(epoch_loss_val)

            current_lr = optimizer.param_groups[0]["lr"]
             

            if val_acc > best_val_acc + 5e-4:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_without_improve = 0
                best_state_dict = copy.deepcopy(cls.state_dict())
            else:
                epochs_without_improve += 1

            if (
                early_stopping_patience is not None
                and epochs_without_improve >= early_stopping_patience
            ):
                break

        cls.load_state_dict(best_state_dict)
        cls_all.append(cls)
        num_params_all.append(num_params)
        best_val_acc_all.append(best_val_acc)
        best_epoch_all.append(best_epoch if best_epoch != 0 else epoch)

    return cls_all, num_params_all, best_epoch_all, best_val_acc_all
