#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多分类图像训练（类别数 >= 3）。

与 main_train.py 共用数据约定：imagepath/CSV/<fold>.csv，列 Path、label、subtype。
指标：Loss、Accuracy、Balanced Accuracy、F1(macro/weighted)、AUROC(OVR, macro)、
每轮验证集混淆矩阵与 classification_report。

示例：
  python Resent/main_train_multiclass.py \\
      --modelname Thyroid_4class \\
      --num_classes 4 \\
      --imagepath ./dataset \\
      --train_data Thyroid_4class_train \\
      --valid_data Thyroid_4class_valid \\
      --test_data Thyroid_4class_test \\
      --batch_size 32 --num_epochs 100 --learning_rate 5e-4
"""

import argparse
import os
import copy
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
    roc_auc_score,
)
from torchvision import models

import scripts.dataset as DATA
import scripts.config as config
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_model(architecture: str, num_classes: int) -> nn.Module:
    architecture = architecture.lower().strip()
    if architecture == "resnet":
        net = models.resnet18(pretrained=True)
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, num_classes)
        return net
    raise ValueError(f"当前仅实现 resnet，收到: {architecture}")


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, num_classes: int) -> dict:
    y_true = y_true.astype(np.int64)
    y_pred = y_proba.argmax(axis=1)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auroc_macro_ovr": float("nan"),
    }
    try:
        if num_classes == 2:
            m["auroc_macro_ovr"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            m["auroc_macro_ovr"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        pass
    return m


def class_weights_from_train_csv(imagepath: str, train_fold: str, num_classes: int) -> torch.Tensor:
    csv_path = os.path.join(imagepath, "CSV", f"{train_fold}.csv")
    df = pd.read_csv(csv_path)
    counts = (
        df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    n = len(df)
    c = num_classes
    w = n / (c * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    num_classes,
    train: bool,
    optimizer=None,
):
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    all_y = []
    all_prob = []

    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long().view(-1)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        all_y.append(labels.detach().cpu().numpy())
        all_prob.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())

    y_true = np.concatenate(all_y, axis=0)
    y_proba = np.concatenate(all_prob, axis=0)
    metrics = compute_metrics(y_true, y_proba, num_classes)
    metrics["loss"] = float(np.mean(losses))
    y_pred = y_proba.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    return metrics, cm, y_true, y_pred


def _epoch_banner(epoch: int, num_epochs: int, log_f) -> None:
    """控制台与日志：每个 epoch 顶部醒目标题。"""
    bar = "=" * 72
    block = f"\n{bar}\n  Epoch {epoch + 1} / {num_epochs}\n{bar}\n"
    print(block, end="")
    if log_f:
        log_f.write(block)


def _epoch_spacer_between(log_f) -> None:
    """与上一 epoch 输出隔开（第 2 轮起在 banner 前调用）。"""
    sep = "\n" + "-" * 72 + "\n\n"
    print(sep, end="")
    if log_f:
        log_f.write(sep)


def log_block(fh, title: str, metrics: dict, cm: np.ndarray, report: str = None):
    line = (
        f"  [{title.strip()}]  loss={metrics['loss']:.4f}  acc={metrics['accuracy']:.4f}  "
        f"bal_acc={metrics['balanced_accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}  "
        f"f1_weighted={metrics['f1_weighted']:.4f}  auroc_ovr_macro={metrics['auroc_macro_ovr']}"
    )
    print(line)
    if fh:
        fh.write(line + "\n")
        fh.write("confusion_matrix (rows=true, cols=pred):\n")
        fh.write(np.array2string(cm) + "\n")
        if report:
            fh.write("classification_report:\n" + report + "\n")
        fh.flush()


def main(args):
    if args.num_classes < 3:
        print("提示: num_classes<3 时也可用本脚本，但二分类更常用 main_train.py；继续运行。")

    label_num, _ = config.THYROID()
    Datasets = DATA.Thyroid_Datasets

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_root, f"multiclass_{args.modelname}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"train_{timestamp}.log")
    metrics_csv = os.path.join(save_dir, "metrics_per_epoch.csv")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"start: {datetime.now()}\n")
        f.write(f"modelname={args.modelname} num_classes={args.num_classes}\n")
        f.write(f"train={args.train_data} valid={args.valid_data} test={args.test_data}\n\n")

    print(f"保存目录: {save_dir}")
    print(f"日志: {log_path}")

    transforms = config.Transforms(args.modelname)
    train_set = Datasets(
        path_to_images=args.imagepath,
        fold=args.train_data,
        PRED_LABEL=label_num,
        transform=transforms["train"],
    )
    valid_set = Datasets(
        path_to_images=args.imagepath,
        fold=args.valid_data,
        PRED_LABEL=label_num,
        transform=transforms["valid"],
    )
    test_set = Datasets(
        path_to_images=args.imagepath,
        fold=args.test_data,
        PRED_LABEL=label_num,
        transform=transforms["valid"],
    )

    nw = args.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    eval_bs = args.eval_batch_size
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.architecture, args.num_classes).to(device)

    if args.modelload_path:
        state = torch.load(args.modelload_path, map_location=device)
        model.load_state_dict(state, strict=False)

    if args.auto_class_weights:
        w = class_weights_from_train_csv(args.imagepath, args.train_data, args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
        print("已启用按训练集频次反比 class weights。")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_score = -1.0
    best_wts = copy.deepcopy(model.state_dict())
    metric_key = args.metric_for_best
    rows = []

    with open(log_path, "a", encoding="utf-8") as log_f:
        for epoch in range(args.num_epochs):
            t0 = time.time()
            if epoch > 0:
                _epoch_spacer_between(log_f)
            _epoch_banner(epoch, args.num_epochs, log_f)

            tr_m, tr_cm, _, _ = run_epoch(
                model, train_loader, criterion, device, args.num_classes, True, optimizer
            )
            va_m, va_cm, y_va_true, y_va_pred = run_epoch(
                model, valid_loader, criterion, device, args.num_classes, False, None
            )
            te_m, te_cm, _, _ = run_epoch(
                model, test_loader, criterion, device, args.num_classes, False, None
            )

            report_va = classification_report(
                y_va_true,
                y_va_pred,
                labels=list(range(args.num_classes)),
                zero_division=0,
            )

            log_block(log_f, "train", tr_m, tr_cm)
            log_f.write("\n")
            print()
            log_block(log_f, "valid", va_m, va_cm, report_va)
            log_f.write("\n")
            print()
            log_block(log_f, "test", te_m, te_cm)
            log_f.write("\n")

            row = {
                "epoch": epoch,
                "train_loss": tr_m["loss"],
                "train_acc": tr_m["accuracy"],
                "train_f1_macro": tr_m["f1_macro"],
                "val_loss": va_m["loss"],
                "val_acc": va_m["accuracy"],
                "val_bal_acc": va_m["balanced_accuracy"],
                "val_f1_macro": va_m["f1_macro"],
                "val_auroc": va_m["auroc_macro_ovr"],
                "test_loss": te_m["loss"],
                "test_acc": te_m["accuracy"],
                "test_f1_macro": te_m["f1_macro"],
                "test_auroc": te_m["auroc_macro_ovr"],
            }
            rows.append(row)

            score = va_m.get(metric_key)
            if score is None or np.isnan(score):
                score = va_m["f1_macro"]
            if score > best_score:
                best_score = score
                best_wts = copy.deepcopy(model.state_dict())
                pth = os.path.join(save_dir, f"best_{metric_key}_{best_score:.4f}.pth")
                torch.save(model.state_dict(), pth)
                nb = f"  * new best  {metric_key}={best_score:.4f}  ->  {pth}\n"
                print(nb, end="")
                log_f.write(f"  *** new best ({metric_key}={best_score:.4f}) -> {pth}\n")

            if args.save_all_checkpoints:
                torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch:03d}.pth"))
            np.savetxt(
                os.path.join(save_dir, f"confusion_valid_epoch_{epoch:03d}.csv"),
                va_cm,
                fmt="%d",
                delimiter=",",
            )

            scheduler.step()
            lr_v = optimizer.param_groups[0]["lr"]
            dt = time.time() - t0
            footer = f"  > lr={lr_v:.6f}  wall_time={dt:.1f}s\n"
            print(footer)
            log_f.write(footer)
            log_f.flush()

    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    torch.save(best_wts, os.path.join(save_dir, "best_model_weights.pth"))
    print(f"结束。最佳验证 {metric_key}={best_score:.4f}，权重: best_model_weights.pth")
    print(f"每轮指标: {metrics_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="多分类（>=3 类）ResNet 训练")
    p.add_argument("--modelname", type=str, default="Thyroid_multiclass")
    p.add_argument("--architecture", type=str, default="resnet", choices=["resnet"])
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--imagepath", type=str, default="./dataset/")
    p.add_argument("--train_data", type=str, default="Thyroid_4class_train")
    p.add_argument("--valid_data", type=str, default="Thyroid_4class_valid")
    p.add_argument("--test_data", type=str, default="Thyroid_4class_test")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--lr_step", type=int, default=20)
    p.add_argument("--lr_gamma", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_root", type=str, default="./modelsaved")
    p.add_argument("--modelload_path", type=str, default=None)
    p.add_argument(
        "--metric_for_best",
        type=str,
        default="f1_macro",
        choices=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auroc_macro_ovr"],
        help="以验证集上该指标最大保存 best 权重",
    )
    p.add_argument(
        "--auto_class_weights",
        action="store_true",
        help="按训练集 label 频次设 CrossEntropyLoss 权重（缓解不均衡）",
    )
    p.add_argument(
        "--save_all_checkpoints",
        action="store_true",
        help="每轮保存 epoch_XXX.pth（默认仅保存刷新 best 时的快照与最终 best_model_weights.pth）",
    )
    main(p.parse_args())
