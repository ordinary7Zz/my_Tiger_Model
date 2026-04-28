#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型测试脚本（二分类，JSON输入版）

与原 test_model.py 的主要区别：
1. 输入由 CSV 改为 JSON（格式同 train_labels.json）
2. 图像路径通过 image_base_path + filename 拼接
3. 标签默认读取 FTCPTC，并默认忽略 -1

JSON 记录格式示例：
[
  {
    "filename": "2016/刘桂英/刘桂英_01_0001_0001.jpg",
    "FTCPTC": 0
  },
  {
    "filename": "2016/张三/张三_01_0001_0001.jpg",
    "FTCPTC": 1
  }
]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


def parse_label_map(label_map_str):
    """
    解析标签映射字符串，格式: "0:0,1:1"
    含义: 原始标签0 -> 评估标签0, 原始标签1 -> 评估标签1
    """
    mapping = {}
    for item in label_map_str.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"label_map 项格式错误: {item}")
        src, dst = parts
        mapping[int(src)] = int(dst)
    if not mapping:
        raise ValueError("label_map 不能为空")
    return mapping


def parse_ignore_labels(ignore_labels_str):
    ignore = set()
    for item in ignore_labels_str.split(","):
        item = item.strip()
        if item:
            ignore.add(int(item))
    return ignore


def load_json_samples(
    json_path,
    image_base_path,
    filename_key="filename",
    label_key="FTCPTC",
    label_map=None,
    ignore_labels=None,
    require_exists=False,
):
    """从 JSON 加载样本，返回 DataFrame(columns=[Path, label])。"""
    if label_map is None:
        label_map = {0: 0, 1: 1}
    if ignore_labels is None:
        ignore_labels = {-1}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是列表")

    rows = []
    stats = {
        "total": 0,
        "missing_keys": 0,
        "ignored": 0,
        "unknown_label": 0,
        "missing_file": 0,
        "accepted": 0,
    }

    for rec in data:
        stats["total"] += 1

        if filename_key not in rec or label_key not in rec:
            stats["missing_keys"] += 1
            continue

        rel_path = str(rec[filename_key]).strip().replace("\\", "/")
        raw_label = rec[label_key]

        if raw_label is None:
            stats["ignored"] += 1
            continue

        try:
            raw_label_int = int(raw_label)
        except (TypeError, ValueError):
            stats["unknown_label"] += 1
            continue

        if raw_label_int in ignore_labels:
            stats["ignored"] += 1
            continue

        if raw_label_int not in label_map:
            stats["unknown_label"] += 1
            continue

        full_path = os.path.join(image_base_path, rel_path)

        if require_exists and not os.path.exists(full_path):
            stats["missing_file"] += 1
            continue

        rows.append({"Path": rel_path, "label": int(label_map[raw_label_int])})

    stats["accepted"] = len(rows)

    if not rows:
        raise ValueError(
            f"JSON 中没有可用样本: {json_path}. 统计信息: {stats}"
        )

    df = pd.DataFrame(rows)
    print(f"加载JSON: {json_path}")
    print(f"样本统计: {stats}")
    print(f"标签分布: {df['label'].value_counts().sort_index().to_dict()}")

    return df


class ThyroidJsonDataset(Dataset):
    def __init__(self, df, image_base_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_base_path = image_base_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx]["Path"]
        label = int(self.df.iloc[idx]["label"])
        full_path = os.path.join(self.image_base_path, str(rel_path))

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as exc:
            print(f"警告: 无法加载图像 {full_path}: {exc}")
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        return image, label, full_path


def load_model(pth_path, num_classes=2, architecture="resnet"):
    print(f"加载模型权重: {pth_path}")

    if architecture == "resnet":
        model = models.resnet18(pretrained=False)
        features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(features, num_classes))
    else:
        raise ValueError(f"不支持的模型架构: {architecture}")

    state_dict = torch.load(pth_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict, strict=False)
    print("模型权重加载成功")
    return model


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="评估中"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def find_optimal_threshold(y_true, y_probs, metric="f1", thresholds=None):
    if y_probs.ndim > 1:
        y_probs_positive = y_probs[:, 1] if y_probs.shape[1] >= 2 else y_probs[:, 0]
    else:
        y_probs_positive = y_probs

    if len(np.unique(y_true)) < 2:
        print("警告: 验证集中只有一个类别，阈值回退到0.5")
        return 0.5, 0.0, []

    if thresholds is None:
        min_prob = y_probs_positive.min()
        max_prob = y_probs_positive.max()
        thresholds = np.linspace(max(0.01, min_prob), min(0.99, max_prob), 100)

    threshold_scores = []
    best_score = -1
    optimal_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_probs_positive >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, average="binary", zero_division=0, pos_label=1)
        elif metric == "youden":
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                score = sensitivity + specificity - 1
            else:
                score = 0.0
        elif metric == "f1_youden":
            f1_val = f1_score(y_true, y_pred, average="binary", zero_division=0, pos_label=1)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                youden = sensitivity + specificity - 1
                score = 0.7 * f1_val + 0.3 * youden
            else:
                score = f1_val
        else:
            raise ValueError(f"未知阈值指标: {metric}")

        threshold_scores.append((threshold, score))
        if score > best_score:
            best_score = score
            optimal_threshold = threshold

    return optimal_threshold, best_score, threshold_scores


def calculate_metrics(y_true, y_probs, y_pred):
    metrics = {}

    if y_probs.ndim > 1:
        y_probs_positive = y_probs[:, 1] if y_probs.shape[1] >= 2 else y_probs[:, 0]
    else:
        y_probs_positive = y_probs

    actual_classes = len(np.unique(y_true))

    try:
        metrics["AUROC"] = roc_auc_score(y_true, y_probs_positive) if actual_classes >= 2 else 0.0
    except Exception:
        metrics["AUROC"] = 0.0

    try:
        metrics["AUPRC"] = average_precision_score(y_true, y_probs_positive) if actual_classes >= 2 else 0.0
    except Exception:
        metrics["AUPRC"] = 0.0

    metrics["Acc"] = accuracy_score(y_true, y_pred)
    metrics["Prec"] = precision_score(y_true, y_pred, average="binary", zero_division=0, pos_label=1)
    metrics["Recall"] = recall_score(y_true, y_pred, average="binary", zero_division=0, pos_label=1)
    metrics["F1"] = f1_score(y_true, y_pred, average="binary", zero_division=0, pos_label=1)
    metrics["Sensitivity"] = metrics["Recall"]

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    elif cm.shape == (1, 1):
        metrics["Specificity"] = 1.0 if np.unique(y_true)[0] == 0 else 0.0
    else:
        metrics["Specificity"] = 0.0

    return metrics


def bootstrap_metric_cis(y_true, y_probs, threshold, n_bootstrap=1000, ci_level=0.95, seed=42):
    if len(y_true) == 0:
        return {}

    if not 0 < ci_level < 1:
        raise ValueError("ci_level 必须在 0 和 1 之间")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap 必须大于 0")

    rng = np.random.default_rng(seed)
    point_probs = y_probs[:, 1] if y_probs.ndim > 1 and y_probs.shape[1] >= 2 else y_probs
    point_pred = (point_probs >= threshold).astype(int)
    metric_names = list(calculate_metrics(y_true, y_probs, point_pred).keys())
    bootstrap_values = {name: [] for name in metric_names}

    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        y_true_boot = y_true[indices]
        y_probs_boot = y_probs[indices]
        probs_positive_boot = y_probs_boot[:, 1] if y_probs_boot.ndim > 1 and y_probs_boot.shape[1] >= 2 else y_probs_boot
        y_pred_boot = (probs_positive_boot >= threshold).astype(int)
        metrics_boot = calculate_metrics(y_true_boot, y_probs_boot, y_pred_boot)
        for name in metric_names:
            bootstrap_values[name].append(float(metrics_boot[name]))

    alpha = (1.0 - ci_level) / 2.0
    metric_cis = {}
    for name, values in bootstrap_values.items():
        lower = float(np.quantile(values, alpha))
        upper = float(np.quantile(values, 1.0 - alpha))
        metric_cis[name] = (lower, upper)

    return metric_cis


def build_dataloader(df, image_base_path, batch_size, num_workers):
    data_transform = transforms.Compose(
        [
            transforms.CenterCrop(512),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ThyroidJsonDataset(df=df, image_base_path=image_base_path, transform=data_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main():
    parser = argparse.ArgumentParser(description="模型测试脚本（JSON版）")

    parser.add_argument("--pth_path", type=str, required=True, help="pth权重文件路径")

    parser.add_argument("--test_json_path", type=str, required=True, help="测试JSON路径")
    parser.add_argument("--valid_json_path", type=str, default=None, help="验证JSON路径（可选）")

    parser.add_argument("--image_base_path", type=str, required=True, help="图像基础目录")

    parser.add_argument("--filename_key", type=str, default="filename", help="JSON中的图像路径字段名")
    parser.add_argument("--label_key", type=str, default="FTCPTC", help="JSON中的标签字段名")
    parser.add_argument("--label_map", type=str, default="0:0,1:1", help="标签映射，例如 0:0,1:1")
    parser.add_argument("--ignore_labels", type=str, default="-1", help="忽略标签，例如 -1 或 -1,99")

    parser.add_argument(
        "--require_exists",
        action="store_true",
        help="开启后过滤磁盘上不存在的图像",
    )

    parser.add_argument("--threshold", type=float, default=None, help="手动阈值")
    parser.add_argument(
        "--threshold_metric",
        type=str,
        default="f1",
        choices=["f1", "youden", "f1_youden"],
        help="验证集阈值选择指标",
    )

    parser.add_argument("--num_classes", type=int, default=2, help="分类数，默认2")
    parser.add_argument("--architecture", type=str, default="resnet", choices=["resnet"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--bootstrap_samples", type=int, default=1000, help="bootstrap 重采样次数")
    parser.add_argument("--ci_level", type=float, default=0.95, help="置信区间置信水平")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    if not os.path.exists(args.pth_path):
        raise FileNotFoundError(f"权重文件不存在: {args.pth_path}")
    if not os.path.exists(args.test_json_path):
        raise FileNotFoundError(f"测试JSON不存在: {args.test_json_path}")
    if args.valid_json_path and not os.path.exists(args.valid_json_path):
        raise FileNotFoundError(f"验证JSON不存在: {args.valid_json_path}")
    if not os.path.exists(args.image_base_path):
        raise FileNotFoundError(f"image_base_path不存在: {args.image_base_path}")

    label_map = parse_label_map(args.label_map)
    ignore_labels = parse_ignore_labels(args.ignore_labels)

    test_df = load_json_samples(
        json_path=args.test_json_path,
        image_base_path=args.image_base_path,
        filename_key=args.filename_key,
        label_key=args.label_key,
        label_map=label_map,
        ignore_labels=ignore_labels,
        require_exists=args.require_exists,
    )

    test_loader = build_dataloader(
        df=test_df,
        image_base_path=args.image_base_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    model = load_model(
        pth_path=args.pth_path,
        num_classes=args.num_classes,
        architecture=args.architecture,
    ).to(device)

    used_threshold = 0.5

    if args.valid_json_path:
        print("=" * 60)
        print("在验证集上搜索最优阈值")
        print("=" * 60)
        valid_df = load_json_samples(
            json_path=args.valid_json_path,
            image_base_path=args.image_base_path,
            filename_key=args.filename_key,
            label_key=args.label_key,
            label_map=label_map,
            ignore_labels=ignore_labels,
            require_exists=args.require_exists,
        )
        valid_loader = build_dataloader(
            df=valid_df,
            image_base_path=args.image_base_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        valid_labels, valid_probs = evaluate_model(model, valid_loader, device)
        used_threshold, best_score, threshold_scores = find_optimal_threshold(
            valid_labels, valid_probs, metric=args.threshold_metric
        )
        print(f"最优阈值: {used_threshold:.4f}")
        print(f"最优{args.threshold_metric.upper()}分数: {best_score:.4f}")
        if threshold_scores:
            print("阈值搜索完成")
    elif args.threshold is not None:
        used_threshold = float(args.threshold)
        print(f"使用手动阈值: {used_threshold:.4f}")
    else:
        print(f"使用默认阈值: {used_threshold:.4f}")

    print("=" * 60)
    print("评估测试集")
    print("=" * 60)

    test_labels, test_probs = evaluate_model(model, test_loader, device)
    probs_positive = test_probs[:, 1] if test_probs.ndim > 1 and test_probs.shape[1] >= 2 else test_probs
    test_preds = (probs_positive >= used_threshold).astype(int)

    metrics = calculate_metrics(test_labels, test_probs, test_preds)
    metric_cis = bootstrap_metric_cis(
        test_labels,
        test_probs,
        threshold=used_threshold,
        n_bootstrap=args.bootstrap_samples,
        ci_level=args.ci_level,
        seed=args.seed,
    )
    metrics["threshold"] = used_threshold
    metrics["threshold_method"] = (
        "validation" if args.valid_json_path else ("manual" if args.threshold is not None else "default")
    )

    print("\n分类指标:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:15s}: {v:.4f}")
            if k in metric_cis:
                ci_lower, ci_upper = metric_cis[k]
                print(f"{(k + ' 95% CI'):15s}: [{ci_lower:.4f}, {ci_upper:.4f}]")
        else:
            print(f"{k:15s}: {v}")

    metrics_row = dict(metrics)
    for k, (ci_lower, ci_upper) in metric_cis.items():
        metrics_row[f"{k}_ci_lower"] = ci_lower
        metrics_row[f"{k}_ci_upper"] = ci_upper

    # 保存指标
    test_json_stem = Path(args.test_json_path).stem
    out_csv = Path(args.test_json_path).with_name(f"{test_json_stem}_metrics.csv")
    pd.DataFrame([metrics_row]).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n结果已保存: {out_csv}")


if __name__ == "__main__":
    main()
