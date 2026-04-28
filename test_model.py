#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型测试脚本（二分类专用）
输入pth权重文件和测试CSV文件路径，计算二分类指标
所有指标均以标签1为正类进行计算
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from tqdm import tqdm
from PIL import Image
import argparse


class Thyroid_Datasets(torch.utils.data.Dataset):
    """测试数据集类"""
    def __init__(self, csv_path, image_base_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_base_path = image_base_path
        self.transform = transform
        print(f"加载CSV文件: {csv_path}, 图像数量: {len(self.df)}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取图像路径和标签
        if 'Path' in self.df.columns:
            image_path = self.df.iloc[idx]['Path']
        else:
            raise ValueError("CSV文件必须包含'Path'列")
        
        if 'label' in self.df.columns:
            label = int(self.df.iloc[idx]['label'])
        else:
            raise ValueError("CSV文件必须包含'label'列")
        
        # 构建完整图像路径
        full_path = os.path.join(self.image_base_path, str(image_path))
        
        # 加载图像
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {full_path}: {e}")
            # 返回一个黑色图像作为占位符
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, full_path


def load_model(pth_path, num_classes=2, architecture='resnet'):
    """
    加载模型权重
    
    Args:
        pth_path: pth权重文件路径
        num_classes: 分类数量，默认2
        architecture: 模型架构，默认'resnet'
    
    Returns:
        加载好权重的模型
    """
    print(f"加载模型权重: {pth_path}")
    
    # 创建模型
    if architecture == 'resnet':
        model = models.resnet18(pretrained=False)
        features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(features, num_classes)
        )
    else:
        raise ValueError(f"不支持的模型架构: {architecture}")
    
    # 加载权重
    try:
        state_dict = torch.load(pth_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict, strict=False)
        print("✅ 模型权重加载成功")
    except Exception as e:
        print(f"❌ 模型权重加载失败: {e}")
        raise
    
    return model


def evaluate_model(model, dataloader, device):
    """
    评估模型并返回预测结果和真实标签
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        all_labels: 所有真实标签
        all_probs: 所有预测概率
        all_preds: 所有预测类别
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    print("开始评估模型...")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="评估中"):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 获取预测概率
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def find_optimal_threshold(y_true, y_probs, metric='f1', thresholds=None):
    """
    在验证集上寻找最优阈值
    
    Args:
        y_true: 真实标签（0或1）
        y_probs: 预测概率（形状为(n_samples, 2)或(n_samples,)）
        metric: 优化指标，可选 'f1', 'youden', 'f1_youden'，默认: 'f1'
        thresholds: 要尝试的阈值列表，如果为None则自动生成
    
    Returns:
        optimal_threshold: 最优阈值
        best_score: 最优分数
        threshold_scores: 所有阈值及其对应的分数
    """
    # 提取正类（标签1）的概率
    if y_probs.ndim > 1:
        if y_probs.shape[1] >= 2:
            y_probs_positive = y_probs[:, 1]
        else:
            y_probs_positive = y_probs[:, 0]
    else:
        y_probs_positive = y_probs
    
    # 检查是否有两个类别
    if len(np.unique(y_true)) < 2:
        print("警告: 验证集中只有一个类别，无法寻找最优阈值，使用默认阈值0.5")
        return 0.5, 0.0, []
    
    # 如果没有指定阈值，自动生成
    if thresholds is None:
        # 在概率范围内生成100个候选阈值
        min_prob = y_probs_positive.min()
        max_prob = y_probs_positive.max()
        thresholds = np.linspace(max(0.01, min_prob), min(0.99, max_prob), 100)
    
    threshold_scores = []
    best_score = -1
    optimal_threshold = 0.5
    
    for threshold in thresholds:
        # 使用当前阈值进行预测
        y_pred_thresh = (y_probs_positive >= threshold).astype(int)
        
        # 计算指标
        if metric == 'f1':
            score = f1_score(y_true, y_pred_thresh, average='binary', zero_division=0, pos_label=1)
        elif metric == 'youden':
            # Youden指数 = Sensitivity + Specificity - 1
            cm = confusion_matrix(y_true, y_pred_thresh)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                score = sensitivity + specificity - 1
            else:
                score = 0.0
        elif metric == 'f1_youden':
            # F1和Youden指数的加权平均
            f1 = f1_score(y_true, y_pred_thresh, average='binary', zero_division=0, pos_label=1)
            cm = confusion_matrix(y_true, y_pred_thresh)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                youden = sensitivity + specificity - 1
                score = 0.7 * f1 + 0.3 * youden  # F1权重更高
            else:
                score = f1
        else:
            raise ValueError(f"未知的优化指标: {metric}")
        
        threshold_scores.append((threshold, score))
        
        if score > best_score:
            best_score = score
            optimal_threshold = threshold
    
    return optimal_threshold, best_score, threshold_scores


def calculate_metrics(y_true, y_probs, y_pred, threshold=0.5):
    """
    计算二分类指标（以标签1为正类）
    
    Args:
        y_true: 真实标签（0或1）
        y_probs: 预测概率（形状为(n_samples, 2)或(n_samples,)）
        y_pred: 预测类别（0或1），如果为None则根据threshold从y_probs计算
        threshold: 分类阈值，默认0.5
    
    Returns:
        metrics_dict: 包含所有指标的字典
    """
    metrics_dict = {}
    
    # 提取正类（标签1）的概率
    if y_probs.ndim > 1:
        if y_probs.shape[1] >= 2:
            y_probs_positive = y_probs[:, 1]  # 使用第二列（标签1的概率）
        else:
            # 如果只有一列，假设是正类的概率分数
            y_probs_positive = y_probs[:, 0]
    else:
        # 一维数组，假设是正类的概率分数
        y_probs_positive = y_probs
    
    # 如果y_pred为None，根据threshold从y_probs计算
    if y_pred is None:
        y_pred = (y_probs_positive >= threshold).astype(int)
    
    # 检查测试集中是否有两个类别
    actual_classes = len(np.unique(y_true))
    
    # AUROC (Area Under ROC Curve) - 基于标签1
    try:
        if actual_classes < 2:
            print(f"警告: 测试集中只有 {actual_classes} 个类别，无法计算AUROC")
            metrics_dict['AUROC'] = 0.0
        else:
            metrics_dict['AUROC'] = roc_auc_score(y_true, y_probs_positive)
    except Exception as e:
        print(f"警告: AUROC计算失败: {e}")
        metrics_dict['AUROC'] = 0.0
    
    # AUPRC (Area Under Precision-Recall Curve) - 基于标签1
    try:
        if actual_classes < 2:
            print(f"警告: 测试集中只有 {actual_classes} 个类别，无法计算AUPRC")
            metrics_dict['AUPRC'] = 0.0
        else:
            metrics_dict['AUPRC'] = average_precision_score(y_true, y_probs_positive)
    except Exception as e:
        print(f"警告: AUPRC计算失败: {e}")
        metrics_dict['AUPRC'] = 0.0
    
    # Accuracy
    metrics_dict['Acc'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 - 基于标签1（正类）
    metrics_dict['Prec'] = precision_score(y_true, y_pred, average='binary', zero_division=0, pos_label=1)
    metrics_dict['Recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0, pos_label=1)
    metrics_dict['F1'] = f1_score(y_true, y_pred, average='binary', zero_division=0, pos_label=1)
    
    # Sensitivity (等同于Recall，基于标签1)
    metrics_dict['Sensitivity'] = metrics_dict['Recall']
    
    # Specificity (基于标签1，即标签0的召回率)
    # Specificity = TN / (TN + FP)，其中TN和FP都是相对于标签0的
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        # sklearn的confusion_matrix格式: [[TN, FP], [FN, TP]]
        # 行表示真实标签，列表示预测标签
        # 第一行第一列是TN（真实0，预测0）
        # 第一行第二列是FP（真实0，预测1）
        # 第二行第一列是FN（真实1，预测0）
        # 第二行第二列是TP（真实1，预测1）
        tn, fp, fn, tp = cm.ravel()
        metrics_dict['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    elif cm.shape == (1, 1):
        # 只有一类的情况
        if np.unique(y_true)[0] == 0:
            # 只有负类（标签0），所有预测都是负类
            metrics_dict['Specificity'] = 1.0
        else:
            # 只有正类（标签1），无法计算Specificity（没有真负例）
            metrics_dict['Specificity'] = 0.0
    else:
        # 其他形状的混淆矩阵（不应该出现）
        print(f"警告: 意外的混淆矩阵形状: {cm.shape}")
        metrics_dict['Specificity'] = 0.0
    
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description='模型测试脚本')
    parser.add_argument('--pth_path', type=str, required=True, help='pth权重文件路径')
    parser.add_argument('--csv_path', type=str, required=True, help='测试CSV文件路径')
    parser.add_argument('--valid_csv_path', type=str, default=None, 
                        help='验证集CSV文件路径（可选，用于自动选择最优阈值）')
    parser.add_argument('--threshold', type=float, default=None,
                        help='分类阈值（如果未指定且未提供验证集，则使用0.5）')
    parser.add_argument('--threshold_metric', type=str, default='f1',
                        choices=['f1', 'youden', 'f1_youden'],
                        help='用于选择最优阈值的指标，默认: f1')
    parser.add_argument('--image_base_path', type=str, default='./dataset', 
                        help='图像基础路径（CSV中的Path列相对于此路径），默认: ./dataset')
    parser.add_argument('--num_classes', type=int, default=2, help='分类数量（固定为2，二分类），默认: 2')
    parser.add_argument('--architecture', type=str, default='resnet', 
                        choices=['resnet'], help='模型架构，默认: resnet')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小，默认: 32')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数，默认: 4')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'cpu'], help='设备选择，默认: auto')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.pth_path):
        print(f"❌ 错误: 权重文件不存在: {args.pth_path}")
        return
    
    if not os.path.exists(args.csv_path):
        print(f"❌ 错误: CSV文件不存在: {args.csv_path}")
        return
    
    if args.valid_csv_path and not os.path.exists(args.valid_csv_path):
        print(f"❌ 错误: 验证集CSV文件不存在: {args.valid_csv_path}")
        return
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 数据预处理（与训练时保持一致）
    data_transform = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 创建数据集和数据加载器
    test_dataset = Thyroid_Datasets(
        csv_path=args.csv_path,
        image_base_path=args.image_base_path,
        transform=data_transform
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 加载模型
    model = load_model(
        pth_path=args.pth_path,
        num_classes=args.num_classes,
        architecture=args.architecture
    )
    model = model.to(device)
    
    # 确定使用的阈值
    used_threshold = 0.5  # 默认阈值
    
    # 如果提供了验证集，在验证集上寻找最优阈值
    if args.valid_csv_path:
        print("\n" + "=" * 60)
        print("在验证集上寻找最优阈值...")
        print("=" * 60)
        
        # 创建验证集数据加载器
        valid_dataset = Thyroid_Datasets(
            csv_path=args.valid_csv_path,
            image_base_path=args.image_base_path,
            transform=data_transform
        )
        
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # 评估验证集
        valid_labels, valid_probs, _ = evaluate_model(model, valid_dataloader, device)
        
        # 寻找最优阈值
        optimal_threshold, best_score, threshold_scores = find_optimal_threshold(
            valid_labels, valid_probs, metric=args.threshold_metric
        )
        
        used_threshold = optimal_threshold
        
        print(f"\n最优阈值: {optimal_threshold:.4f}")
        print(f"最优{args.threshold_metric.upper()}分数: {best_score:.4f}")
        print(f"\n验证集标签分布: {dict(zip(*np.unique(valid_labels, return_counts=True)))}")
        
        # 显示最优阈值附近的阈值分数（用于参考）
        print(f"\n阈值选择详情（显示最优阈值前后5个阈值）:")
        sorted_scores = sorted(threshold_scores, key=lambda x: x[0])
        optimal_idx = next(i for i, (t, _) in enumerate(sorted_scores) if abs(t - optimal_threshold) < 1e-6)
        start_idx = max(0, optimal_idx - 5)
        end_idx = min(len(sorted_scores), optimal_idx + 6)
        for threshold, score in sorted_scores[start_idx:end_idx]:
            marker = " <-- 最优" if abs(threshold - optimal_threshold) < 1e-6 else ""
            print(f"  阈值 {threshold:.4f}: {args.threshold_metric.upper()} = {score:.4f}{marker}")
        
        print("=" * 60)
    elif args.threshold is not None:
        # 如果用户手动指定了阈值
        used_threshold = args.threshold
        print(f"\n使用手动指定的阈值: {used_threshold:.4f}")
    else:
        print(f"\n使用默认阈值: {used_threshold:.4f}")
        print("提示: 可以使用 --valid_csv_path 在验证集上自动选择最优阈值")
    
    # 评估测试集
    print("\n" + "=" * 60)
    print("评估测试集...")
    print("=" * 60)
    all_labels, all_probs, _ = evaluate_model(model, test_dataloader, device)
    
    # 使用最优阈值进行预测
    if all_probs.ndim > 1 and all_probs.shape[1] >= 2:
        probs_positive = all_probs[:, 1]
    else:
        probs_positive = all_probs if all_probs.ndim == 1 else all_probs[:, 0]
    
    all_preds = (probs_positive >= used_threshold).astype(int)
    
    # 计算指标（二分类，以标签1为正类）
    print("\n计算二分类指标（以标签1为正类）...")
    print(f"使用阈值: {used_threshold:.4f}")
    
    # 诊断信息：检查概率分布
    print(f"\n诊断信息:")
    print(f"  正类概率范围: [{probs_positive.min():.4f}, {probs_positive.max():.4f}]")
    print(f"  正类概率均值: {probs_positive.mean():.4f}")
    print(f"  正类概率中位数: {np.median(probs_positive):.4f}")
    print(f"  预测为正类的样本数（阈值{used_threshold:.4f}）: {np.sum(all_preds == 1)}")
    print(f"  真实正类样本数: {np.sum(all_labels == 1)}")
    print(f"  真实负类样本数: {np.sum(all_labels == 0)}")
    
    # 检查不同阈值下的Recall（用于对比）
    test_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    if used_threshold not in test_thresholds:
        test_thresholds.append(used_threshold)
        test_thresholds.sort()
    print(f"\n不同阈值下的Recall（测试集）:")
    for thresh in test_thresholds:
        preds_thresh = (probs_positive >= thresh).astype(int)
        recall_thresh = recall_score(all_labels, preds_thresh, pos_label=1, zero_division=0)
        marker = " <-- 使用中" if abs(thresh - used_threshold) < 1e-6 else ""
        print(f"  阈值 {thresh:.4f}: Recall = {recall_thresh:.4f}, 预测为正类数 = {np.sum(preds_thresh)}{marker}")
    
    # 使用最优阈值计算指标
    metrics = calculate_metrics(all_labels, all_probs, all_preds, threshold=used_threshold)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("分类指标结果")
    print("=" * 60)
    print(f"测试样本数: {len(all_labels)}")
    print(f"标签分布: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    print(f"使用阈值: {used_threshold:.4f}")
    if args.valid_csv_path:
        print(f"阈值选择方法: 基于验证集的{args.threshold_metric.upper()}最大化")
    elif args.threshold is not None:
        print(f"阈值选择方法: 手动指定")
    else:
        print(f"阈值选择方法: 默认值")
    print("\n指标值:")
    print("-" * 60)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:15s}: {metric_value:.4f}")
    print("=" * 60)
    
    # 保存结果到CSV
    metrics['threshold'] = used_threshold
    metrics['threshold_method'] = 'validation_f1' if args.valid_csv_path else ('manual' if args.threshold is not None else 'default')
    results_df = pd.DataFrame([metrics])
    output_csv = args.csv_path.replace('.csv', '_metrics.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到: {output_csv}")


if __name__ == "__main__":
    main()

