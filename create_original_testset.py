#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建仅包含原始图像的测试集CSV文件
用于模型性能的真实评估
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


def get_image_files(directory):
    """获取目录中的所有图像文件"""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


def create_original_testset(
    subtype1_dir, 
    subtype1_name, 
    subtype2_dir, 
    subtype2_name,
    output_csv_path,
    image_base_path="./dataset",
    subtype1_test_ratio=0.10,
    subtype2_test_ratio=0.10,
    random_state=42
):
    """
    创建仅包含原始图像的测试集
    
    Args:
        subtype1_dir: 第一个亚型的原始图像目录（绝对路径或相对路径）
        subtype1_name: 第一个亚型名称（如 "PTC"）
        subtype2_dir: 第二个亚型的原始图像目录（绝对路径或相对路径）
        subtype2_name: 第二个亚型名称（如 "FTC"）
        output_csv_path: 输出CSV文件路径
        image_base_path: 图像基础路径，CSV中的Path将相对于此路径，默认: ./dataset
        subtype1_test_ratio: 第一个亚型的测试集比例，默认0.10
        subtype2_test_ratio: 第二个亚型的测试集比例，默认0.10
        random_state: 随机种子，默认42
    """
    print("=" * 60)
    print("创建仅包含原始图像的测试集")
    print("=" * 60)
    
    # 获取原始图像
    subtype1_images = get_image_files(subtype1_dir)
    subtype2_images = get_image_files(subtype2_dir)
    
    print(f"\n{subtype1_name} 原始图像: {len(subtype1_images)} 张")
    print(f"{subtype2_name} 原始图像: {len(subtype2_images)} 张")
    
    if len(subtype1_images) == 0 and len(subtype2_images) == 0:
        print("❌ 错误: 没有找到原始图像")
        return
    
    print(f"\n测试集划分比例:")
    print(f"  {subtype1_name}: {subtype1_test_ratio:.1%}")
    print(f"  {subtype2_name}: {subtype2_test_ratio:.1%}")
    
    # 转换为绝对路径以便计算相对路径
    subtype1_dir_abs = os.path.abspath(subtype1_dir)
    subtype2_dir_abs = os.path.abspath(subtype2_dir)
    image_base_path_abs = os.path.abspath(image_base_path)
    
    # 准备 subtype1 数据
    subtype1_data = []
    for img in subtype1_images:
        img_full_path = os.path.join(subtype1_dir_abs, img)
        rel_path = os.path.relpath(img_full_path, image_base_path_abs)
        rel_path = rel_path.replace('\\', '/')
        subtype1_data.append({
            'Path': rel_path,
            'label': 0,  # subtype1 = 0
            'subtype': subtype1_name,
            'type': 'original'
        })
    
    # 准备 subtype2 数据
    subtype2_data = []
    for img in subtype2_images:
        img_full_path = os.path.join(subtype2_dir_abs, img)
        rel_path = os.path.relpath(img_full_path, image_base_path_abs)
        rel_path = rel_path.replace('\\', '/')
        subtype2_data.append({
            'Path': rel_path,
            'label': 1,  # subtype2 = 1
            'subtype': subtype2_name,
            'type': 'original'
        })
    
    # 分别对两个亚型进行分割
    if len(subtype1_data) > 0:
        subtype1_train_val, subtype1_test = train_test_split(
            subtype1_data,
            test_size=subtype1_test_ratio,
            random_state=random_state
        )
    else:
        subtype1_train_val, subtype1_test = [], []
    
    if len(subtype2_data) > 0:
        subtype2_train_val, subtype2_test = train_test_split(
            subtype2_data,
            test_size=subtype2_test_ratio,
            random_state=random_state
        )
    else:
        subtype2_train_val, subtype2_test = [], []
    
    # 合并测试集
    test_data = subtype1_test + subtype2_test
    train_val_data = subtype1_train_val + subtype2_train_val
    
    print(f"\n分割结果:")
    print(f"  {subtype1_name}:")
    print(f"    训练+验证集: {len(subtype1_train_val)} 张")
    print(f"    测试集: {len(subtype1_test)} 张")
    print(f"  {subtype2_name}:")
    print(f"    训练+验证集: {len(subtype2_train_val)} 张")
    print(f"    测试集: {len(subtype2_test)} 张")
    print(f"\n总计:")
    print(f"  训练+验证集: {len(train_val_data)} 张")
    print(f"  测试集: {len(test_data)} 张")
    
    # 统计测试集标签分布
    test_df = pd.DataFrame(test_data)
    if len(test_df) > 0:
        label_counts = test_df['label'].value_counts().sort_index()
        print(f"\n测试集标签分布:")
        for label, count in label_counts.items():
            subtype = test_df[test_df['label'] == label]['subtype'].iloc[0]
            print(f"  {subtype} (label={label}): {count} 张")
        
        # 计算测试集平衡度
        if len(label_counts) == 2:
            ratio = label_counts.iloc[0] / label_counts.iloc[1] if label_counts.iloc[1] > 0 else float('inf')
            print(f"  测试集比例: {label_counts.index[0]}:{label_counts.index[1]} = {ratio:.2f}:1")
    else:
        print("\n⚠️  警告: 测试集为空")
    
    # 保存测试集CSV
    # 调整路径格式（如果需要相对路径）
    test_df_output = test_df.copy()
    
    # 如果Path是绝对路径，可能需要转换为相对路径
    # 这里保持原样，用户可以根据需要调整
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # 保存CSV
    test_df_output[['Path', 'label', 'subtype']].to_csv(output_csv_path, index=False)
    print(f"\n✅ 测试集CSV已保存: {output_csv_path}")
    print(f"   包含 {len(test_df)} 条记录（仅原始图像）")
    
    return test_df


def main():
    parser = argparse.ArgumentParser(description='创建仅包含原始图像的测试集')
    parser.add_argument('--subtype1_dir', type=str, required=True,
                        help='第一个亚型的原始图像目录（如 ./dataset/subtype1）')
    parser.add_argument('--subtype1_name', type=str, default='PTC',
                        help='第一个亚型名称，默认: PTC')
    parser.add_argument('--subtype2_dir', type=str, required=True,
                        help='第二个亚型的原始图像目录（如 ./dataset/subtype2）')
    parser.add_argument('--subtype2_name', type=str, default='FTC',
                        help='第二个亚型名称，默认: FTC')
    parser.add_argument('--output_csv', type=str, 
                        default='./dataset/CSV/PTC_vs_FTC_test_original_only.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--image_base_path', type=str, default='./dataset',
                        help='图像基础路径，CSV中的Path将相对于此路径，默认: ./dataset')
    parser.add_argument('--subtype1_test_ratio', type=float, default=0.10,
                        help='第一个亚型的测试集比例，默认: 0.10')
    parser.add_argument('--subtype2_test_ratio', type=float, default=0.10,
                        help='第二个亚型的测试集比例，默认: 0.10')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子，默认: 42')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.subtype1_dir):
        print(f"❌ 错误: 目录不存在: {args.subtype1_dir}")
        return
    
    if not os.path.exists(args.subtype2_dir):
        print(f"❌ 错误: 目录不存在: {args.subtype2_dir}")
        return
    
    # 创建测试集
    create_original_testset(
        subtype1_dir=args.subtype1_dir,
        subtype1_name=args.subtype1_name,
        subtype2_dir=args.subtype2_dir,
        subtype2_name=args.subtype2_name,
        output_csv_path=args.output_csv,
        image_base_path=args.image_base_path,
        subtype1_test_ratio=args.subtype1_test_ratio,
        subtype2_test_ratio=args.subtype2_test_ratio,
        random_state=args.random_state
    )
    
    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print(f"使用以下命令评估模型:")
    print(f"python test_model.py \\")
    print(f"    --pth_path <your_model.pth> \\")
    print(f"    --csv_path {args.output_csv} \\")
    print(f"    --image_base_path {args.image_base_path}")


if __name__ == "__main__":
    main()

