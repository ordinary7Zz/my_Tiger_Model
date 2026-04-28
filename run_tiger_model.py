#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案一：使用 Tiger-Model 生成图像 + 原始图像训练 ResNet
完整实现脚本
"""

import os
import json
import shutil
import random
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
import cv2

# ========== 配置参数 ==========
# 请根据实际情况修改以下路径

# 模型路径
PRETRAIN_MODEL_PATH = "./model/pretrain"
CONTROLNET_BG_PATH = "./model/fine-train-model/controlnet_bg"
CONTROLNET_ND_PATH = "./model/fine-train-model/controlnet_nd"

# 原始图像目录（两个亚型）
SUBTYPE1_NAME = "PTC"  # 经典型甲状腺乳头癌
SUBTYPE1_DIR = "./dataset/subtype_1_processed_all"  # 修改为您的实际路径

SUBTYPE2_NAME = "FTC"  # 滤泡型
SUBTYPE2_DIR = "./dataset/subtype_2_processed_all"  # 修改为您的实际路径

# 数据集分割比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 生成图像数量（每个原始图像生成多少张）
# 可以分别为两个亚型设置不同的扩展倍数，解决数据不平衡问题
GENERATE_PER_IMAGE = 5  # 默认值（如果未设置亚型特定值，将使用此值）

# 分别为两个亚型设置生成倍数（可选）
# 如果设置了这些值，将覆盖 GENERATE_PER_IMAGE
# 例如：如果 SUBTYPE1 有100张图像，SUBTYPE2 有50张图像
# 设置 SUBTYPE1_GENERATE_PER_IMAGE = 3, SUBTYPE2_GENERATE_PER_IMAGE = 6
# 则 SUBTYPE1 生成 100*3=300张，SUBTYPE2 生成 50*6=300张，实现数据平衡
SUBTYPE1_GENERATE_PER_IMAGE = 1  # 例如: 3，设置为 None 表示使用 GENERATE_PER_IMAGE
SUBTYPE2_GENERATE_PER_IMAGE = 3  # 例如: 6，设置为 None 表示使用 GENERATE_PER_IMAGE

# 测试集排除选项（避免数据泄露）
# 如果指定了测试集CSV文件，将排除其中的原始图像，不用于训练/验证
EXCLUDE_TEST_CSV = None  # 例如: "./dataset/CSV/PTC_vs_FTC_test_original_only.csv"
                         # 设置为 None 表示不排除任何图像
# 影像组学特征CSV文件路径（用于生成个性化prompt）
RADIOMICS_FEATURES_CSV = "./output/radiomics_features.csv"  # 设置为 None 表示使用固定prompt

# 统一的 ResNet 数据目录配置（单一数据源）
RESNET_DATA_ROOT = "dataset/Resnet_training_data"
DIRS_TO_CREATE = [
    # 统一的训练数据目录
    f"{RESNET_DATA_ROOT}/train/0",
    f"{RESNET_DATA_ROOT}/train/1",
    f"{RESNET_DATA_ROOT}/valid/0",
    f"{RESNET_DATA_ROOT}/valid/1",
    f"{RESNET_DATA_ROOT}/test/0",
    f"{RESNET_DATA_ROOT}/test/1",
    # 其他必要目录
    "dataset/generated_images/PTC",
    "dataset/generated_images/FTC",
    "dataset/CSV",
    "Figure/paper/image",
    "Figure/paper/mask_nd",
    "Figure/paper/mask_bg",
    "dataset/Allclass/condition_bg"
]


def get_resnet_data_root_from_dirs(dirs):
    """从目录配置中解析 ResNet 数据根目录，避免路径写死。"""
    for d in dirs:
        normalized = d.replace("\\", "/")
        if normalized.endswith("/train/0"):
            return normalized[: -len("/train/0")]
    raise ValueError("无法从 dirs 配置中解析 ResNet 数据根目录")

# ========== 辅助函数 ==========

def create_dirs():
    """创建必要的目录"""
    # 统一目录结构：所有数据放在同一个目录下
    # train/0/ - PTC图像 (label=0)
    # train/1/ - FTC图像 (label=1)
    # valid/0/ - PTC图像 (label=0)
    # valid/1/ - FTC图像 (label=1)
    # test/0/ - PTC图像 (label=0)
    # test/1/ - FTC图像 (label=1)
    for d in DIRS_TO_CREATE:
        os.makedirs(d, exist_ok=True)
    print("✅ 目录创建完成")

def get_image_files(directory):
    """获取目录中的所有图像文件"""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def generate_simple_mask(image_path, output_mask_path):
    """生成简单的掩码图像（中心区域）"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 创建中心区域的掩码（假设结节在中心）
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 3
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    cv2.imwrite(output_mask_path, mask)
    return True

def load_tiger_model():
    """加载 Tiger-Model"""
    print("加载 Tiger-Model...")
    try:
        controlnet_bg = ControlNetModel.from_pretrained(
            CONTROLNET_BG_PATH,
            torch_dtype=torch.float16
        )
        
        pipe_bg = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            PRETRAIN_MODEL_PATH,
            controlnet=controlnet_bg,
            torch_dtype=torch.float16
        )
        pipe_bg.scheduler = DDIMScheduler.from_config(pipe_bg.scheduler.config)
        pipe_bg.enable_model_cpu_offload()
        print("✅ Tiger-Model 加载成功")
        return pipe_bg
    except Exception as e:
        print(f"❌ Tiger-Model 加载失败: {e}")
        return None

def generate_images_batch(pipe, input_image_path, mask_path, control_bg_path, 
                          output_dir, subtype_name, num_images=5, features_dict=None):
    """批量生成图像"""
    if pipe is None:
        return []
    
    try:
        init_image = Image.open(input_image_path).resize((512, 512))
        mask_image = Image.open(mask_path).resize((512, 512))
        control_image_bg = Image.open(control_bg_path).resize((512, 512))
        
        generated_files = []
        
        # 获取图像文件名（用于查找特征）
        input_filename = os.path.basename(input_image_path)
        
        # 如果有特征字典，使用特征生成个性化prompt
        if features_dict is not None:
            base_prompt = generate_prompt_from_features(input_filename, features_dict, subtype_name)
            # 为每张生成图像创建略有变化的prompt（添加一些随机变化）
            prompt_variations = [
                base_prompt,
                base_prompt + ", solid",
                base_prompt + ", clear boundary",
                base_prompt + ", typical features",
                base_prompt + ", characteristic appearance"
            ]
            prompt_list = prompt_variations
            print(f"  使用特征生成的prompt: {base_prompt}")
        else:
            # 使用固定的提示词（原有逻辑）
            prompts = {
                "PTC": [
                    "papillary thyroid carcinoma, malignant, wider-than-tall, clear, regular",
                    "papillary thyroid carcinoma, solid, hypoechoic, irregular",
                    "papillary thyroid carcinoma, taller-than-wide, unclear, regular",
                    "papillary thyroid carcinoma, solid, heterogeneous, clear",
                    "papillary thyroid carcinoma, wider-than-tall, hypoechoic, irregular"
                ],
                "FTC": [
                    "follicular thyroid carcinoma, malignant, taller-than-wide, clear, regular",
                    "follicular thyroid carcinoma, solid, isoechoic, regular",
                    "follicular thyroid carcinoma, wider-than-tall, unclear, irregular",
                    "follicular thyroid carcinoma, solid, homogeneous, clear",
                    "follicular thyroid carcinoma, circular, hypoechoic, regular"
                ]
            }
            prompt_list = prompts.get(subtype_name, prompts["PTC"])
        
        for i in range(num_images):
            seed = random.randint(1, 1000000)
            generator = torch.Generator().manual_seed(seed)
            
            prompt = prompt_list[i % len(prompt_list)]
            
            try:
                image = pipe(
                    prompt=prompt,
                    negative_prompt="",
                    num_inference_steps=20,
                    guidance_scale=0.02,
                    generator=generator,
                    eta=1.0,
                    controlnet_conditioning_scale=1.0,
                    image=init_image,
                    mask_image=mask_image,
                    control_image=control_image_bg,
                ).images[0]
                
                # 在文件名中包含原始图像名（不含扩展名），便于追溯来源和排除测试集
                original_img_name = os.path.splitext(os.path.basename(input_image_path))[0]
                output_file = f"{output_dir}/generated_{subtype_name}_{original_img_name}_{i}_{seed}.png"
                image.save(output_file)
                generated_files.append(output_file)
                
            except Exception as e:
                print(f"⚠️  生成图像失败 (seed={seed}): {e}")
                continue
        
        return generated_files
        
    except Exception as e:
        print(f"❌ 批量生成失败: {e}")
        return []

def load_radiomics_features(csv_path):
    """
    从CSV文件加载影像组学特征
    
    Args:
        csv_path: 特征CSV文件路径
    
    Returns:
        features_dict: {filename: {feature_name: value}} 映射字典
    """
    if csv_path is None or not os.path.exists(csv_path):
        print("⚠️  未找到特征CSV文件，将使用固定prompt")
        return None
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        features_dict = {}
        for _, row in df.iterrows():
            filename = row.get('filename', '')
            if filename:
                features_dict[filename] = row.to_dict()
        
        print(f"✅ 加载影像组学特征: {csv_path}")
        print(f"   包含 {len(features_dict)} 个图像的特征")
        return features_dict
    except Exception as e:
        print(f"⚠️  警告: 无法读取特征CSV {csv_path}: {e}")
        print("   将使用固定prompt")
        return None


def generate_prompt_from_features(filename, features_dict, subtype_name):
    """
    根据影像组学特征生成个性化prompt
    
    Args:
        filename: 图像文件名
        features_dict: 特征字典
        subtype_name: 亚型名称 (PTC/FTC)
    
    Returns:
        prompt: 生成的prompt字符串
    """
    if features_dict is None or filename not in features_dict:
        # 如果没有特征，使用默认prompt
        default_prompts = {
            "PTC": "papillary thyroid carcinoma, malignant, irregular",
            "FTC": "follicular thyroid carcinoma, malignant, regular"
        }
        return default_prompts.get(subtype_name, "thyroid nodule")
    
    features = features_dict[filename]
    
    # 构建prompt各部分
    prompt_parts = []
    
    # 1. 基础类型描述
    if subtype_name == "PTC":
        prompt_parts.append("papillary thyroid carcinoma")
    elif subtype_name == "FTC":
        prompt_parts.append("follicular thyroid carcinoma")
    
    # 2. 形态学特征 (shape features)
    # 使用elongation判断形状：接近1是圆形，远离1是椭圆/不规则
    elongation = features.get('original_shape2D_Elongation', 0.7)
    if elongation > 0.85:
        prompt_parts.append("taller-than-wide")
    elif elongation < 0.65:
        prompt_parts.append("wider-than-tall")
    else:
        prompt_parts.append("oval shape")
    
    # 使用sphericity判断规则性：接近1是规则，远离1是不规则
    sphericity = features.get('original_shape2D_Sphericity', 0.5)
    if sphericity > 0.7:
        prompt_parts.append("regular margin")
    elif sphericity < 0.4:
        prompt_parts.append("irregular margin")
    else:
        prompt_parts.append("moderately regular")
    
    # 3. 纹理特征 (texture features)
    # 使用GLCM Contrast判断对比度
    contrast = features.get('original_glcm_Contrast', 0.5)
    if contrast > 0.7:
        prompt_parts.append("high contrast")
    elif contrast < 0.3:
        prompt_parts.append("low contrast")
    
    # 使用GLCM Homogeneity/Idm判断均匀性
    homogeneity = features.get('original_glcm_Idm', 0.8)
    if homogeneity > 0.9:
        prompt_parts.append("homogeneous")
    elif homogeneity < 0.75:
        prompt_parts.append("heterogeneous")
    
    # 4. 灰度特征 (firstorder features)
    # 使用Entropy判断复杂度
    entropy = features.get('original_firstorder_Entropy', 2.0)
    if entropy > 2.5:
        prompt_parts.append("complex texture")
    elif entropy < 1.5:
        prompt_parts.append("simple texture")
    
    # 使用Mean判断回声
    mean_intensity = features.get('original_firstorder_Mean', 100)
    if mean_intensity > 110:
        prompt_parts.append("hyperechoic")
    elif mean_intensity < 70:
        prompt_parts.append("hypoechoic")
    else:
        prompt_parts.append("isoechoic")
    
    # 组合成最终prompt
    prompt = ", ".join(prompt_parts)
    return prompt


def load_excluded_test_images(exclude_test_csv, image_base_path="dataset"):
    """
    从测试集CSV中加载需要排除的图像文件名
    
    Args:
        exclude_test_csv: 测试集CSV文件路径
        image_base_path: 图像基础路径
    
    Returns:
        excluded_files: 需要排除的文件名集合（仅原始图像，不包括生成图像）
    """
    if exclude_test_csv is None or not os.path.exists(exclude_test_csv):
        return set()
    
    try:
        import pandas as pd
        df = pd.read_csv(exclude_test_csv)
        
        excluded_files = set()
        for _, row in df.iterrows():
            path = row.get('Path', '')
            # 只排除原始图像（通过路径判断，生成图像通常在 generated_images 目录）
            if 'generated_images' not in str(path):
                # 提取文件名
                filename = os.path.basename(str(path))
                excluded_files.add(filename)
        
        print(f"📋 从测试集CSV加载: {exclude_test_csv}")
        print(f"   排除 {len(excluded_files)} 个原始图像文件（不用于训练/验证）")
        return excluded_files
    except Exception as e:
        print(f"⚠️  警告: 无法读取测试集CSV {exclude_test_csv}: {e}")
        print("   将继续使用所有图像")
        return set()


def prepare_resnet_data(subtype_name, images_dir, generated_dir, base_dir="dataset", excluded_files=None, resnet_data_root=None):
    """
    准备 ResNet 训练数据（原始图像 + 生成图像）
    
    Args:
        subtype_name: 亚型名称
        images_dir: 原始图像目录
        generated_dir: 生成图像目录
        base_dir: 基础目录
        excluded_files: 需要排除的文件名集合（来自测试集）
    """
    print(f"\n准备 {subtype_name} 的 ResNet 数据...")
    
    if excluded_files is None:
        excluded_files = set()
    if resnet_data_root is None:
        resnet_data_root = get_resnet_data_root_from_dirs(DIRS_TO_CREATE)
    
    # 获取原始图像
    original_images = get_image_files(images_dir)
    print(f"原始图像: {len(original_images)} 张")
    
    # 排除测试集中的图像
    if excluded_files:
        original_images_filtered = [img for img in original_images if img not in excluded_files]
        excluded_count = len(original_images) - len(original_images_filtered)
        if excluded_count > 0:
            print(f"   排除测试集图像: {excluded_count} 张")
        original_images = original_images_filtered
    
    # 获取生成图像
    generated_images = get_image_files(generated_dir) if os.path.exists(generated_dir) else []
    
    # 排除来自测试集原始图像的生成图像
    # 生成图像文件名格式: generated_{subtype_name}_{original_img_name}_{i}_{seed}.png
    if excluded_files and generated_images:
        generated_images_filtered = []
        excluded_generated_count = 0
        for gen_img in generated_images:
            # 从生成图像文件名中提取原始图像名
            # 新格式: generated_{subtype_name}_{original_img_name}_{i}_{seed}.png
            # 旧格式: generated_{subtype_name}_{i}_{seed}.png
            parts = gen_img.replace('.png', '').split('_')
            excluded = False
            
            if len(parts) >= 4 and parts[0] == 'generated' and parts[1] == subtype_name:
                # 新格式：包含原始图像名
                # 从第3个部分开始，直到遇到数字（i），这些部分组合起来就是原始图像名
                original_img_parts = []
                for idx in range(2, len(parts)):
                    if parts[idx].isdigit():
                        # 找到第一个数字，之前的都是原始图像名
                        break
                    original_img_parts.append(parts[idx])
                
                if original_img_parts:
                    original_img_name = '_'.join(original_img_parts)
                    # 检查是否在排除列表中（比较文件名，不含扩展名）
                    for excluded_file in excluded_files:
                        excluded_name = os.path.splitext(excluded_file)[0]
                        if original_img_name == excluded_name:
                            excluded_generated_count += 1
                            excluded = True
                            break
            elif len(parts) >= 3 and parts[0] == 'generated' and parts[1] == subtype_name:
                # 旧格式：不包含原始图像名，无法识别来源
                # 为了安全，如果无法识别，保留该图像（保守策略）
                pass
            
            if not excluded:
                generated_images_filtered.append(gen_img)
        
        if excluded_generated_count > 0:
            print(f"   排除测试集相关的生成图像: {excluded_generated_count} 张")
        generated_images = generated_images_filtered
    
    print(f"生成图像: {len(generated_images)} 张")
    
    # 合并所有图像
    all_images = []
    
    # 添加原始图像路径
    for img in original_images:
        all_images.append({
            'file': img,
            'path': os.path.join(images_dir, img),
            'type': 'original'
        })
    
    # 添加生成图像路径
    for img in generated_images:
        all_images.append({
            'file': img,
            'path': os.path.join(generated_dir, img),
            'type': 'generated'
        })
    
    if len(all_images) == 0:
        print(f"⚠️  {subtype_name} 没有图像，跳过")
        return
    
    # 分割数据集
    train_files, temp = train_test_split(
        all_images,
        test_size=1-TRAIN_RATIO,
        random_state=42
    )
    val_files, test_files = train_test_split(
        temp,
        test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO),
        random_state=42
    )
    
    print(f"分割: train={len(train_files)}, valid={len(val_files)}, test={len(test_files)}")
    
    # 根据亚型分配标签：PTC=0, FTC=1
    if subtype_name == SUBTYPE1_NAME:  # PTC
        label = "0"
    elif subtype_name == SUBTYPE2_NAME:  # FTC
        label = "1"
    else:
        label = "0"  # 默认值
    
    # 在复制文件前，先清理目标目录中的旧文件（避免重复统计）
    # 统一目录结构：不再按亚型分开，直接使用 train/0, train/1 等
    for split_name in ["train", "valid", "test"]:
        dst_dir = os.path.join(resnet_data_root, split_name, label)
        if os.path.exists(dst_dir):
            # 清理旧文件
            old_files = get_image_files(dst_dir)
            if old_files:
                for old_file in old_files:
                    try:
                        os.remove(os.path.join(dst_dir, old_file))
                    except Exception as e:
                        pass  # 忽略删除错误
    
    # 复制新文件
    for split_name, split_files in [("train", train_files), ("valid", val_files), ("test", test_files)]:
        for item in split_files:
            # 统一目录结构：所有数据放在 train/0, train/1 等目录下
            dst_dir = os.path.join(resnet_data_root, split_name, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, item['file'])
            shutil.copy(item['path'], dst_path)

def create_csv_files(base_dir="dataset", resnet_data_root=None):
    """创建 CSV 文件用于 ResNet 训练（合并 PTC 和 FTC）"""
    import pandas as pd
    
    if resnet_data_root is None:
        resnet_data_root = get_resnet_data_root_from_dirs(DIRS_TO_CREATE)

    # 创建合并的 CSV 文件（包含 PTC 和 FTC）
    # 统一目录结构：直接从 train/0, train/1 等目录读取
    for split in ["train", "valid", "test"]:
        csv_data = []
        
        # PTC (label=0) - 从统一的 train/0 目录读取
        label_dir = os.path.join(resnet_data_root, split, "0")
        if os.path.exists(label_dir):
            images = get_image_files(label_dir)
            for img in images:
                rel_path = os.path.relpath(os.path.join(label_dir, img), start=base_dir).replace("\\", "/")
                csv_data.append({
                    'Path': rel_path,
                    'label': 0,  # PTC = 0
                    'subtype': 'PTC'
                })
        
        # FTC (label=1) - 从统一的 train/1 目录读取
        label_dir = os.path.join(resnet_data_root, split, "1")
        if os.path.exists(label_dir):
            images = get_image_files(label_dir)
            for img in images:
                rel_path = os.path.relpath(os.path.join(label_dir, img), start=base_dir).replace("\\", "/")
                csv_data.append({
                    'Path': rel_path,
                    'label': 1,  # FTC = 1
                    'subtype': 'FTC'
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = f"{base_dir}/CSV/PTC_vs_FTC_{split}.csv"
            os.makedirs(f"{base_dir}/CSV", exist_ok=True)
            df.to_csv(csv_path, index=False)
            
            # 统计信息
            label_counts = df['label'].value_counts().sort_index()
            print(f"✅ 创建 CSV: {csv_path} ({len(df)} 条记录)")
            print(f"   标签分布: {dict(label_counts)} (0=PTC, 1=FTC)")

def main():
    print("=" * 60)
    print("方案一：Tiger-Model 生成 + ResNet 训练")
    print("=" * 60)
    
    # 步骤 1: 创建目录
    print("\n步骤 1: 创建目录结构...")
    create_dirs()
    
    # 步骤 2: 检查原始图像
    print("\n步骤 2: 检查原始图像...")
    subtype1_images = get_image_files(SUBTYPE1_DIR)
    subtype2_images = get_image_files(SUBTYPE2_DIR)
    
    print(f"{SUBTYPE1_NAME}: {len(subtype1_images)} 张")
    print(f"{SUBTYPE2_NAME}: {len(subtype2_images)} 张")
    
    if len(subtype1_images) == 0 and len(subtype2_images) == 0:
        print("❌ 没有找到原始图像，请检查路径配置")
        return
    
    # 显示数据不平衡情况和生成计划
    if len(subtype1_images) > 0 and len(subtype2_images) > 0:
        ratio = len(subtype1_images) / len(subtype2_images) if len(subtype2_images) > 0 else 0
        print(f"\n数据不平衡情况:")
        print(f"  {SUBTYPE1_NAME} / {SUBTYPE2_NAME} = {ratio:.2f}")
        
        # 计算生成倍数
        subtype1_gen = SUBTYPE1_GENERATE_PER_IMAGE if SUBTYPE1_GENERATE_PER_IMAGE is not None else GENERATE_PER_IMAGE
        subtype2_gen = SUBTYPE2_GENERATE_PER_IMAGE if SUBTYPE2_GENERATE_PER_IMAGE is not None else GENERATE_PER_IMAGE
        
        subtype1_total = len(subtype1_images) * subtype1_gen
        subtype2_total = len(subtype2_images) * subtype2_gen
        
        print(f"\n生成计划:")
        print(f"  {SUBTYPE1_NAME}: {len(subtype1_images)} 张原始图像 × {subtype1_gen} 倍 = {subtype1_total} 张生成图像")
        print(f"  {SUBTYPE2_NAME}: {len(subtype2_images)} 张原始图像 × {subtype2_gen} 倍 = {subtype2_total} 张生成图像")
        
        if subtype1_total > 0 and subtype2_total > 0:
            final_ratio = subtype1_total / subtype2_total
            print(f"  生成后比例: {SUBTYPE1_NAME} / {SUBTYPE2_NAME} = {final_ratio:.2f}")
            if abs(final_ratio - 1.0) < 0.1:
                print(f"  ✅ 数据已平衡")
            else:
                print(f"  ⚠️  数据仍不平衡，建议调整生成倍数")
    
    # 步骤 3: 加载影像组学特征（用于生成个性化prompt）
    print("\n步骤 3: 加载影像组学特征...")
    features_dict = load_radiomics_features(RADIOMICS_FEATURES_CSV)
    
    # 步骤 3.1: 加载需要排除的测试集图像（提前加载，用于后续排除）
    print("\n步骤 3.1: 检查测试集排除选项...")
    excluded_files = load_excluded_test_images(EXCLUDE_TEST_CSV, image_base_path="dataset")
    
    # 计算每个亚型的生成倍数，决定是否需要生成图像
    print("\n步骤 3.2: 检查图像生成需求...")
    
    # 计算每个亚型的生成倍数
    subtype1_gen = SUBTYPE1_GENERATE_PER_IMAGE if SUBTYPE1_GENERATE_PER_IMAGE is not None else GENERATE_PER_IMAGE
    subtype2_gen = SUBTYPE2_GENERATE_PER_IMAGE if SUBTYPE2_GENERATE_PER_IMAGE is not None else GENERATE_PER_IMAGE
    
    # 排除测试集图像后，判断哪些亚型需要生成图像（生成倍数 > 1）
    need_generate = {}
    if subtype1_gen > 1:
        # 排除测试集图像
        subtype1_images_filtered = [img for img in subtype1_images if img not in excluded_files]
        if len(subtype1_images_filtered) > 0:
            need_generate[SUBTYPE1_NAME] = (subtype1_images_filtered, subtype1_gen)
            print(f"  {SUBTYPE1_NAME}: 需要生成图像（倍数: {subtype1_gen}，{len(subtype1_images_filtered)} 张原始图像）")
        else:
            print(f"  {SUBTYPE1_NAME}: 跳过图像生成（所有图像都在测试集中）")
    else:
        print(f"  {SUBTYPE1_NAME}: 跳过图像生成（倍数: {subtype1_gen}，使用原始图像）")
    
    if subtype2_gen > 1:
        # 排除测试集图像
        subtype2_images_filtered = [img for img in subtype2_images if img not in excluded_files]
        if len(subtype2_images_filtered) > 0:
            need_generate[SUBTYPE2_NAME] = (subtype2_images_filtered, subtype2_gen)
            print(f"  {SUBTYPE2_NAME}: 需要生成图像（倍数: {subtype2_gen}，{len(subtype2_images_filtered)} 张原始图像）")
        else:
            print(f"  {SUBTYPE2_NAME}: 跳过图像生成（所有图像都在测试集中）")
    else:
        print(f"  {SUBTYPE2_NAME}: 跳过图像生成（倍数: {subtype2_gen}，使用原始图像）")
    
    # 只为需要生成的亚型准备输入文件
    if need_generate:
        print("\n准备生成所需的输入文件...")
        USE_ALL_IMAGES_FOR_GENERATION = True  # 设置为 True 处理所有图像，False 只处理部分
        
        sample_images = {}
        for subtype_name, (images, gen_num) in need_generate.items():
            if USE_ALL_IMAGES_FOR_GENERATION:
                sample_images[subtype_name] = images
                print(f"  {subtype_name}: 将处理所有 {len(images)} 张图像")
            else:
                sample_images[subtype_name] = images[:min(5, len(images))]
                print(f"  {subtype_name}: 将处理 {len(sample_images[subtype_name])} 张图像（示例）")
        
        # 创建掩码和背景条件图（简化处理）
        print("生成掩码和背景条件图...")
        for subtype_name, images in sample_images.items():
            for img_file in images:
                src_path = os.path.join(SUBTYPE1_DIR if subtype_name == SUBTYPE1_NAME else SUBTYPE2_DIR, img_file)
                
                # 复制到输入目录
                dst_input = f"Figure/paper/image/{img_file}"
                shutil.copy(src_path, dst_input)
                
                # 生成掩码
                mask_path = f"Figure/paper/mask_bg/{img_file}"
                generate_simple_mask(src_path, mask_path)
                
                # 使用原图作为背景条件图（简化处理）
                bg_condition = f"dataset/Allclass/condition_bg/{img_file}"
                shutil.copy(src_path, bg_condition)
    else:
        print("  所有亚型都不需要生成图像，跳过输入文件准备")
        sample_images = {}
    
    # 步骤 4: 清理旧的生成图像（如果存在），避免统计错误
    if need_generate:
        print("\n步骤 4: 清理旧的生成图像...")
        for subtype_name in need_generate.keys():
            output_dir = f"dataset/generated_images/{subtype_name}"
            if os.path.exists(output_dir):
                old_images = get_image_files(output_dir)
                if old_images:
                    print(f"  {subtype_name}: 发现 {len(old_images)} 张旧生成图像，正在清理...")
                    for old_img in old_images:
                        try:
                            os.remove(os.path.join(output_dir, old_img))
                        except Exception as e:
                            print(f"    警告: 无法删除 {old_img}: {e}")
                    print(f"  {subtype_name}: 清理完成")
    
    # 步骤 5: 加载 Tiger-Model 并生成图像（仅对需要生成的亚型）
    if need_generate and sample_images:
        print("\n步骤 5: 使用 Tiger-Model 生成图像...")
        pipe = load_tiger_model()
        
        if pipe is not None:
            # 准备一个背景条件图
            bg_condition_file = None
            if os.path.exists("dataset/Allclass/condition_bg"):
                bg_files = get_image_files("dataset/Allclass/condition_bg")
                if bg_files:
                    bg_condition_file = f"dataset/Allclass/condition_bg/{bg_files[0]}"
            
            if bg_condition_file:
                for subtype_name, images in sample_images.items():
                    # 获取该亚型的生成倍数
                    generate_num = need_generate[subtype_name][1]
                    
                    print(f"\n为 {subtype_name} 生成图像...")
                    output_dir = f"dataset/generated_images/{subtype_name}"
                    
                    print(f"  生成倍数: 每个原始图像生成 {generate_num} 张")
                    print(f"  将处理 {len(images)} 张原始图像")
                    
                    # 处理所有图像
                    for img_file in images:
                        input_path = f"Figure/paper/image/{img_file}"
                        mask_path = f"Figure/paper/mask_bg/{img_file}"
                        
                        if os.path.exists(input_path) and os.path.exists(mask_path):
                            generated = generate_images_batch(
                                pipe, input_path, mask_path, bg_condition_file,
                                output_dir, subtype_name, num_images=generate_num,
                                features_dict=features_dict
                            )
                            print(f"  {img_file}: 生成 {len(generated)} 张图像")
            else:
                print("⚠️  未找到背景条件图，跳过图像生成")
        else:
            print("⚠️  Tiger-Model 加载失败，将只使用原始图像")
    else:
        print("\n步骤 4: 跳过图像生成（所有亚型生成倍数 <= 1）")
    
    # 步骤 5: 准备 ResNet 训练数据（excluded_files 已在步骤3加载）
    print("\n步骤 6: 准备 ResNet 训练数据...")
    prepare_resnet_data(SUBTYPE1_NAME, SUBTYPE1_DIR, 
                       f"dataset/generated_images/{SUBTYPE1_NAME}",
                       excluded_files=excluded_files)
    prepare_resnet_data(SUBTYPE2_NAME, SUBTYPE2_DIR,
                       f"dataset/generated_images/{SUBTYPE2_NAME}",
                       excluded_files=excluded_files)
    
    # 步骤 7: 创建 CSV 文件
    print("\n步骤 7: 创建 CSV 文件...")
    create_csv_files()
    
    # 步骤 8: 训练 ResNet
    print("\n步骤 8: 开始训练 ResNet...")
    print("\n运行以下命令训练 ResNet:")
    print("\npython Resent/main_train.py \\")
    print("    --modelname 'Thyroid_PTC_vs_FTC_with_augmentation' \\")
    print("    --architecture 'resnet' \\")
    print("    --imagepath './dataset' \\")
    print("    --train_data 'PTC_vs_FTC_train' \\")
    print("    --valid_data 'PTC_vs_FTC_valid' \\")
    print("    --test_data 'PTC_vs_FTC_test' \\")
    print("    --learning_rate 0.0005 \\")
    print("    --batch_size 64 \\")
    print("    --num_epochs 200 \\")
    print("    --Class 2")
    
    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print("\n注意:")
    print("1. 如果 Tiger-Model 生成失败，脚本会继续使用原始图像")
    print("2. 请检查生成图像的质量")
    print("3. 可以根据需要调整生成参数")
    if EXCLUDE_TEST_CSV:
        print(f"4. ✅ 已排除测试集图像（来自: {EXCLUDE_TEST_CSV}），避免数据泄露")
    else:
        print("4. ⚠️  未指定测试集排除选项，所有图像都会用于训练/验证/测试分割")
        print("   建议：先运行 create_original_testset.py 创建测试集，然后设置 EXCLUDE_TEST_CSV")

if __name__ == "__main__":
    main()

