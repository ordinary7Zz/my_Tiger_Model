#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
四分类版：使用 Tiger-Model 生成图像 + 原始图像训练 ResNet。
逻辑与 run_tiger_model.py 一致，支持 4 个亚型（标签 0–3）。
"""

import os
import shutil
import random
import torch
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
import cv2

# ========== 配置参数 ==========

PRETRAIN_MODEL_PATH = "./model/pretrain"
CONTROLNET_BG_PATH = "./model/fine-train-model/controlnet_bg"
CONTROLNET_ND_PATH = "./model/fine-train-model/controlnet_nd"

# 四个亚型：名称、原始图像目录、生成倍数（None 表示使用 GENERATE_PER_IMAGE）
SUBTYPE1_NAME = "乳头状癌"
SUBTYPE1_DIR = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/cropped/1"

SUBTYPE2_NAME = "滤泡状癌"
SUBTYPE2_DIR = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/cropped/2"

SUBTYPE3_NAME = "髓样癌"
SUBTYPE3_DIR = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/cropped/3"

SUBTYPE4_NAME = "未分化癌"
SUBTYPE4_DIR = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Subtype_4cls/cropped/4"

SUBTYPE1_GENERATE_PER_IMAGE = 1
SUBTYPE2_GENERATE_PER_IMAGE = 1
SUBTYPE3_GENERATE_PER_IMAGE = 1
SUBTYPE4_GENERATE_PER_IMAGE = 1

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

GENERATE_PER_IMAGE = 5

EXCLUDE_TEST_CSV = None
RADIOMICS_FEATURES_CSV = "./output/radiomics_features_4cls.csv"

# 与二分类脚本区分，避免共用同一目录时互相覆盖
RESNET_DATA_ROOT = "./dataset/Resnet_training_data_4class"
NUM_CLASSES = 4

SUBTYPE_SPECS = [
    (SUBTYPE1_NAME, SUBTYPE1_DIR, SUBTYPE1_GENERATE_PER_IMAGE),
    (SUBTYPE2_NAME, SUBTYPE2_DIR, SUBTYPE2_GENERATE_PER_IMAGE),
    (SUBTYPE3_NAME, SUBTYPE3_DIR, SUBTYPE3_GENERATE_PER_IMAGE),
    (SUBTYPE4_NAME, SUBTYPE4_DIR, SUBTYPE4_GENERATE_PER_IMAGE),
]

SUBTYPE_LABEL_MAP = {name: str(i) for i, (name, _, _) in enumerate(SUBTYPE_SPECS)}
SUBTYPE_NAMES_ORDERED = [name for name, _, _ in SUBTYPE_SPECS]


def _build_dirs_to_create():
    dirs = []
    for i in range(NUM_CLASSES):
        for split in ("train", "valid", "test"):
            dirs.append(f"{RESNET_DATA_ROOT}/{split}/{i}")
    for name in SUBTYPE_NAMES_ORDERED:
        dirs.append(f"dataset/generated_images/{name}")
    dirs.extend([
        "dataset/CSV",
        "Figure/paper/image",
        "Figure/paper/mask_nd",
        "Figure/paper/mask_bg",
        "dataset/Allclass/condition_bg",
    ])
    return dirs


DIRS_TO_CREATE = _build_dirs_to_create()


def get_resnet_data_root_from_dirs(dirs):
    for d in dirs:
        normalized = d.replace("\\", "/")
        if normalized.endswith("/train/0"):
            return normalized[: -len("/train/0")]
    raise ValueError("无法从 dirs 配置中解析 ResNet 数据根目录")


def get_subtype_image_dir(subtype_name):
    for name, image_dir, _ in SUBTYPE_SPECS:
        if name == subtype_name:
            return image_dir
    return SUBTYPE1_DIR


def create_dirs():
    for d in DIRS_TO_CREATE:
        os.makedirs(d, exist_ok=True)
    print("✅ 目录创建完成")


def get_image_files(directory):
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


def generate_simple_mask(image_path, output_mask_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 3
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    cv2.imwrite(output_mask_path, mask)
    return True


def load_tiger_model():
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
    if pipe is None:
        return []
    try:
        init_image = Image.open(input_image_path).resize((512, 512))
        mask_image = Image.open(mask_path).resize((512, 512))
        control_image_bg = Image.open(control_bg_path).resize((512, 512))
        generated_files = []
        input_filename = os.path.basename(input_image_path)

        if features_dict is not None:
            base_prompt = generate_prompt_from_features(input_filename, features_dict, subtype_name)
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
            prompts = {
                "乳头状癌": [
                    "papillary thyroid carcinoma, malignant, wider-than-tall, clear, regular",
                    "papillary thyroid carcinoma, solid, hypoechoic, irregular",
                    "papillary thyroid carcinoma, taller-than-wide, unclear, regular",
                    "papillary thyroid carcinoma, solid, heterogeneous, clear",
                    "papillary thyroid carcinoma, wider-than-tall, hypoechoic, irregular"
                ],
                "滤泡状癌": [
                    "follicular thyroid carcinoma, malignant, taller-than-wide, clear, regular",
                    "follicular thyroid carcinoma, solid, isoechoic, regular",
                    "follicular thyroid carcinoma, wider-than-tall, unclear, irregular",
                    "follicular thyroid carcinoma, solid, homogeneous, clear",
                    "follicular thyroid carcinoma, circular, hypoechoic, regular"
                ],
                "髓样癌": [
                    "medullary thyroid carcinoma, malignant, hypoechoic, irregular margin",
                    "medullary thyroid carcinoma, solid nodule, heterogeneous, vascular",
                    "medullary thyroid carcinoma, taller-than-wide, coarse calcification hint",
                    "medullary thyroid carcinoma, hypoechoic, lobulated",
                    "medullary thyroid carcinoma, solid, irregular shape"
                ],
                "未分化癌": [
                    "anaplastic thyroid carcinoma, malignant, large invasive mass, ill-defined irregular margin, heterogeneous",
                    "anaplastic thyroid carcinoma, hypoechoic, solid, locally advanced appearance, vascular",
                    "anaplastic thyroid carcinoma, rapidly growing nodule, coarse texture, irregular shape",
                    "anaplastic thyroid carcinoma, heterogeneous echotexture, posterior shadowing suggestion, malignant",
                    "anaplastic thyroid carcinoma, solid mass, lobulated contour, hypoechoic, aggressive"
                ],
            }
            default_list = [
                f"{subtype_name} thyroid ultrasound nodule, medical imaging, B-mode"
            ] * 5
            prompt_list = prompts.get(subtype_name, default_list)

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
    if features_dict is None or filename not in features_dict:
        default_prompts = {
            "乳头状癌": "papillary thyroid carcinoma, malignant, wider-than-tall or irregular, hypoechoic",
            "滤泡状癌": "follicular thyroid carcinoma, malignant, encapsulated or solid, isoechoic to hypoechoic",
            "髓样癌": "medullary thyroid carcinoma, malignant, solid, hypoechoic, coarse calcification hint",
            "未分化癌": "anaplastic thyroid carcinoma, malignant, large invasive mass, ill-defined margin, heterogeneous",
        }
        return default_prompts.get(subtype_name, f"{subtype_name}, thyroid ultrasound nodule, medical B-mode")

    features = features_dict[filename]
    prompt_parts = []

    if subtype_name == "乳头状癌":
        prompt_parts.append("papillary thyroid carcinoma")
    elif subtype_name == "滤泡状癌":
        prompt_parts.append("follicular thyroid carcinoma")
    elif subtype_name == "髓样癌":
        prompt_parts.append("medullary thyroid carcinoma")
    elif subtype_name == "未分化癌":
        prompt_parts.append("anaplastic thyroid carcinoma")
    else:
        prompt_parts.append(f"{subtype_name} thyroid nodule")

    elongation = features.get('original_shape2D_Elongation', 0.7)
    if elongation > 0.85:
        prompt_parts.append("taller-than-wide")
    elif elongation < 0.65:
        prompt_parts.append("wider-than-tall")
    else:
        prompt_parts.append("oval shape")

    sphericity = features.get('original_shape2D_Sphericity', 0.5)
    if sphericity > 0.7:
        prompt_parts.append("regular margin")
    elif sphericity < 0.4:
        prompt_parts.append("irregular margin")
    else:
        prompt_parts.append("moderately regular")

    contrast = features.get('original_glcm_Contrast', 0.5)
    if contrast > 0.7:
        prompt_parts.append("high contrast")
    elif contrast < 0.3:
        prompt_parts.append("low contrast")

    homogeneity = features.get('original_glcm_Idm', 0.8)
    if homogeneity > 0.9:
        prompt_parts.append("homogeneous")
    elif homogeneity < 0.75:
        prompt_parts.append("heterogeneous")

    entropy = features.get('original_firstorder_Entropy', 2.0)
    if entropy > 2.5:
        prompt_parts.append("complex texture")
    elif entropy < 1.5:
        prompt_parts.append("simple texture")

    mean_intensity = features.get('original_firstorder_Mean', 100)
    if mean_intensity > 110:
        prompt_parts.append("hyperechoic")
    elif mean_intensity < 70:
        prompt_parts.append("hypoechoic")
    else:
        prompt_parts.append("isoechoic")

    return ", ".join(prompt_parts)


def load_excluded_test_images(exclude_test_csv, image_base_path="dataset"):
    if exclude_test_csv is None or not os.path.exists(exclude_test_csv):
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(exclude_test_csv)
        excluded_files = set()
        for _, row in df.iterrows():
            path = row.get('Path', '')
            if 'generated_images' not in str(path):
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
    print(f"\n准备 {subtype_name} 的 ResNet 数据...")
    if excluded_files is None:
        excluded_files = set()
    if resnet_data_root is None:
        resnet_data_root = get_resnet_data_root_from_dirs(DIRS_TO_CREATE)

    original_images = get_image_files(images_dir)
    print(f"原始图像: {len(original_images)} 张")
    if excluded_files:
        original_images_filtered = [img for img in original_images if img not in excluded_files]
        excluded_count = len(original_images) - len(original_images_filtered)
        if excluded_count > 0:
            print(f"   排除测试集图像: {excluded_count} 张")
        original_images = original_images_filtered

    generated_images = get_image_files(generated_dir) if os.path.exists(generated_dir) else []
    if excluded_files and generated_images:
        generated_images_filtered = []
        excluded_generated_count = 0
        for gen_img in generated_images:
            parts = gen_img.replace('.png', '').split('_')
            excluded = False
            if len(parts) >= 4 and parts[0] == 'generated' and parts[1] == subtype_name:
                original_img_parts = []
                for idx in range(2, len(parts)):
                    if parts[idx].isdigit():
                        break
                    original_img_parts.append(parts[idx])
                if original_img_parts:
                    original_img_name = '_'.join(original_img_parts)
                    for excluded_file in excluded_files:
                        excluded_name = os.path.splitext(excluded_file)[0]
                        if original_img_name == excluded_name:
                            excluded_generated_count += 1
                            excluded = True
                            break
            if not excluded:
                generated_images_filtered.append(gen_img)
        if excluded_generated_count > 0:
            print(f"   排除测试集相关的生成图像: {excluded_generated_count} 张")
        generated_images = generated_images_filtered

    print(f"生成图像: {len(generated_images)} 张")
    all_images = []
    for img in original_images:
        all_images.append({'file': img, 'path': os.path.join(images_dir, img), 'type': 'original'})
    for img in generated_images:
        all_images.append({'file': img, 'path': os.path.join(generated_dir, img), 'type': 'generated'})

    if len(all_images) == 0:
        print(f"⚠️  {subtype_name} 没有图像，跳过")
        return

    train_files, temp = train_test_split(all_images, test_size=1 - TRAIN_RATIO, random_state=42)
    val_files, test_files = train_test_split(
        temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42
    )
    print(f"分割: train={len(train_files)}, valid={len(val_files)}, test={len(test_files)}")

    label = SUBTYPE_LABEL_MAP.get(subtype_name, "0")
    for split_name in ["train", "valid", "test"]:
        dst_dir = os.path.join(resnet_data_root, split_name, label)
        if os.path.exists(dst_dir):
            old_files = get_image_files(dst_dir)
            for old_file in old_files:
                try:
                    os.remove(os.path.join(dst_dir, old_file))
                except Exception:
                    pass

    for split_name, split_files in [("train", train_files), ("valid", val_files), ("test", test_files)]:
        for item in split_files:
            dst_dir = os.path.join(resnet_data_root, split_name, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, item['file'])
            shutil.copy(item['path'], dst_path)


def create_csv_files(base_dir="dataset", resnet_data_root=None):
    import pandas as pd
    if resnet_data_root is None:
        resnet_data_root = get_resnet_data_root_from_dirs(DIRS_TO_CREATE)

    label_legend = {i: SUBTYPE_NAMES_ORDERED[i] for i in range(NUM_CLASSES)}

    for split in ["train", "valid", "test"]:
        csv_data = []
        for lbl in range(NUM_CLASSES):
            label_dir = os.path.join(resnet_data_root, split, str(lbl))
            if not os.path.exists(label_dir):
                continue
            images = get_image_files(label_dir)
            subtype = label_legend[lbl]
            for img in images:
                rel_path = os.path.relpath(os.path.join(label_dir, img), start=base_dir).replace("\\", "/")
                csv_data.append({
                    'Path': rel_path,
                    'label': lbl,
                    'subtype': subtype,
                })
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = f"{base_dir}/CSV/Thyroid_4class_{split}.csv"
            os.makedirs(f"{base_dir}/CSV", exist_ok=True)
            df.to_csv(csv_path, index=False)
            label_counts = df['label'].value_counts().sort_index()
            legend_str = ", ".join(f"{i}={label_legend[i]}" for i in range(NUM_CLASSES))
            print(f"✅ 创建 CSV: {csv_path} ({len(df)} 条记录)")
            print(f"   标签分布: {dict(label_counts)} ({legend_str})")


def main():
    print("=" * 60)
    print("四分类：Tiger-Model 生成 + ResNet 训练")
    print("=" * 60)

    print("\n步骤 1: 创建目录结构...")
    create_dirs()

    print("\n步骤 2: 检查原始图像...")
    counts = {}
    for name, image_dir, _ in SUBTYPE_SPECS:
        imgs = get_image_files(image_dir)
        counts[name] = imgs
        print(f"{name}: {len(imgs)} 张")

    if all(len(v) == 0 for v in counts.values()):
        print("❌ 没有找到原始图像，请检查路径配置")
        return

    non_empty = [len(counts[n]) for n in SUBTYPE_NAMES_ORDERED if len(counts[n]) > 0]
    if len(non_empty) >= 2:
        print("\n数据量概览（非空类）:")
        mx, mn = max(non_empty), min(non_empty)
        print(f"  最多 / 最少（张）: {mx} / {mn}  （比例 {mx/mn:.2f}）" if mn else "")

    print("\n步骤 3: 加载影像组学特征...")
    features_dict = load_radiomics_features(RADIOMICS_FEATURES_CSV)

    print("\n步骤 3.1: 检查测试集排除选项...")
    excluded_files = load_excluded_test_images(EXCLUDE_TEST_CSV, image_base_path="dataset")

    print("\n步骤 3.2: 检查图像生成需求...")
    need_generate = {}
    for name, image_dir, gen_override in SUBTYPE_SPECS:
        gen_n = gen_override if gen_override is not None else GENERATE_PER_IMAGE
        if gen_n > 1:
            filtered = [img for img in counts[name] if img not in excluded_files]
            if len(filtered) > 0:
                need_generate[name] = (filtered, gen_n)
                print(f"  {name}: 需要生成图像（倍数: {gen_n}，{len(filtered)} 张原始图像）")
            else:
                print(f"  {name}: 跳过图像生成（所有图像都在测试集中）")
        else:
            print(f"  {name}: 跳过图像生成（倍数: {gen_n}，使用原始图像）")

    if need_generate:
        print("\n准备生成所需的输入文件...")
        USE_ALL_IMAGES_FOR_GENERATION = True
        sample_images = {}
        for subtype_name, (images, gen_num) in need_generate.items():
            if USE_ALL_IMAGES_FOR_GENERATION:
                sample_images[subtype_name] = images
                print(f"  {subtype_name}: 将处理所有 {len(images)} 张图像")
            else:
                sample_images[subtype_name] = images[:min(5, len(images))]
                print(f"  {subtype_name}: 将处理 {len(sample_images[subtype_name])} 张图像（示例）")

        print("生成掩码和背景条件图...")
        for subtype_name, images in sample_images.items():
            src_root = get_subtype_image_dir(subtype_name)
            for img_file in images:
                src_path = os.path.join(src_root, img_file)
                dst_input = f"Figure/paper/image/{img_file}"
                shutil.copy(src_path, dst_input)
                mask_path = f"Figure/paper/mask_bg/{img_file}"
                generate_simple_mask(src_path, mask_path)
                bg_condition = f"dataset/Allclass/condition_bg/{img_file}"
                shutil.copy(src_path, bg_condition)
    else:
        print("  所有亚型都不需要生成图像，跳过输入文件准备")
        sample_images = {}

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

    if need_generate and sample_images:
        print("\n步骤 5: 使用 Tiger-Model 生成图像...")
        pipe = load_tiger_model()
        if pipe is not None:
            bg_condition_file = None
            if os.path.exists("dataset/Allclass/condition_bg"):
                bg_files = get_image_files("dataset/Allclass/condition_bg")
                if bg_files:
                    bg_condition_file = f"dataset/Allclass/condition_bg/{bg_files[0]}"
            if bg_condition_file:
                for subtype_name, images in sample_images.items():
                    generate_num = need_generate[subtype_name][1]
                    print(f"\n为 {subtype_name} 生成图像...")
                    output_dir = f"dataset/generated_images/{subtype_name}"
                    print(f"  生成倍数: 每个原始图像生成 {generate_num} 张")
                    print(f"  将处理 {len(images)} 张原始图像")
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
        print("\n步骤 4/5: 跳过图像生成（所有亚型生成倍数 <= 1）")

    print("\n步骤 6: 准备 ResNet 训练数据...")
    for name, image_dir, _ in SUBTYPE_SPECS:
        prepare_resnet_data(
            name,
            image_dir,
            f"dataset/generated_images/{name}",
            excluded_files=excluded_files,
        )

    print("\n步骤 7: 创建 CSV 文件...")
    create_csv_files()

    print("\n步骤 8: 开始训练 ResNet...")
    print("\n运行以下命令训练 ResNet:")
    print("\npython Resent/main_train.py \\")
    print("    --modelname 'Thyroid_4class_with_augmentation' \\")
    print("    --architecture 'resnet' \\")
    print("    --imagepath './dataset' \\")
    print("    --train_data 'Thyroid_4class_train' \\")
    print("    --valid_data 'Thyroid_4class_valid' \\")
    print("    --test_data 'Thyroid_4class_test' \\")
    print("    --learning_rate 0.0005 \\")
    print("    --batch_size 64 \\")
    print("    --num_epochs 200 \\")
    print("    --Class 4")

    print("\n" + "=" * 60)
    print("数据准备完成！")
    print("=" * 60)
    print("\n注意:")
    print("1. 如果 Tiger-Model 生成失败，脚本会继续使用原始图像")
    print("2. 请检查生成图像的质量")
    print("3. 可以根据需要调整生成参数与 SUBTYPE_SPECS 中的目录、类别名")
    if EXCLUDE_TEST_CSV:
        print(f"4. ✅ 已排除测试集图像（来自: {EXCLUDE_TEST_CSV}），避免数据泄露")
    else:
        print("4. ⚠️  未指定测试集排除选项，所有图像都会用于训练/验证/测试分割")


if __name__ == "__main__":
    main()
