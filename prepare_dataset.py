# coding=utf-8
"""
准备 ResNet 训练数据集的脚本

功能：
1. 从 PTC 和 FTC 图像生成合成图像（使用 Tiger Model）
2. 动态生成背景掩码和条件图像
3. 跳过测试集中的文件（从 xlsx 读取）
4. 自动划分 train/valid/test
5. 生成 ResNet 训练所需的 CSV 文件和图像目录结构

使用方法：
python prepare_dataset.py \
    --ptc_image_dir /path/to/ptc/images \
    --ftc_image_dir /path/to/ftc/images \
    --ptc_mask_dir /path/to/ptc/masks \
    --ftc_mask_dir /path/to/ftc/masks \
    --test_xlsx /path/to/test_set.csv \
    --base_model_path ./model/pretrain \
    --controlnet_path ./model/fine-train-model/controlnet_bg \
    --output_dir ./dataset \
    --ptc_multiplier 1 \
    --ftc_multiplier 10

模型路径说明：
--base_model_path: Stable Diffusion 基础模型路径
    项目中的路径: ./model/pretrain
    该目录应包含: model_index.json, unet/, vae/, text_encoder/, tokenizer/, scheduler/ 等

--controlnet_path: ControlNet 模型路径（用于背景生成）
    项目中的路径: ./model/fine-train-model/controlnet_bg
    该目录应包含: config.json, diffusion_pytorch_model.safetensors 等

输出结构：
dataset/
├── CSV/
│   ├── train.csv      # 训练集（原始+生成图像）
│   ├── valid.csv      # 验证集（原始+生成图像）
│   └── test.csv       # 测试集（仅原始图像）
├── PTC/
│   ├── image_001.png  # 原始图像
│   └── image_001_gen_1.png  # 生成图像（如果 multiplier > 1）
└── FTC/
    ├── image_002.png
    └── image_002_gen_1.png ... gen_10.png
"""

import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import random
from pathlib import Path
import shutil
from tqdm import tqdm
import sys

# 导入 Tiger Model 相关库
sys.path.append(os.path.join(os.path.dirname(__file__), "Tiger Model"))
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler


def make_inpaint_condition(image, image_mask):
    """动态生成条件图像（从 generation.py 复制）"""
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image1 = torch.from_numpy(image)
    return image1


def generate_background_mask(mask_nd_path):
    """从结节掩码生成背景掩码"""
    mask_nd = Image.open(mask_nd_path).convert("L")
    mask_nd_array = np.array(mask_nd)
    
    # 反向掩码
    if mask_nd_array.max() > 1:
        mask_bg_array = 255 - mask_nd_array
    else:
        mask_bg_array = (1 - mask_nd_array) * 255
    
    mask_bg = Image.fromarray(mask_bg_array.astype(np.uint8), mode='L')
    return mask_bg


def load_test_set(test_file_path):
    """
    从 CSV 或 Excel 文件加载测试集文件列表
    
    支持的格式：
    - CSV 文件（.csv）：应包含 Path 列或第一列为文件路径
    - Excel 文件（.xlsx, .xls）：应包含文件路径列
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(test_file_path):
            print(f"错误：测试集文件不存在: {test_file_path}")
            return set()
        
        # 根据文件扩展名选择读取方式
        file_ext = os.path.splitext(test_file_path)[1].lower()
        
        if file_ext == '.xlsx':
            # Excel 2007+ 格式，使用 openpyxl engine
            try:
                df = pd.read_excel(test_file_path, engine='openpyxl')
            except ImportError:
                print("错误：需要安装 openpyxl 库。请运行: pip install openpyxl")
                return set()
        elif file_ext == '.xls':
            # Excel 97-2003 格式，使用 xlrd engine
            try:
                df = pd.read_excel(test_file_path, engine='xlrd')
            except ImportError:
                print("错误：需要安装 xlrd 库。请运行: pip install xlrd")
                return set()
        elif file_ext in ['.csv', '.txt']:
            # CSV 格式文件
            df = pd.read_csv(test_file_path)
        else:
            # 尝试自动检测（先尝试 Excel，再尝试 CSV）
            try:
                df = pd.read_excel(test_file_path, engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(test_file_path, engine='xlrd')
                except:
                    try:
                        df = pd.read_csv(test_file_path)
                    except Exception as e:
                        print(f"错误：无法读取文件 {test_file_path}，错误: {e}")
                        return set()
        
        print(f"读取测试集文件 {test_file_path}，共 {len(df)} 行，列名: {list(df.columns)}")
        
        # 优先查找 Path 列（CSV 格式常见）
        test_files = set()
        if 'Path' in df.columns:
            # 从 Path 列读取文件路径
            paths = df['Path'].astype(str).str.strip().str.replace('\\', '/')
            for path in paths:
                if path and path != 'nan' and path.lower() != 'none':
                    # 提取文件名（不含路径）
                    filename = os.path.basename(path)
                    test_files.add(filename)
                    # 也添加完整路径（用于匹配）
                    test_files.add(path)
            print(f"从 'Path' 列读取到 {len(test_files)} 个测试集文件标识")
        else:
            # 尝试其他常见的列名
            possible_columns = ['file_name', 'filename', 'File', 'file', 'path', 'image', 'Image', 'name', 'Name']
            for col in possible_columns:
                if col in df.columns:
                    paths = df[col].astype(str).str.strip().str.replace('\\', '/')
                    for path in paths:
                        if path and path != 'nan' and path.lower() != 'none':
                            filename = os.path.basename(path)
                            test_files.add(filename)
                            test_files.add(path)
                    print(f"从列 '{col}' 读取到 {len(test_files)} 个测试集文件标识")
                    break
        
        if len(test_files) == 0:
            # 如果没有找到，尝试第一列
            if len(df.columns) > 0:
                paths = df.iloc[:, 0].astype(str).str.strip().str.replace('\\', '/')
                for path in paths:
                    if path and path != 'nan' and path.lower() != 'none':
                        filename = os.path.basename(path)
                        test_files.add(filename)
                        test_files.add(path)
                print(f"从第一列读取到 {len(test_files)} 个测试集文件标识")
        
        # 同时提取文件名（不含扩展名）用于更灵活的匹配
        test_files_without_ext = set()
        for f in test_files:
            # 添加完整路径或文件名
            test_files_without_ext.add(f)
            # 提取文件名（如果 f 是路径）
            filename = os.path.basename(f)
            test_files_without_ext.add(filename)
            # 添加不含扩展名的文件名
            base_name = os.path.splitext(filename)[0]
            test_files_without_ext.add(base_name)
        
        print(f"测试集文件匹配标识总数: {len(test_files_without_ext)}")
        return test_files_without_ext
    except ImportError as e:
        print(f"错误：缺少必要的库: {e}")
        print("请安装所需的库:")
        print("  - Excel 文件 (.xlsx): pip install openpyxl")
        print("  - Excel 文件 (.xls): pip install xlrd")
        return set()
    except Exception as e:
        print(f"警告：读取测试集文件失败: {e}")
        print(f"文件路径: {test_file_path}")
        print("提示：")
        print("  1. 确保文件格式正确（.xlsx, .xls, 或 .csv）")
        print("  2. 如果是 Excel 文件，请安装相应库: pip install openpyxl")
        print("  3. 如果是 CSV 文件，请确保编码正确（通常为 UTF-8）")
        print("  4. CSV 文件应包含 'Path' 列或第一列为文件路径")
        print("将跳过所有文件（不生成任何图像）")
        return set()


def get_image_files(image_dir, mask_dir, test_files, subtype):
    """
    获取需要处理的图像文件列表（排除测试集）
    
    注意：此函数会过滤掉测试集文件（从 xlsx 读取的文件），
    只返回非测试集文件。这些非测试集文件将用于：
    - 原始图像：可以用于 train/valid/test 划分
    - 生成图像：只能用于 train/valid，不能用于 test
    """
    image_files = []
    
    if not os.path.exists(image_dir):
        print(f"警告：图像目录不存在: {image_dir}")
        return image_files
    
    for img_file in os.listdir(image_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        # 检查是否在测试集中（支持多种匹配方式）
        base_name = os.path.splitext(img_file)[0]
        img_file_lower = img_file.lower()
        base_name_lower = base_name.lower()
        
        # 检查各种可能的匹配
        is_test_file = False
        for test_file in test_files:
            test_file_lower = str(test_file).lower()
            test_base = os.path.splitext(os.path.basename(str(test_file)))[0].lower()
            
            if (img_file_lower == test_file_lower or 
                base_name_lower == test_file_lower or
                base_name_lower == test_base or
                img_file_lower.endswith(test_file_lower) or
                test_file_lower.endswith(img_file_lower)):
                is_test_file = True
                break
        
        if is_test_file:
            continue
        
        # 检查对应的掩码文件是否存在
        # 尝试多种可能的掩码文件命名方式
        mask_file = None
        mask_path = None
        
        # 可能的掩码文件命名模式
        possible_mask_names = [
            img_file,  # 1. 完全相同的文件名
            base_name + '.png',  # 2. 相同基础名 + .png
            base_name + '.jpg',  # 3. 相同基础名 + .jpg
            base_name + '_mask.png',  # 4. 基础名 + _mask.png
            base_name + '_mask.jpg',  # 5. 基础名 + _mask.jpg
            base_name + '_nd.png',  # 6. 基础名 + _nd.png (nodule mask)
            base_name + '_nd.jpg',  # 7. 基础名 + _nd.jpg
            base_name + 'mask.png',  # 8. 基础名 + mask.png (无下划线)
            base_name + 'mask.jpg',  # 9. 基础名 + mask.jpg
        ]
        
        # 尝试查找掩码文件
        for mask_name in possible_mask_names:
            mask_path_candidate = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path_candidate):
                mask_file = mask_name
                mask_path = mask_path_candidate
                break
        
        # 如果还是找不到，尝试不区分大小写的匹配
        if mask_path is None:
            mask_dir_files = os.listdir(mask_dir) if os.path.exists(mask_dir) else []
            base_name_lower = base_name.lower()
            for mask_candidate in mask_dir_files:
                mask_candidate_lower = mask_candidate.lower()
                # 检查是否匹配（忽略扩展名）
                mask_candidate_base = os.path.splitext(mask_candidate)[0].lower()
                if (mask_candidate_base == base_name_lower or
                    mask_candidate_base == base_name_lower + '_mask' or
                    mask_candidate_base == base_name_lower + '_nd' or
                    mask_candidate_base == base_name_lower + 'mask'):
                    mask_file = mask_candidate
                    mask_path = os.path.join(mask_dir, mask_candidate)
                    break
        
        if mask_path is None or not os.path.exists(mask_path):
            print(f"警告：未找到 {img_file} 对应的掩码文件，跳过")
            print(f"  图像路径: {os.path.join(image_dir, img_file)}")
            print(f"  掩码目录: {mask_dir}")
            print(f"  尝试过的命名: {possible_mask_names[:5]}...")
            continue
        
        image_files.append({
            'image_path': os.path.join(image_dir, img_file),
            'mask_path': mask_path,
            'filename': img_file,
            'base_name': base_name
        })
    
    return image_files


def generate_image_with_tiger_model(pipe_bg, init_image, mask_image_nd, mask_image_bg, 
                                    prompt="black and white definition detail", seed=None):
    """使用 Tiger Model 生成图像（保留结节，生成新背景）"""
    # 动态生成条件图像
    control_image_bg = make_inpaint_condition(init_image, mask_image_bg)
    
    # 设置随机种子
    if seed is None:
        seed = random.randint(1, 1000000)
    generator = torch.Generator().manual_seed(seed)
    
    # 生成图像
    try:
        generated_image = pipe_bg(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=20,
            guidance_scale=0.02,
            generator=generator,
            eta=1.0,
            controlnet_conditioning_scale=1.0,
            image=init_image,
            mask_image=mask_image_bg,
            control_image=control_image_bg,
        ).images[0]
        return generated_image
    except Exception as e:
        print(f"生成图像时出错: {e}")
        return None


def load_tiger_model(base_model_path, controlnet_path, device="cuda"):
    """加载 Tiger Model"""
    print("正在加载 Tiger Model...")
    
    try:
        controlnet_bg = ControlNetModel.from_pretrained(
            controlnet_path, 
            torch_dtype=torch.float16
        )
        
        pipe_bg = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet_bg,
            torch_dtype=torch.float16
        )
        pipe_bg.scheduler = DDIMScheduler.from_config(pipe_bg.scheduler.config)
        pipe_bg.enable_model_cpu_offload()
        
        print("Tiger Model 加载完成")
        return pipe_bg
    except Exception as e:
        print(f"加载 Tiger Model 失败: {e}")
        raise


def process_subtype(pipe_bg, image_files, subtype, label, output_dir, 
                   generation_multiplier, test_files, prompt="black and white definition detail",
                   split_ratios=(0.7, 0.15, 0.15)):
    """
    处理一个亚型的数据
    
    Args:
        pipe_bg: Tiger Model 管道
        image_files: 图像文件列表
        subtype: 亚型名称（"PTC" 或 "FTC"）
        label: 标签（0 或 1）
        output_dir: 输出目录
        generation_multiplier: 生成倍数（PTC=1, FTC=10）
        test_files: 测试集文件集合
        prompt: 图像生成的文本提示（prompt）
        split_ratios: 数据划分比例 (train, valid, test)
    """
    print(f"\n处理 {subtype} 亚型...")
    print(f"  总文件数: {len(image_files)}")
    print(f"  生成倍数: {generation_multiplier}")
    
    # 创建输出目录
    subtype_output_dir = os.path.join(output_dir, subtype)
    os.makedirs(subtype_output_dir, exist_ok=True)
    
    # 存储原始图像记录和生成图像记录（分开存储）
    original_records = []
    generated_records = []
    
    # 处理每个图像
    for img_info in tqdm(image_files, desc=f"处理 {subtype}"):
        image_path = img_info['image_path']
        mask_path = img_info['mask_path']
        filename = img_info['filename']
        base_name = img_info['base_name']
        
        try:
            # 加载图像和掩码
            init_image = Image.open(image_path).convert("RGB")
            init_image = init_image.resize((512, 512))
            
            mask_image_nd = Image.open(mask_path).convert("L")
            mask_image_nd = mask_image_nd.resize((512, 512))
            
            # 生成背景掩码
            mask_image_bg = generate_background_mask(mask_path)
            mask_image_bg = mask_image_bg.resize((512, 512))
            
            # 保存原始图像
            original_output_path = os.path.join(subtype_output_dir, filename)
            init_image.save(original_output_path)
            relative_path = os.path.join(subtype, filename)
            original_records.append({
                'Path': relative_path,
                'label': label,
                'subtype': subtype
            })
            
            # 生成合成图像
            # 注意：生成图像只能用于 train/valid，不能用于 test
            # 即使原始图像被划分到 test，其生成图像仍然可以用于 train/valid
            if generation_multiplier > 1:
                for i in range(generation_multiplier):
                    seed = random.randint(1, 1000000)
                    generated_image = generate_image_with_tiger_model(
                        pipe_bg, init_image, mask_image_nd, mask_image_bg, 
                        prompt=prompt, seed=seed
                    )
                    
                    if generated_image is not None:
                        gen_filename = f"{base_name}_gen_{i+1}.png"
                        gen_output_path = os.path.join(subtype_output_dir, gen_filename)
                        generated_image.save(gen_output_path)
                        relative_path = os.path.join(subtype, gen_filename)
                        generated_records.append({
                            'Path': relative_path,
                            'label': label,
                            'subtype': subtype
                        })
        
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            continue
    
    # 划分数据集
    # 逻辑说明：
    # 1. 测试集图像（从 xlsx 读取）：已在 get_image_files 中被过滤，不会进入此函数
    # 2. 非测试集原始图像：可以用于 train/valid/test 划分
    # 3. 非测试集生成图像：只能用于 train/valid，不能用于 test
    #    即使某个原始图像被划分到 test，其对应的生成图像仍然可以用于 train/valid
    np.random.seed(42)
    random.seed(42)
    np.random.shuffle(original_records)
    np.random.shuffle(generated_records)
    
    n_original = len(original_records)
    n_train_orig = int(n_original * split_ratios[0])
    n_valid_orig = int(n_original * split_ratios[1])
    
    orig_train = original_records[:n_train_orig]
    orig_valid = original_records[n_train_orig:n_train_orig + n_valid_orig]
    orig_test = original_records[n_train_orig + n_valid_orig:]
    
    # 生成图像只用于 train/valid（不包含 test）
    # 注意：即使原始图像被划分到 test，其生成图像仍然可以用于 train/valid
    n_generated = len(generated_records)
    if n_generated > 0:
        # 按照 train:valid 的比例划分生成图像
        train_valid_ratio = split_ratios[0] / (split_ratios[0] + split_ratios[1])
        n_train_gen = int(n_generated * train_valid_ratio)
        
        gen_train = generated_records[:n_train_gen]
        gen_valid = generated_records[n_train_gen:]
    else:
        gen_train = []
        gen_valid = []
    
    # 合并
    train_records = orig_train + gen_train
    valid_records = orig_valid + gen_valid
    test_records = orig_test  # 测试集只包含原始图像（不包含生成图像）
    
    print(f"  {subtype} 数据划分:")
    print(f"    Train: {len(train_records)} (原始: {len(orig_train)}, 生成: {len(gen_train)})")
    print(f"    Valid: {len(valid_records)} (原始: {len(orig_valid)}, 生成: {len(gen_valid)})")
    print(f"    Test: {len(test_records)} (仅原始)")
    
    return train_records, valid_records, test_records


def create_test_set_from_xlsx(test_file_path, output_dir, ptc_image_dir, ftc_image_dir):
    """
    从 CSV 或 Excel 文件创建测试集（只包含原始图像，不包含生成的）
    
    注意：测试集图像（从文件读取）：
    - 只包含原始图像，不生成合成图像
    - 只用于 test 集，不用于 train/valid
    """
    test_files = load_test_set(test_file_path)
    test_records = []
    
    # 处理 PTC 测试集
    if os.path.exists(ptc_image_dir):
        for img_file in os.listdir(ptc_image_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            base_name = os.path.splitext(img_file)[0]
            if base_name in test_files or img_file in test_files:
                src_path = os.path.join(ptc_image_dir, img_file)
                dst_dir = os.path.join(output_dir, "PTC")
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, img_file)
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    relative_path = os.path.join("PTC", img_file)
                    test_records.append({
                        'Path': relative_path,
                        'label': 0,
                        'subtype': 'PTC'
                    })
    
    # 处理 FTC 测试集
    if os.path.exists(ftc_image_dir):
        for img_file in os.listdir(ftc_image_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            base_name = os.path.splitext(img_file)[0]
            if base_name in test_files or img_file in test_files:
                src_path = os.path.join(ftc_image_dir, img_file)
                dst_dir = os.path.join(output_dir, "FTC")
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, img_file)
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    relative_path = os.path.join("FTC", img_file)
                    test_records.append({
                        'Path': relative_path,
                        'label': 1,
                        'subtype': 'FTC'
                    })
    
    return test_records


def main():
    parser = argparse.ArgumentParser(description="准备 ResNet 训练数据集")
    
    # 输入路径
    parser.add_argument("--ptc_image_dir", type=str, required=True,
                       help="PTC 图像目录路径")
    parser.add_argument("--ftc_image_dir", type=str, required=True,
                       help="FTC 图像目录路径")
    parser.add_argument("--ptc_mask_dir", type=str, required=True,
                       help="PTC 结节掩码目录路径")
    parser.add_argument("--ftc_mask_dir", type=str, required=True,
                       help="FTC 结节掩码目录路径")
    parser.add_argument("--test_xlsx", type=str, required=True,
                       help="测试数据集 xlsx 文件路径")
    
    # Tiger Model 路径
    parser.add_argument("--base_model_path", type=str, required=True,
                       help="Stable Diffusion 基础模型路径 (例如: ./model/pretrain)")
    parser.add_argument("--controlnet_path", type=str, required=True,
                       help="ControlNet 模型路径 (例如: ./model/fine-train-model/controlnet_bg)")
    
    # 输出路径
    parser.add_argument("--output_dir", type=str, default="./dataset",
                       help="输出目录（默认: ./dataset）")
    
    # 生成参数
    parser.add_argument("--ptc_multiplier", type=int, default=1,
                       help="PTC 生成倍数（默认: 1，不生成）")
    parser.add_argument("--ftc_multiplier", type=int, default=10,
                       help="FTC 生成倍数（默认: 10）")
    
    # Prompt 参数
    parser.add_argument("--prompt", type=str, default="black and white definition detail",
                       help="图像生成的文本提示（prompt），用于所有图像（默认: 'black and white definition detail'）")
    parser.add_argument("--ptc_prompt", type=str, default=None,
                       help="PTC 图像生成的文本提示（如果指定，将覆盖 --prompt）")
    parser.add_argument("--ftc_prompt", type=str, default=None,
                       help="FTC 图像生成的文本提示（如果指定，将覆盖 --prompt）")
    
    # 数据划分比例
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="训练集比例（默认: 0.7）")
    parser.add_argument("--valid_ratio", type=float, default=0.15,
                       help="验证集比例（默认: 0.15）")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                       help="测试集比例（默认: 0.15）")
    
    args = parser.parse_args()
    
    # 验证划分比例
    if abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) > 1e-6:
        print("错误：train_ratio + valid_ratio + test_ratio 必须等于 1.0")
        return
    
    # 验证模型路径
    if not os.path.exists(args.base_model_path):
        print(f"错误：基础模型路径不存在: {args.base_model_path}")
        print("提示：请检查路径是否正确，项目中的默认路径为: ./model/pretrain")
        return
    
    model_index = os.path.join(args.base_model_path, "model_index.json")
    if not os.path.exists(model_index):
        print(f"警告：未找到 model_index.json，路径可能不正确: {args.base_model_path}")
        print("提示：基础模型目录应包含 model_index.json 文件")
    
    if not os.path.exists(args.controlnet_path):
        print(f"错误：ControlNet 模型路径不存在: {args.controlnet_path}")
        print("提示：请检查路径是否正确，项目中的默认路径为: ./model/fine-train-model/controlnet_bg")
        return
    
    controlnet_config = os.path.join(args.controlnet_path, "config.json")
    if not os.path.exists(controlnet_config):
        print(f"警告：未找到 config.json，路径可能不正确: {args.controlnet_path}")
        print("提示：ControlNet 目录应包含 config.json 文件")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    csv_dir = os.path.join(args.output_dir, "CSV")
    os.makedirs(csv_dir, exist_ok=True)
    
    # 加载测试集文件列表
    print("加载测试集文件列表...")
    test_files = load_test_set(args.test_xlsx)
    print(f"测试集文件数量: {len(test_files)}")
    
    # 加载 Tiger Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    pipe_bg = load_tiger_model(args.base_model_path, args.controlnet_path, device)
    
    # 获取需要处理的文件列表
    print("\n获取文件列表...")
    ptc_files = get_image_files(args.ptc_image_dir, args.ptc_mask_dir, test_files, "PTC")
    ftc_files = get_image_files(args.ftc_image_dir, args.ftc_mask_dir, test_files, "FTC")
    
    print(f"PTC 训练文件数: {len(ptc_files)}")
    print(f"FTC 训练文件数: {len(ftc_files)}")
    
    # 确定使用的 prompt
    ptc_prompt = args.ptc_prompt if args.ptc_prompt is not None else args.prompt
    ftc_prompt = args.ftc_prompt if args.ftc_prompt is not None else args.prompt
    
    print(f"\n使用的 Prompt:")
    print(f"  PTC: {ptc_prompt}")
    print(f"  FTC: {ftc_prompt}")
    
    # 处理 PTC
    ptc_train, ptc_valid, ptc_test = process_subtype(
        pipe_bg, ptc_files, "PTC", 0, args.output_dir,
        args.ptc_multiplier, test_files, ptc_prompt,
        (args.train_ratio, args.valid_ratio, args.test_ratio)
    )
    
    # 处理 FTC
    ftc_train, ftc_valid, ftc_test = process_subtype(
        pipe_bg, ftc_files, "FTC", 1, args.output_dir,
        args.ftc_multiplier, test_files, ftc_prompt,
        (args.train_ratio, args.valid_ratio, args.test_ratio)
    )
    
    # 合并数据集
    # 注意：ptc_test 和 ftc_test 只包含非测试集的原始图像（从非测试集中划分出来的）
    train_records = ptc_train + ftc_train
    valid_records = ptc_valid + ftc_valid
    test_records = ptc_test + ftc_test
    
    # 从 CSV/Excel 文件添加测试集原始图像
    # 测试集图像（从文件读取）：
    # - 只包含原始图像，不生成合成图像
    # - 只用于 test 集，不用于 train/valid
    test_records_from_xlsx = create_test_set_from_xlsx(
        args.test_xlsx, args.output_dir, args.ptc_image_dir, args.ftc_image_dir
    )
    test_records.extend(test_records_from_xlsx)
    
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_records)} 张（包含原始图像和生成图像）")
    print(f"  验证集: {len(valid_records)} 张（包含原始图像和生成图像）")
    print(f"  测试集: {len(test_records)} 张（仅原始图像，包含从 xlsx 读取的测试集）")
    
    # 保存 CSV 文件
    print("\n保存 CSV 文件...")
    
    # 打乱数据
    np.random.seed(42)
    random.seed(42)
    np.random.shuffle(train_records)
    np.random.shuffle(valid_records)
    np.random.shuffle(test_records)
    
    # 保存
    train_df = pd.DataFrame(train_records)
    valid_df = pd.DataFrame(valid_records)
    test_df = pd.DataFrame(test_records)
    
    train_csv = os.path.join(csv_dir, "train.csv")
    valid_csv = os.path.join(csv_dir, "valid.csv")
    test_csv = os.path.join(csv_dir, "test.csv")
    
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"\n数据集准备完成！")
    print(f"  训练集: {len(train_records)} 张图像 -> {train_csv}")
    print(f"  验证集: {len(valid_records)} 张图像 -> {valid_csv}")
    print(f"  测试集: {len(test_records)} 张图像 -> {test_csv}")
    print(f"\n输出目录: {args.output_dir}")
    print(f"CSV 目录: {csv_dir}")


if __name__ == "__main__":
    main()

