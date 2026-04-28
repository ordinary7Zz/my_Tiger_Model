import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import time
import copy
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.model import train_model
import scripts.dataset as DATA
import scripts.config as config
import argparse
from PIL import ImageFile
from datetime import datetime
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(args):
    modelname = args.modelname
    imagepath = args.imagepath
    label_num,subgroup_num = config.THYROID()
    Datasets = DATA.Thyroid_Datasets

    if args.architecture =='resnet':
        net = models.resnet18(pretrained=True)  
        features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Linear(features, args.Class))


    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./modelsaved/train_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建log文件
    log_file = os.path.join(save_dir, f'train_{timestamp}.log')
    
    # 保存训练参数到log文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"模型名称: {modelname}\n")
        f.write(f"架构: {args.architecture}\n")
        f.write(f"学习率: {args.learning_rate}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"训练轮数: {args.num_epochs}\n")
        f.write(f"分类数: {args.Class}\n")
        f.write(f"训练数据: {args.train_data}\n")
        f.write(f"验证数据: {args.valid_data}\n")
        f.write(f"测试数据: {args.test_data if args.test_data else 'None'}\n")
        f.write("=" * 60 + "\n\n")
    
    print(f"📁 保存目录: {save_dir}")
    print(f"📝 日志文件: {log_file}")


    data_transforms = config.Transforms(modelname)

    print("%s Initializing Datasets and Dataloaders..." % modelname)
    
    transformed_datasets = {}
    transformed_datasets['train'] = Datasets(
        path_to_images=imagepath,
        fold=args.train_data,
        PRED_LABEL=label_num,
        transform=data_transforms['train'])
    transformed_datasets['valid'] = Datasets(
        path_to_images=imagepath,
        fold=args.valid_data,
        PRED_LABEL=label_num,
        transform=data_transforms['valid'])
    if args.test_data:
        transformed_datasets['test'] = Datasets(
            path_to_images=imagepath,
            fold=args.test_data,
            PRED_LABEL=label_num,
            transform=data_transforms['valid'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=24)
    dataloaders['valid'] = torch.utils.data.DataLoader(
        transformed_datasets['valid'],
        batch_size=1,
        shuffle=False,
        num_workers=24)
    if 'test' in transformed_datasets:
        dataloaders['test'] = torch.utils.data.DataLoader(
            transformed_datasets['test'],
            batch_size=1,
            shuffle=False,
            num_workers=24)


    if args.modelload_path:
        net.load_state_dict(torch.load('%s' % args.modelload_path , map_location=lambda storage, loc: storage),strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_labels = transformed_datasets['train'].df["label"].astype(int).to_numpy()
    class_counts = np.bincount(train_labels, minlength=args.Class)
    total_samples = class_counts.sum()
    class_weights = np.ones(args.Class, dtype=np.float32)
    nonzero_mask = class_counts > 0
    class_weights[nonzero_mask] = total_samples / (args.Class * class_counts[nonzero_mask])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"训练集类别计数: {class_counts.tolist()}\n")
        f.write(f"训练集类别权重: {class_weights.tolist()}\n\n")
    print(f"训练集类别计数: {class_counts.tolist()}")
    print(f"训练集类别权重: {class_weights.tolist()}")

    net = net.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate, betas=(0.9, 0.99),weight_decay=0.03)


    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    train_model(net, dataloaders, criterion, optimizer, args.num_epochs, modelname, device, save_dir, log_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, default="Thyroid")
    parser.add_argument("--architecture", type=str, choices= ["resnet","densnet","efficientnet"], default="resnet")
    parser.add_argument("--modelload_path", type=str,  default= None)
    parser.add_argument("--imagepath", type=str,  default="./dataset/")
    parser.add_argument("--train_data", type=str, default='thyroid_train')
    parser.add_argument("--valid_data", type=str, default='thyroid_valid')
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--Class", type=int, default=2)
    args = parser.parse_args()
    main(args)

   
