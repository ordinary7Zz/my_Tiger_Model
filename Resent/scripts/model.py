import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, classification_report
from torch.autograd import Variable
from scripts.multiAUC import Metric
import numpy
from tqdm import tqdm
from random import sample
from scripts.plot import bootstrap_auc, result_csv, plotimage, plot_combined_metrics, plot_combined_recall_metrics
import pynvml
pynvml.nvmlInit()
from prettytable import PrettyTable


def train_model(model, dataloaders, criterion, optimizer, num_epochs, modelname, device, save_dir=None, log_file=None):
    """
    训练模型
    
    Args:
        model: 模型
        dataloaders: 数据加载器字典
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        modelname: 模型名称
        device: 设备
        save_dir: 保存目录（带时间戳）
        log_file: 日志文件路径
    """
    global VAL_auc,TEST_auc
    since = time.time()
    train_loss_history, valid_loss_history, test_loss_history= [], [], []
    test_maj_history, test_min_history = [], []
    train_auc_history, val_auc_history, test_auc_history = [], [], []
    train_recall_history, val_recall_history, test_recall_history = [], [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 日志记录函数
    def log_print(*args, **kwargs):
        """同时打印到控制台和日志文件"""
        print(*args, **kwargs)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                # 将print的内容转换为字符串
                message = ' '.join(str(arg) for arg in args)
                f.write(message + '\n')
                f.flush()
    
    # 设置默认保存目录
    if save_dir is None:
        save_dir = f'./modelsaved/{modelname}'
        os.makedirs(save_dir, exist_ok=True)
    for epoch in range(num_epochs):
        start = time.time()

        log_print('{}  Epoch {}/{}  {}'.format('-' * 30, epoch, num_epochs - 1, '-' * 30))
        for phase in ['train','valid', 'test']:
            if phase == 'train' and epoch != 0:
                model.train()
            else:
                model.eval()
            running_loss,running_corrects,prob_all, label_all = [], [], [], []
            Output, Label = [], []
            G, G1 = [], []
            Batch = len(dataloaders[phase])
            outputs_out = []
            
            with tqdm(range(len(dataloaders[phase])),desc='%s' % phase, ncols=100) as t:
                if epoch == 0 :
                    t.set_postfix(L = 0.000, usedMemory = 0)

                for data in dataloaders[phase]:
                    inputs, labels, sub = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad(set_to_none=True) 
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train' and epoch != 0:
                            loss.backward()
                            optimizer.step()
                    running_loss.append(loss.item())
                    running_corrects.append((preds.cpu().detach() == labels.cpu().detach()).numpy())

                    # 将 logits 转换为概率（softmax）
                    outputs_prob = torch.softmax(outputs, dim=1).cpu().detach().numpy()
                    labels_np = labels.cpu().detach().numpy()
                    
                    # 对于二分类，使用第二个类别的概率
                    prob_all.extend(outputs_prob[:, 1])
                    label_all.extend(labels_np)
                    Output.extend(outputs_prob[:, 1])
                    Label.extend(labels_np)
                    outputs_out.extend(outputs_prob)

                    """
                    B:batch 
                    L:Loss
                    maj: Maj group AUC
                    min: Min group AUC
                    n: NVIDIA Memory used
                    """   
                    gpu_device = pynvml.nvmlDeviceGetHandleByIndex(0)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).total
                    usedMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used
                    usedMemory = usedMemory/meminfo              
                    t.set_postfix(loss = loss.data.item(), usedMemory = usedMemory)  # 
                    t.update()



            # num = len(label_all)
            # auc = roc_auc_score(label_all, prob_all)
            # epoch_loss = np.mean(running_loss)
            # label_all = np.array(label_all)
            # prob_all = np.array(prob_all)
            # statistics = bootstrap_auc(label_all, prob_all, [0,1,3,4,5])
            # max_auc = np.max(statistics, axis=1).max()
            # min_auc = np.min(statistics, axis=1).max()
            # print('{} --> Num: {} Loss: {:.4f}  AUROC: {:.4f} ({:.2f} ~ {:.2f})'.format(
            #     phase, num, epoch_loss, auc, min_auc, max_auc ))
            if modelname == "Thyroid_PF":
                if len(Label) > 0 and len(Output) > 0:
                    Label = np.array(Label)
                    Output = np.array(Output)
                    
                    try:
                        data_auc = roc_auc_score(Label, Output)
                        Data_auc_maj = data_auc
                        Data_auc_min = data_auc
                    except:
                        data_auc = 0.5
                        Data_auc_maj = 0.5
                        Data_auc_min = 0.5
                    
                    epoch_loss = np.mean(running_loss)
                    try:
                        statistics = bootstrap_auc(Label, Output, [0, 1])
                        max_auc = np.max(statistics, axis=1).max()
                        min_auc = np.min(statistics, axis=1).max()
                    except:
                        max_auc = data_auc
                        min_auc = data_auc
                    
                    if len(G) == 0 and phase == "train":
                        G1.append(0)
                    elif phase == "train":
                        G1.append(sum(G)/len(G) if len(G) > 0 else 0)
                    
                    # 计算 Recall（对于 Thyroid_PF 模型）
                    pred_labels = (Output >= 0.5).astype(int)
                    try:
                        if len(np.unique(Label)) < 2:
                            data_recall = 0.0
                        else:
                            data_recall = recall_score(Label, pred_labels, average='binary', zero_division=0)
                    except Exception as e:
                        data_recall = 0.0
                    
                    log_print('{} --> Num: {} Loss: {:.4f}  Gamma: {:.4f} AUROC: {:.4f} ({:.2f} ~ {:.2f}) (Maj {:.4f}, Min {:.4f})  Recall: {:.4f}'.format(
                        phase, len(outputs_out), epoch_loss, G1[-1] if len(G1) > 0 else 0, data_auc, min_auc, max_auc, Data_auc_maj, Data_auc_min, data_recall))
                else:
                    epoch_loss = np.mean(running_loss)
                    auc = 0.0
                    data_auc = 0.0

            else:
                if len(Label) > 0 and len(Output) > 0:
                    Label = np.array(Label)
                    Output = np.array(Output)
                    
                    # 调试信息：检查标签分布
                    unique_labels = np.unique(Label)
                    label_counts = np.bincount(Label.astype(int))
                    output_min, output_max = Output.min(), Output.max()
                    output_mean, output_std = Output.mean(), Output.std()
                    
                    # 只在第一个 epoch 的第一个 phase 打印详细信息
                    if epoch == 0 and phase == 'train':
                        log_print(f"   DEBUG: Label unique values: {unique_labels}")
                        log_print(f"   DEBUG: Label counts: {label_counts}")
                        log_print(f"   DEBUG: Output range: [{output_min:.4f}, {output_max:.4f}], mean={output_mean:.4f}, std={output_std:.4f}")
                    
                    # 对于二分类，直接使用 roc_auc_score
                    try:
                        if len(unique_labels) < 2:
                            log_print(f"⚠️  ERROR: Only {len(unique_labels)} unique label(s) found: {unique_labels}. Cannot calculate AUC.")
                            log_print(f"   This usually means all samples have the same label. Check your CSV files!")
                            data_auc = 0.5
                        elif np.all(Output == Output[0]):
                            log_print(f"⚠️  WARNING: All predictions are identical ({Output[0]:.4f}). Model may not be learning.")
                            data_auc = 0.5
                        else:
                            data_auc = roc_auc_score(Label, Output)
                    except Exception as e:
                        log_print(f"⚠️  ERROR calculating AUC: {e}")
                        log_print(f"   Label unique values: {unique_labels}, Label counts: {label_counts}")
                        data_auc = 0.5
                    
                    # 计算 bootstrap AUC
                    try:
                        statistics = bootstrap_auc(Label, Output, [0, 1])
                        max_auc = np.max(statistics, axis=1).max()
                        min_auc = np.min(statistics, axis=1).max()
                    except:
                        max_auc = data_auc
                        min_auc = data_auc
                    
                    auc = data_auc
                    epoch_loss = np.mean(running_loss)
                    data_auc_maj = data_auc
                    data_auc_min = data_auc
                    
                    # 计算 Recall
                    # 将概率转换为预测类别（阈值0.5）
                    pred_labels = (Output >= 0.5).astype(int)
                    try:
                        if len(unique_labels) < 2:
                            data_recall = 0.0
                        else:
                            # 计算 recall（对于二分类，使用正类的recall）
                            data_recall = recall_score(Label, pred_labels, average='binary', zero_division=0)
                    except Exception as e:
                        log_print(f"⚠️  ERROR calculating Recall: {e}")
                        data_recall = 0.0
                    
                    recall = data_recall
                    log_print('{} --> Num: {} Loss: {:.4f}  AUROC: {:.4f} ({:.2f} ~ {:.2f})  Recall: {:.4f}'.format(
                        phase, len(outputs_out), epoch_loss, data_auc, min_auc, max_auc, data_recall))
                else:
                    epoch_loss = np.mean(running_loss)
                    auc = 0.0
                    data_auc = 0.0
                    recall = 0.0
                    data_recall = 0.0
                    log_print('{} --> Num: {} Loss: {:.4f}  AUROC: N/A  Recall: N/A'.format(
                        phase, len(outputs_out), epoch_loss))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                if len(Label) > 0 and len(Output) > 0:
                    current_auc = auc if 'auc' in locals() else (data_auc if 'data_auc' in locals() else 0.0)
                    current_recall = recall if 'recall' in locals() else (data_recall if 'data_recall' in locals() else 0.0)
                    train_auc_history.append(current_auc)
                    train_recall_history.append(current_recall)
                else:
                    train_auc_history.append(0.0)
                    train_recall_history.append(0.0)

            if phase == 'valid':
                valid_loss_history.append(epoch_loss)
                if len(Label) > 0 and len(Output) > 0:
                    current_auc = auc if 'auc' in locals() else (data_auc if 'data_auc' in locals() else 0.0)
                    current_recall = recall if 'recall' in locals() else (data_recall if 'data_recall' in locals() else 0.0)
                    val_auc_history.append(current_auc)
                    val_recall_history.append(current_recall)
                else:
                    val_auc_history.append(0.0)
                    val_recall_history.append(0.0)

            if phase == 'test':
                test_loss_history.append(epoch_loss)
                if len(Label) > 0 and len(Output) > 0:
                    current_auc = auc if 'auc' in locals() else (data_auc if 'data_auc' in locals() else 0.0)
                    current_recall = recall if 'recall' in locals() else (data_recall if 'data_recall' in locals() else 0.0)
                    test_auc_history.append(current_auc)
                    test_recall_history.append(current_recall)
                else:
                    test_auc_history.append(0.0)
                    test_recall_history.append(0.0)
      
            if phase == 'valid' and len(train_auc_history) > 0 and train_auc_history[-1] >= 0.9:
                if len(val_auc_history) > 0 and len(test_auc_history) > 0:
                    if val_auc_history[-1] >= max(val_auc_history) or test_auc_history[-1] >= max(test_auc_history):
                        log_print("In epoch %d, better AUC(%.3f) and save model. " % (epoch, float(val_auc_history[-1])))
                        PATH = os.path.join(save_dir, 'e%d_%s_V%.3fT%.3f_best.pth' % (epoch, modelname, val_auc_history[-1], test_auc_history[-1]))
                        torch.save(model.state_dict(), PATH)
                        best_model_wts = copy.deepcopy(model.state_dict())
        
        # 每个epoch保存权重
        if len(val_auc_history) > 0 and len(test_auc_history) > 0:
            val_auc_str = "%.3f" % val_auc_history[-1] if len(val_auc_history) > 0 else "0.000"
            test_auc_str = "%.3f" % test_auc_history[-1] if len(test_auc_history) > 0 else "0.000"
            PATH = os.path.join(save_dir, 'epoch_%03d_%s_V%s_T%s.pth' % (epoch, modelname, val_auc_str, test_auc_str))
            torch.save(model.state_dict(), PATH)
            log_print("✅ Epoch %d: 模型权重已保存到 %s" % (epoch, PATH))
    
        log_print("learning rate = %.6f     time: %.1f sec" % (optimizer.param_groups[-1]['lr'], time.time() - start))
        if epoch != 0:
            scheduler.step()
        log_print()

        # 绘制loss、train auroc和test auroc在一张图上
        plot_combined_metrics(train_loss_history, train_auc_history, test_auc_history, save_dir, modelname, epoch)
        
        # 绘制loss、train recall和test recall在一张图上
        plot_combined_recall_metrics(train_loss_history, train_recall_history, test_recall_history, save_dir, modelname, epoch)
        
        # 原有的单独绘图（保留）
        plotimage(train_auc_history, val_auc_history, test_auc_history,"AUC", modelname, save_dir)        
        plotimage(train_loss_history, valid_loss_history, test_loss_history,"Loss", modelname, save_dir)
        plotimage(train_recall_history, val_recall_history, test_recall_history,"Recall", modelname, save_dir)
        result_csv(train_auc_history, val_auc_history, test_auc_history, modelname, save_dir)
        
    time_elapsed = time.time() - since
    log_print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)