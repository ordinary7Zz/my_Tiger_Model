import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from random import sample
from sklearn.metrics import roc_auc_score
from .multiAUC import Metric

def bootstrap_auc(label, output, classes, bootstraps=5, fold_size=1000):
    """
    Bootstrap AUC calculation for binary classification
    """
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    if not isinstance(output, np.ndarray):
        output = np.array(output)
    
    # 判断是二分类（一维数组）还是多分类（二维数组）
    is_binary = (label.ndim == 1) and (output.ndim == 1)
    
    if is_binary:
        # 二分类：直接使用 roc_auc_score
        statistics = np.zeros((1, bootstraps))
        df = pd.DataFrame({'y': label, 'pred': output})
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        
        if len(df_pos) == 0 or len(df_neg) == 0:
            return np.array([[0.5] * bootstraps])
        
        prevalence = len(df_pos) / len(df)
        
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)
            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            
            if len(np.unique(y_sample)) < 2:
                score = 0.5
            else:
                try:
                    score = roc_auc_score(y_sample, pred_sample)
                except:
                    score = 0.5
            statistics[0][i] = score
    else:
        # 多分类：使用 Metric.auROC()
        statistics = np.zeros((len(classes), bootstraps))
        for c in range(len(classes)):
            for i in range(bootstraps):
                L = []
                for k in range(len(label)):
                    L.append([output[k], label[k]])
                if fold_size <= len(L):
                    X = sample(L, fold_size)
                else:
                    X = sample(L, len(L))
                for b in range(len(X)):
                    if b == 0:
                        Output = np.array([X[b][0]])
                        Label = np.array([X[b][1]])
                    else:
                        Output = np.concatenate((Output, np.array([X[b][0]])), axis=0)
                        Label = np.concatenate((Label, np.array([X[b][1]])), axis=0)
                
                try:
                    myMetic = Metric(Output, Label)
                    AUROC1, auc = myMetic.auROC()
                    statistics[c][i] = AUROC1
                except:
                    statistics[c][i] = 0.5
    return statistics

def plotimage(train_history, valid_history, test_history, metric_name, modelname, save_dir=None):
    """
    Plot training curves for train/valid/test
    """
    if save_dir is None:
        save_dir = './result/%s' % modelname
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_history) + 1)
    
    plt.plot(epochs, train_history, 'b-', label='Train', linewidth=2)
    if valid_history:
        plt.plot(epochs, valid_history, 'g-', label='Valid', linewidth=2)
    if test_history:
        plt.plot(epochs, test_history, 'r-', label='Test', linewidth=2)
    
    plt.title(f'{metric_name} Curve - {modelname}', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, '%s_%s.png' % (modelname, metric_name))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def result_csv(train_history, valid_history, test_history, modelname, save_dir=None):
    """
    Save training results to CSV file
    """
    if save_dir is None:
        save_dir = './result/%s' % modelname
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    max_len = max(len(train_history), len(valid_history), len(test_history))
    
    # Pad shorter lists with NaN
    train_padded = train_history + [np.nan] * (max_len - len(train_history))
    valid_padded = valid_history + [np.nan] * (max_len - len(valid_history))
    test_padded = test_history + [np.nan] * (max_len - len(test_history))
    
    df = pd.DataFrame({
        'Epoch': range(1, max_len + 1),
        'Train': train_padded,
        'Valid': valid_padded,
        'Test': test_padded
    })
    
    csv_path = os.path.join(save_dir, '%s_results.csv' % modelname)
    df.to_csv(csv_path, index=False)


def plot_combined_metrics(train_loss_history, train_auc_history, test_auc_history, save_dir, modelname, epoch):
    """
    将loss、train auroc和test auroc画在一张图上
    
    Args:
        train_loss_history: 训练损失历史
        train_auc_history: 训练AUC历史
        test_auc_history: 测试AUC历史
        save_dir: 保存目录
        modelname: 模型名称
        epoch: 当前epoch
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 确保所有历史记录长度一致
    max_len = max(len(train_loss_history), len(train_auc_history), len(test_auc_history))
    epochs = range(1, max_len + 1)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 左y轴：Loss
    loss_color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=loss_color, fontsize=12)
    if len(train_loss_history) > 0:
        ax1.plot(epochs[:len(train_loss_history)], train_loss_history, '-', label='Train Loss', linewidth=2, color=loss_color)
    ax1.tick_params(axis='y', labelcolor=loss_color)
    ax1.grid(True, alpha=0.3)
    
    # 右y轴：AUC
    ax2 = ax1.twinx()
    auc_color = 'tab:blue'
    ax2.set_ylabel('AUROC', color=auc_color, fontsize=12)
    if len(train_auc_history) > 0:
        ax2.plot(epochs[:len(train_auc_history)], train_auc_history, '-', label='Train AUROC', linewidth=2, color='tab:blue')
    if len(test_auc_history) > 0:
        ax2.plot(epochs[:len(test_auc_history)], test_auc_history, '-', label='Test AUROC', linewidth=2, color='tab:green')
    ax2.tick_params(axis='y', labelcolor=auc_color)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.title(f'Training Metrics - {modelname} (Epoch {epoch})', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, '%s_combined_metrics.png' % modelname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_recall_metrics(train_loss_history, train_recall_history, test_recall_history, save_dir, modelname, epoch):
    """
    将loss、train recall和test recall画在一张图上
    
    Args:
        train_loss_history: 训练损失历史
        train_recall_history: 训练Recall历史
        test_recall_history: 测试Recall历史
        save_dir: 保存目录
        modelname: 模型名称
        epoch: 当前epoch
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 确保所有历史记录长度一致
    max_len = max(len(train_loss_history), len(train_recall_history), len(test_recall_history))
    epochs = range(1, max_len + 1)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 左y轴：Loss
    loss_color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=loss_color, fontsize=12)
    if len(train_loss_history) > 0:
        ax1.plot(epochs[:len(train_loss_history)], train_loss_history, '-', label='Train Loss', linewidth=2, color=loss_color)
    ax1.tick_params(axis='y', labelcolor=loss_color)
    ax1.grid(True, alpha=0.3)
    
    # 右y轴：Recall
    ax2 = ax1.twinx()
    recall_color = 'tab:blue'
    ax2.set_ylabel('Recall', color=recall_color, fontsize=12)
    if len(train_recall_history) > 0:
        ax2.plot(epochs[:len(train_recall_history)], train_recall_history, '-', label='Train Recall', linewidth=2, color='tab:blue')
    if len(test_recall_history) > 0:
        ax2.plot(epochs[:len(test_recall_history)], test_recall_history, '-', label='Test Recall', linewidth=2, color='tab:green')
    ax2.tick_params(axis='y', labelcolor=recall_color)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.title(f'Training Metrics (Loss & Recall) - {modelname} (Epoch {epoch})', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, '%s_combined_recall_metrics.png' % modelname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

