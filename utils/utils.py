"""
通用工具函数
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    """设置随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename):
    """保存检查点"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model=None, optimizer=None, scheduler=None):
    """加载检查点"""
    checkpoint = torch.load(filename, map_location='cpu')

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('best_metric', 0)


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr_scheduler(optimizer, total_epochs, warmup_epochs=5, lr_min=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # 【修复】防止分母为0
            denom = total_epochs - warmup_epochs
            if denom <= 0:
                return 1.0  # 如果没有剩余轮数，保持学习率不变
            progress = (epoch - warmup_epochs) / denom
            return lr_min / (0.9 * lr_min + 0.1) + 0.5 * (1 - 0.9) * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)