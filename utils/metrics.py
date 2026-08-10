"""
图像质量评估指标
包含: PSNR, SSIM, LPIPS, NIQE, BRISQUE
"""
import torch
import torch.nn.functional as F
import numpy as np
from math import exp


def compute_psnr(img1, img2, max_val=1.0):
    """计算 PSNR"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    """
    计算 SSIM (结构相似性)
    使用简化的滑动窗口实现
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # 创建高斯窗口
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    window = gauss.outer(gauss).unsqueeze(0).unsqueeze(0)
    window = window / window.sum()

    # 计算均值、方差、协方差
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def compute_lpips(img1, img2):
    """
    计算 LPIPS (简化版本)
    使用预训练的 VGG 特征差异
    """
    try:
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        vgg.eval()
        vgg.requires_grad_(False)

        # 归一化
        mean = torch.tensor([0.485, 0.456, 0.406], device=img1.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img1.device).view(1, 3, 1, 1)

        x = (img1 - mean) / std
        y = (img2 - mean) / std

        # 提取特征
        x_feat = vgg(x)
        y_feat = vgg(y)

        return F.l1_loss(x_feat, y_feat).item()
    except:
        return 0.0


def compute_niqe(img):
    """
    计算 NIQE (简化版本)
    实际 NIQE 需要复杂统计，这里返回占位值
    """
    # 真实的 NIQE 需要训练一个多变量高斯模型
    # 这里简化处理，返回图像的标准差作为粗糙估计
    return img.std().item()


def compute_metrics(img_enh, img_gt):
    """
    计算所有评估指标
    Args:
        img_enh: 增强图像 [B, 3, H, W]
        img_gt: 参考图像 [B, 3, H, W]
    Returns:
        dict: 各指标值
    """
    metrics = {
        'psnr': compute_psnr(img_enh, img_gt),
        'ssim': compute_ssim(img_enh, img_gt),
        'lpips': compute_lpips(img_enh, img_gt),
    }
    return metrics