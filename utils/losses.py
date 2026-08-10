"""
UniLFS 损失函数 (完整版)

包含:
1. 对数域 Retinex 重构损失
2. 边缘感知光照平滑损失
3. 纹理保持损失 (梯度空间)
4. 颜色恒常性损失
5. 曝光控制损失
6. 感知损失 (VGG)
7. 总变分损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """
    VGG 感知损失
    使用预训练的 VGG-16 提取特征，比较特征图差异
    """
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        self.layers = layers

        # 加载预训练 VGG
        # 加载完整官方预训练vgg16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # 只取出features卷积部分用于感知损失
        self.vgg_features = vgg.features

        self.vgg_layers = nn.ModuleList()
        self.layer_names = []

        for name, module in vgg._modules.items():
            if isinstance(module, nn.ReLU):
                module = nn.ReLU(inplace=False)
            self.vgg_layers.append(module)
            self.layer_names.append(f'conv{name}')

            if f'relu{name.split("_")[0]}' in layers:
                break

        # 冻结参数
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # 归一化
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
            # ===== 检查输入 =====
        if torch.isnan(I).any() or torch.isinf(I).any():
            print("⚠️  Input contains NaN/Inf, returning zero loss")
            return torch.tensor(0.0, device=I.device), {k: 0.0 for k in ['recon', 'smooth', 'texture', 'color', 'exp']}
    
        if I.max() > 1.0 or I.min() < 0.0:
            print(f"⚠️  Input out of range: min={I.min():.4f}, max={I.max():.4f}")
            I = torch.clamp(I, 0.0, 1.0)
    # =========================
    
        """
        Args:
            x, y: [B, 3, H, W] 图像 (范围 [0,1])
        Returns:
            loss: 感知损失
        """
        # 归一化到 ImageNet 统计
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        loss = 0
        for layer in self.vgg_layers:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)

        return loss / len(self.vgg_layers)


class UniLFSLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_perceptual = hasattr(cfg, 'w_perceptual') and cfg.w_perceptual > 0
        if self.use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()

    def forward(self, outputs, I):
        log_I = outputs['log_I']
        log_R = outputs['log_R']
        log_X = outputs['log_X']
        I_enh = outputs['I_enhanced']
        delta_X = outputs.get('delta_X', torch.zeros_like(I))
        
        B, _, H, W = I.shape
        
        # ===== 1. 结构一致性损失（替代原来的 L_recon） =====
        # 不是 MSE(delta_X, 0)，而是约束增强图像与输入的结构相似
        def gradient_loss(x, y):
            gx_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            gy_x = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
            gx_y = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
            gy_y = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
            gx_x = F.pad(gx_x, (0, 1, 0, 0), value=0)
            gy_x = F.pad(gy_x, (0, 0, 0, 1), value=0)
            gx_y = F.pad(gx_y, (0, 1, 0, 0), value=0)
            gy_y = F.pad(gy_y, (0, 0, 0, 1), value=0)
            return F.l1_loss(gx_x, gx_y) + F.l1_loss(gy_x, gy_y)
        
        L_structure = F.l1_loss(I_enh, torch.exp(log_I) - self.cfg.eps) + 0.1 * gradient_loss(I_enh, torch.exp(log_I) - self.cfg.eps)
        
        #print(f"DEBUG: I_enh mean={I_enh.mean():.4f}, target mean={(torch.exp(log_I)-self.cfg.eps).mean():.4f}")
        #print(f"DEBUG: L_structure raw = {F.l1_loss(I_enh, torch.exp(log_I) - self.cfg.eps):.6f}")
        # 额外的 delta_X 稀疏性约束：让 delta_X 只在边缘区域有响应
        # 这样 delta_X 不会归零，而是学习增强边缘
        edge_mask = gradient_loss(I, torch.zeros_like(I)) > 0.1
        L_delta_sparse = (delta_X.abs() * (~edge_mask).float()).mean() - (delta_X.abs() * edge_mask.float()).mean()
        L_delta_sparse = torch.tanh(L_delta_sparse)  # 稳定训练
        
        # ===== 2. 边缘感知光照平滑损失 =====
        # 在边缘区域允许光照突变，在平滑区域约束光照平滑
        grad_Rx = torch.abs(log_R[:, :, :, 1:] - log_R[:, :, :, :-1])
        grad_Ry = torch.abs(log_R[:, :, 1:, :] - log_R[:, :, :-1, :])

        # 计算边缘权重 (使用输入图像的梯度)
        gray = log_I.mean(dim=1, keepdim=True)
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=I.device).float().view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=I.device).float().view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        edge = torch.exp(-10 * torch.sqrt(gx ** 2 + gy ** 2 + 1e-8))

        L_smooth = (edge[:, :, :, :-1] * grad_Rx).mean() + (edge[:, :, :-1, :] * grad_Ry).mean()

        # ===== 3. 纹理保持损失 =====
        # 约束 log_X 与 log_I 的梯度结构一致
        def grad_magnitude(x):
            gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
            gx = F.pad(gx, (0, 1, 0, 0), mode='constant', value=0)
            gy = F.pad(gy, (0, 0, 0, 1), mode='constant', value=0)
            return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        grad_X = grad_magnitude(log_X)
        grad_I = grad_magnitude(log_I)
        L_texture = F.l1_loss(grad_X, grad_I)

        # ===== 4. 颜色恒常性损失 =====
        # 约束增强图像的三通道均值接近
        mean_rgb = I_enh.mean(dim=(2, 3))  # [B, 3]
        L_color = (
            (mean_rgb[:, 0] - mean_rgb[:, 1]).pow(2).mean() +
            (mean_rgb[:, 1] - mean_rgb[:, 2]).pow(2).mean() +
            (mean_rgb[:, 2] - mean_rgb[:, 0]).pow(2).mean()
        ) / 3

        # ===== 5. 曝光控制损失 =====
        # 约束局部块的平均亮度接近 0.6
        patch_size = 16
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        if num_patches_h > 0 and num_patches_w > 0:
            L_exp = 0.0
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    patch = I_enh[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                    lum = (0.299 * patch[:, 0] + 0.587 * patch[:, 1] + 0.114 * patch[:, 2]).mean()
                    L_exp += (lum - 0.6).pow(2)
            L_exp = L_exp / (num_patches_h * num_patches_w + 1e-8)
        else:
            L_exp = torch.tensor(0.0, device=I.device)

        # ===== 6. 总变分损失 (平滑约束) =====
        L_tv = (
            torch.abs(I_enh[:, :, :, 1:] - I_enh[:, :, :, :-1]).mean() +
            torch.abs(I_enh[:, :, 1:, :] - I_enh[:, :, :-1, :]).mean()
        )

        # ===== 7. 感知损失 (可选) =====
        if self.use_perceptual:
            L_perceptual = self.perceptual_loss(I_enh, torch.exp(log_I) - self.cfg.eps)
        else:
            L_perceptual = torch.tensor(0.0, device=I.device)

        # ===== 总损失 =====
        total = (
                    self.cfg.w_recon * L_structure +
                    self.cfg.w_smooth * L_smooth +
                    self.cfg.w_texture * L_texture +
                    self.cfg.w_color * L_color +
                    self.cfg.w_exp * L_exp +
                    0.01 * L_delta_sparse  # 新增
                )

        loss_dict = {
            'structur': L_structure.item(),
            'smooth': L_smooth.item(),
            'texture': L_texture.item(),
            'color': L_color.item(),
            'exp': L_exp.item(),
            'tv': L_tv.item(),
            'perceptual': L_perceptual.item() if self.use_perceptual else 0,
        }

        return total, loss_dict