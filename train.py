# train.py（最终修复版，解决类型不匹配+所有已知问题）
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import os
import numpy as np
from pathlib import Path

# 导入模型和工具
from models.unet import ConditionalUNet
from models.rddm import ResidualDiffusionModel
from utils.data_loader import get_dataloader
from utils.noise_augment import random_noise_augment

# 噪声类型损失权重
NOISE_TYPE_WEIGHTS = {
    'gaussian15': 1.2,
    'gaussian25': 1.2,
    'gaussian50': 1.2,
    'poisson': 1.0,
    'mixed': 1.0
}

def main():
    parser = argparse.ArgumentParser(description='扩散模型去噪训练（适配BSD68多高斯噪声）')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_dir', type=str, default=r'D:\BISHE\data\train')
    parser.add_argument('--checkpoint_dir', type=str, default=r'D:\BISHE\checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)  # 新手设为0避免多线程错误
    parser.add_argument('--augment', action='store_true', help='随机噪声增强')
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"使用设备: {device}")
    print(f"训练目录: {args.train_dir}")
    print(f"是否启用噪声增强: {args.augment}")
    
    # 初始化模型 + 强制float32
    print("初始化模型...")
    model = ConditionalUNet(
        in_channels=3,
        out_channels=6,
        time_dim=32,
        hidden_dim=32
    ).to(device).float()  # 关键：模型参数转float32
    
    # 初始化扩散模型 + 强制float32
    diffusion = ResidualDiffusionModel(num_steps=1000)
    diffusion.betas = diffusion.betas.float().to(device)
    diffusion.alphas = diffusion.alphas.float().to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.float().to(device)
    diffusion.alphas_cumprod_prev = diffusion.alphas_cumprod_prev.float().to(device)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.float().to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.float().to(device)
    
    # 数据加载器
    print("加载训练数据...")
    train_loader = get_dataloader(
        root_dir=args.train_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    
    # TensorBoard日志
    writer = SummaryWriter(r'D:\BISHE\runs\diffusion-denoise-bsd68')
    
    # 训练循环
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        loss_stats = {nt: 0.0 for nt in NOISE_TYPE_WEIGHTS.keys()}
        loss_stats['total'] = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for batch in pbar:
            # 核心修复：所有输入转float32
            x_0 = batch['x_0'].to(device).float()
            y = batch['y'].to(device).float()
            noise_types = batch['noise_type']
            
            # 数据增强
            if args.augment:
                y_aug = []
                aug_noise_types = []
                for i in range(x_0.shape[0]):
                    aug_img, aug_type = random_noise_augment(x_0[i:i+1])
                    y_aug.append(aug_img.float().to(device))
                    aug_noise_types.append(aug_type)
                y = torch.cat(y_aug, dim=0)
                noise_types = aug_noise_types
            
            # 随机时间步 + 转float32
            t = torch.randint(0, 1000, (x_0.shape[0],), device=device).long()
            
            # 扩散前向过程
            x_t, noise_target, residual_target = diffusion.add_noise(x_0, y, t)
            noise_target = noise_target.float()
            residual_target = residual_target.float()
            
            # 模型预测
            pred = model(x_t, y, t.float())  # 模型需要float32的时间步，这里显式转换
            pred_noise = pred[:, :3, :, :].float()
            pred_residual = pred[:, 3:, :, :].float()
            
            # 多任务损失计算
            batch_loss = 0.0
            for i in range(x_0.shape[0]):
                nt = noise_types[i] if noise_types[i] in NOISE_TYPE_WEIGHTS else 'gaussian25'
                weight = NOISE_TYPE_WEIGHTS[nt]
                
                loss_n = mse_criterion(pred_noise[i:i+1], noise_target[i:i+1])
                loss_r = l1_criterion(pred_residual[i:i+1], residual_target[i:i+1])
                sample_loss = weight * (loss_n + 0.5 * loss_r)
                
                batch_loss += sample_loss
                loss_stats[nt] += sample_loss.item()
                loss_stats['total'] += sample_loss.item()
            
            batch_loss = batch_loss / x_0.shape[0]
            
            # 反向传播
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 进度条更新
            total_loss += batch_loss.item()
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'g15': f'{loss_stats["gaussian15"]/(global_step+1):.4f}',
                'g25': f'{loss_stats["gaussian25"]/(global_step+1):.4f}',
                'g50': f'{loss_stats["gaussian50"]/(global_step+1):.4f}',
                'poi': f'{loss_stats["poisson"]/(global_step+1):.4f}',
                'mix': f'{loss_stats["mixed"]/(global_step+1):.4f}'
            })
            
            # 日志记录
            writer.add_scalar('loss/total', batch_loss.item(), global_step)
            for nt in NOISE_TYPE_WEIGHTS.keys():
                writer.add_scalar(f'loss/{nt}', loss_stats[nt]/(global_step+1), global_step)
            global_step += 1
        
        # epoch结束统计
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1} 平均损失: {avg_loss:.6f}')
        for nt in ['gaussian15', 'gaussian25', 'gaussian50', 'poisson', 'mixed']:
            print(f'  - {nt} 损失: {loss_stats[nt]/len(train_loader):.6f}')
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            ckpt_path = Path(args.checkpoint_dir) / f'model_epoch_{epoch+1}_bsd68.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f'✓ 检查点保存: {ckpt_path}')
    
    writer.close()
    print("\n✓ 训练完成！")

if __name__ == '__main__':
    main()