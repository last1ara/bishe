import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class MultiNoiseDenoisingDataset(Dataset):
    """BSD68多噪声类型去噪数据集加载器"""
    def __init__(self, root_dir, is_train=True):
        """
        Args:
            root_dir: 数据集根目录（如 D:\\BISHE\\data\\train）
            is_train: 是否为训练集（True/False）
        """
        # 修复路径转义问题（使用原始字符串）
        self.root_dir = Path(root_dir) if root_dir else (
            Path(r'D:\BISHE\data\train') if is_train else Path(r'D:\BISHE\data\test')
        )
        self.is_train = is_train
        
        # 数据预处理（强制转float32，解决类型不匹配）
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 统一图像尺寸
            transforms.ToTensor(),          # [0,255] -> [0,1] (float32)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1,1]
            lambda x: x.float()  # 显式强制转为float32
        ])
        
        # 清晰图目录
        self.clean_dir = self.root_dir / 'original'
        if not self.clean_dir.exists():
            raise FileNotFoundError(f"清晰图目录不存在: {self.clean_dir}")
        
        # 噪声类型目录（适配已有高斯+生成的泊松/混合）
        self.noise_dirs = {
            'gaussian15': self.root_dir / 'gaussian15',
            'gaussian25': self.root_dir / 'gaussian25',
            'gaussian50': self.root_dir / 'gaussian50',
            'poisson': self.root_dir / 'poisson',
            'mixed': self.root_dir / 'mixed'
        }
        
        # 过滤存在的噪声目录
        self.valid_noise_types = [nt for nt, dir_path in self.noise_dirs.items() if dir_path.exists()]
        if not self.valid_noise_types:
            raise ValueError(f"未找到任何噪声目录，检查路径: {self.root_dir}")
        
        # 加载清晰图列表（按名称排序）
        self.clean_paths = sorted([p for p in self.clean_dir.glob('*.png') if p.is_file()])
        if len(self.clean_paths) == 0:
            raise ValueError(f"清晰图目录下无PNG文件: {self.clean_dir}")
        
        # 过滤有效样本（确保有对应的噪声图）
        self.valid_samples = []
        for clean_path in self.clean_paths:
            # 检查该清晰图是否有至少一种噪声图
            has_noise = any(
                (self.noise_dirs[nt] / clean_path.name).exists() 
                for nt in self.valid_noise_types
            )
            if has_noise:
                self.valid_samples.append(clean_path)
        
        print(f"{'训练' if is_train else '测试'}集加载完成: "
              f"有效样本数={len(self.valid_samples)}, "
              f"可用噪声类型={self.valid_noise_types}")

    def __len__(self):
        """数据集长度"""
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """获取单样本（清晰图 + 随机噪声图）"""
        # 1. 加载清晰图
        clean_path = self.valid_samples[idx]
        try:
            clean_img = Image.open(clean_path).convert('RGB')
            clean_tensor = self.transform(clean_img)
        except Exception as e:
            raise RuntimeError(f"加载清晰图失败: {clean_path}, 错误: {e}")
        
        # 2. 随机选择噪声类型
        noise_type = np.random.choice(self.valid_noise_types)
        noise_dir = self.noise_dirs[noise_type]
        noisy_path = noise_dir / clean_path.name
        
        # 3. 加载噪声图（若当前类型不存在，选第一个可用类型）
        if not noisy_path.exists():
            noise_type = self.valid_noise_types[0]
            noise_dir = self.noise_dirs[noise_type]
            noisy_path = noise_dir / clean_path.name
        
        try:
            noisy_img = Image.open(noisy_path).convert('RGB')
            noisy_tensor = self.transform(noisy_img)
        except Exception as e:
            raise RuntimeError(f"加载噪声图失败: {noisy_path}, 错误: {e}")
        
        # 4. 返回样本（所有张量均为float32）
        return {
            'x_0': clean_tensor,          # 清晰图 [3, 256, 256] (float32)
            'y': noisy_tensor,            # 噪声图 [3, 256, 256] (float32)
            'noise_type': noise_type,     # 噪声类型（字符串）
            'filename': clean_path.name   # 文件名
        }

def get_dataloader(root_dir, batch_size=4, num_workers=0, is_train=True):
    """
    获取数据加载器（新手友好：num_workers默认0，避免多线程错误）
    Args:
        root_dir: 数据集根目录
        batch_size: 批次大小
        num_workers: 工作线程数（新手建议设0）
        is_train: 是否为训练集
    Returns:
        DataLoader实例
    """
    dataset = MultiNoiseDenoisingDataset(root_dir=root_dir, is_train=is_train)
    
    # 数据加载器配置
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,  # 训练集打乱，测试集不打乱
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True if is_train else False  # 训练集丢弃最后不完整批次
    )
    
    return dataloader

# 测试数据加载器（可选）
if __name__ == "__main__":
    # 测试训练集加载
    train_loader = get_dataloader(
        root_dir=r'D:\BISHE\data\train',
        batch_size=1,
        num_workers=0,
        is_train=True
    )
    
    # 遍历一个批次
    for batch in train_loader:
        print(f"清晰图形状: {batch['x_0'].shape}, 类型: {batch['x_0'].dtype}")
        print(f"噪声图形状: {batch['y'].shape}, 类型: {batch['y'].dtype}")
        print(f"噪声类型: {batch['noise_type']}")
        print(f"文件名: {batch['filename']}")
        break