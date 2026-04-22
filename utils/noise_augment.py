# utils/noise_augment.py
import numpy as np
import torch

def add_gaussian_noise_tensor(x, sigma_range=(15, 50)):
    """添加随机强度高斯噪声（适配15/25/50）"""
    sigma = np.random.uniform(*sigma_range) / 255.0
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, -1, 1)

def add_poisson_noise_tensor(x, peak_range=(50, 200)):
    """添加泊松噪声"""
    x_np = (x.cpu().numpy() + 1) / 2  # 反归一化到[0,1]
    peak = np.random.uniform(*peak_range)
    img = np.clip(x_np * peak, 0, peak)
    noisy = np.random.poisson(img) / peak
    # 重新归一化到[-1,1]
    noisy_tensor = torch.from_numpy(noisy).to(x.device)
    noisy_tensor = noisy_tensor * 2 - 1
    return torch.clamp(noisy_tensor, -1, 1)

def add_mixed_noise_tensor(x, gauss_sigma_range=(15, 50), poisson_peak_range=(50, 200)):
    """添加高斯+泊松混合噪声"""
    gauss_noisy = add_gaussian_noise_tensor(x, gauss_sigma_range)
    poisson_noisy = add_poisson_noise_tensor(x, poisson_peak_range)
    ratio = np.random.uniform(0.3, 0.7)
    mixed = ratio * gauss_noisy + (1 - ratio) * poisson_noisy
    return torch.clamp(mixed, -1, 1)

def random_noise_augment(clean_tensor):
    """随机选择噪声类型增强"""
    noise_type = np.random.choice(['gaussian15', 'gaussian25', 'gaussian50', 'poisson', 'mixed'])
    
    if 'gaussian' in noise_type:
        sigma = int(noise_type.replace('gaussian', ''))
        noisy = add_gaussian_noise_tensor(clean_tensor, sigma_range=(sigma, sigma))
    elif noise_type == 'poisson':
        noisy = add_poisson_noise_tensor(clean_tensor)
    elif noise_type == 'mixed':
        noisy = add_mixed_noise_tensor(clean_tensor)
    else:
        noisy = add_gaussian_noise_tensor(clean_tensor)
    
    return noisy, noise_type