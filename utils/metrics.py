# utils/metrics.py
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_lpips import LPIPS

class MetricsCalculator:
    """性能评估指标计算"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_fn = LPIPS(net='alex').to(device).eval()
    
    def psnr(self, y_true, y_pred):
        """计算PSNR"""
        # 转换回0-1范围用于计算
        if y_true.dim() == 4:  # Batch
            psnr_list = []
            for i in range(y_true.shape[0]):
                p = peak_signal_noise_ratio(
                    (y_true[i].permute(1,2,0).cpu().numpy() + 1) / 2,
                    (y_pred[i].permute(1,2,0).cpu().numpy() + 1) / 2,
                    data_range=1.0
                )
                psnr_list.append(p)
            return np.mean(psnr_list)
        else:
            return peak_signal_noise_ratio(
                (y_true.permute(1,2,0).cpu().numpy() + 1) / 2,
                (y_pred.permute(1,2,0).cpu().numpy() + 1) / 2,
                data_range=1.0
            )
    
    def ssim(self, y_true, y_pred):
        """计算SSIM"""
        if y_true.dim() == 4:  # Batch
            ssim_list = []
            for i in range(y_true.shape[0]):
                s = structural_similarity(
                    (y_true[i].permute(1,2,0).cpu().numpy() + 1) / 2,
                    (y_pred[i].permute(1,2,0).cpu().numpy() + 1) / 2,
                    data_range=1.0,
                    channel_axis=2
                )
                ssim_list.append(s)
            return np.mean(ssim_list)
        else:
            return structural_similarity(
                (y_true.permute(1,2,0).cpu().numpy() + 1) / 2,
                (y_pred.permute(1,2,0).cpu().numpy() + 1) / 2,
                data_range=1.0,
                channel_axis=2
            )
    
    def lpips(self, y_true, y_pred):
       
        with torch.no_grad():
            lpips_score = self.lpips_fn(y_true.to(self.device), y_pred.to(self.device))
        return lpips_score.item()