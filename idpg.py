# models/idpg.py
import torch

class IterativePreconditioning_Guidance:
    """迭代预处理引导 (IDPG)"""
    
    def __init__(self, num_steps=1000, guidance_start=0.7):
        self.num_steps = num_steps
        self.guidance_start = guidance_start  # 何时从BP切换到LS
    
    def get_guidance_weight(self, t_current, t_total):
        """
        动态调整引导权重
        - 早期（t大）：使用反向投影（BP）引导，权重大
        - 后期（t小）：使用最小二乘（LS）引导，权重小
        """
        progress = 1.0 - (t_current / t_total)  # 0到1的进度
        
        if progress < self.guidance_start:
            # 反向投影阶段：强引导
            return min(1.0, progress / self.guidance_start * 2.0)
        else:
            # 最小二乘阶段：弱引导
            remaining = (progress - self.guidance_start) / (1.0 - self.guidance_start)
            return max(0.1, 1.0 - remaining * 0.9)
    
    def backward_projection_guidance(self, x_pred, y, lambda_weight=0.5):
        """
        反向投影（BP）引导：
        通过最小化 ||Hx_pred - y||^2 来约束x_pred符合观测y
        H在图像去噪中是恒等算子（Identity），所以�� ||x_pred - y||^2
        """
        bp_gradient = 2.0 * (x_pred - y)
        return -lambda_weight * bp_gradient  # 负梯度方向是下降方向
    
    def least_squares_guidance(self, x_pred, y, residual_pred, lambda_weight=0.1):
        """
        最小二乘（LS）引导：
        最小化 ||x_pred + residual_pred - y||^2
        用于细节微调，同时抑制观测噪声
        """
        ls_residual = x_pred + residual_pred - y
        ls_gradient = 2.0 * ls_residual
        return -lambda_weight * ls_gradient
    
    def apply_guidance(self, x_pred, y, residual_pred, t_current, t_total, guidance_scale=0.1):
        """
        应用IDPG引导
        
        Args:
            x_pred: 预测的清晰图
            y: 观测噪声图
            residual_pred: 预测的残差
            t_current: 当前时间步
            t_total: 总时间步数
            guidance_scale: 引导强度缩放因子
        
        Returns:
            x_guided: 经过引导的预测
        """
        weight = self.get_guidance_weight(t_current, t_total)
        
        # 早期使用BP，后期使用LS
        progress = 1.0 - (t_current / t_total)
        
        if progress < self.guidance_start:
            guidance = self.backward_projection_guidance(x_pred, y, lambda_weight=guidance_scale)
        else:
            guidance = self.least_squares_guidance(x_pred, y, residual_pred, lambda_weight=guidance_scale)
        
        # 以权重的方式应用引导
        x_guided = x_pred + weight * guidance
        
        return torch.clamp(x_guided, -1.0, 1.0)