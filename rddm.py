import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDiffusionModel(nn.Module):
    """
    残差扩散模型（Residual Diffusion Denoising Model, RDDM）
    适配BSD68图像去噪任务，核心特性：
    1. 统一张量类型为float32，避免类型不匹配
    2. 强制时间步为long类型，解决索引错误
    3. 自动设备对齐，兼容CPU/GPU训练
    4. 包含完整的前向/反向扩散逻辑
    """
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # 1. 预计算扩散核心参数（统一为float32）
        self.betas = torch.linspace(beta_start, beta_end, num_steps).float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).float()
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).float(), self.alphas_cumprod[:-1]])

        # 2. 预计算扩散过程所需的平方根项
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt((1.0 / self.alphas_cumprod) - 1).float()

        # 3. 预计算反向扩散所需参数
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).float()
        # 数值稳定：t=0时方差设为0
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20).float()
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance).float()
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).float()
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        ).float()

    def _move_to_device(self, tensor):
        """
        辅助函数：将张量移到模型所在设备（自动适配CPU/GPU）
        解决模型参数和输入张量设备不匹配问题
        """
        if self._parameters:  # 模型已初始化参数
            return tensor.to(next(self.parameters()).device)
        return tensor  # 模型未初始化时返回原张量

    def add_noise(self, x_0, y, t):
        """
        前向扩散过程：给清晰图添加噪声，返回加噪图+噪声目标+残差目标
        Args:
            x_0: 清晰图像 [B, 3, H, W] (float32)
            y: 噪声图像（条件） [B, 3, H, W] (float32)
            t: 时间步 [B] (支持float/long，内部自动转为long)
        Returns:
            x_t: 加噪图像 [B, 3, H, W] (float32)
            noise_target: 噪声预测目标 [B, 3, H, W] (float32)
            residual_target: 残差预测目标 [B, 3, H, W] (float32)
        """
        # 核心修复：强制时间步转为long类型（解决索引错误）
        t = t.long() if not t.dtype in (torch.long, torch.int) else t
        
        # 确保扩散参数在正确设备上且为float32
        sqrt_alphas = self._move_to_device(self.sqrt_alphas_cumprod)[t].view(-1, 1, 1, 1).float()
        sqrt_one_minus_alphas = self._move_to_device(self.sqrt_one_minus_alphas_cumprod)[t].view(-1, 1, 1, 1).float()

        # 生成与输入同分布、同设备的随机噪声（float32）
        noise = torch.randn_like(x_0, dtype=torch.float32, device=x_0.device)

        # 计算加噪图：x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1-alpha_cumprod) * noise
        x_t = (sqrt_alphas * x_0) + (sqrt_one_minus_alphas * noise)
        x_t = x_t.float()  # 显式确保类型

        # 计算残差目标：噪声图 - 清晰图（残差扩散核心）
        residual_target = (y - x_0).float()

        return x_t, noise, residual_target

    def predict_x0_from_noise(self, x_t, t, noise):
        """
        从加噪图和预测噪声恢复清晰图x_0
        Args:
            x_t: 加噪图像 [B, 3, H, W] (float32)
            t: 时间步 [B] (long)
            noise: 预测的噪声 [B, 3, H, W] (float32)
        Returns:
            x_0: 恢复的清晰图 [B, 3, H, W] (float32)
        """
        t = t.long()
        sqrt_recip_alphas_cumprod = self._move_to_device(self.sqrt_recip_alphas_cumprod)[t].view(-1, 1, 1, 1).float()
        sqrt_recipm1_alphas_cumprod = self._move_to_device(self.sqrt_recipm1_alphas_cumprod)[t].view(-1, 1, 1, 1).float()
        
        x_0 = sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise
        return x_0.float()

    def reverse_step(self, x_t, pred_noise, pred_residual, t, use_clip=True):
        """
        反向扩散单步：从x_t恢复x_{t-1}
        Args:
            x_t: 当前加噪图像 [B, 3, H, W] (float32)
            pred_noise: 预测的噪声 [B, 3, H, W] (float32)
            pred_residual: 预测的残差 [B, 3, H, W] (float32)
            t: 时间步 [B] (long)
            use_clip: 是否裁剪x_0到[-1,1]（稳定训练）
        Returns:
            x_t_prev: 上一步加噪图像 [B, 3, H, W] (float32)
        """
        t = t.long()
        batch_size = x_t.shape[0]

        # 1. 从预测噪声恢复x_0
        x_0 = self.predict_x0_from_noise(x_t, t, pred_noise)
        
        # 2. 裁剪x_0到[-1,1]（数值稳定）
        if use_clip:
            x_0 = torch.clamp(x_0, -1.0, 1.0)

        # 3. 计算反向扩散的均值和方差
        betas_t = self._move_to_device(self.betas)[t].view(-1, 1, 1, 1).float()
        alphas_t = self._move_to_device(self.alphas)[t].view(-1, 1, 1, 1).float()
        posterior_mean_coef1 = self._move_to_device(self.posterior_mean_coef1)[t].view(-1, 1, 1, 1).float()
        posterior_mean_coef2 = self._move_to_device(self.posterior_mean_coef2)[t].view(-1, 1, 1, 1).float()
        posterior_variance_t = self._move_to_device(self.posterior_variance)[t].view(-1, 1, 1, 1).float()
        posterior_log_variance_t = self._move_to_device(self.posterior_log_variance_clipped)[t].view(-1, 1, 1, 1).float()

        # 4. 计算x_{t-1}的均值
        model_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        # 残差修正（残差扩散核心）
        model_mean = model_mean + pred_residual

        # 5. 处理t=0的边界情况（无噪声）
        noise = torch.randn_like(x_t) if any(t > 0) else torch.zeros_like(x_t)
        x_t_prev = model_mean + torch.exp(0.5 * posterior_log_variance_t) * noise

        # 6. t=0时直接返回x_0
        x_t_prev = torch.where(
            t.view(-1, 1, 1, 1) > 0, 
            x_t_prev, 
            x_0
        )

        return x_t_prev.float()

    def sample(self, model, y, img_size=(256, 256), num_samples=1, device='cpu'):
        """
        完整反向扩散采样：从纯噪声生成去噪图像
        Args:
            model: 训练好的ConditionalUNet模型
            y: 噪声图像（条件） [num_samples, 3, H, W] (float32)
            img_size: 图像尺寸 (H, W)
            num_samples: 采样数量
            device: 采样设备
        Returns:
            x_0: 去噪后的清晰图像 [num_samples, 3, H, W] (float32)
        """
        self.eval()
        model.eval()

        # 1. 初始化x_T为纯噪声
        x_t = torch.randn(num_samples, 3, img_size[0], img_size[1], device=device).float()

        # 2. 反向扩散迭代
        with torch.no_grad():
            for t in reversed(range(0, self.num_steps)):
                # 生成当前时间步张量
                t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
                
                # 模型预测噪声和残差
                pred = model(x_t, y, t_tensor.float())
                pred_noise = pred[:, :3, :, :]
                pred_residual = pred[:, 3:, :, :]
                
                # 反向扩散一步
                x_t = self.reverse_step(x_t, pred_noise, pred_residual, t_tensor)

        # 3. 裁剪到[-1,1]并归一化到[0,1]
        x_0 = torch.clamp(x_t, -1.0, 1.0)
        x_0 = (x_0 + 1.0) / 2.0

        return x_0.float()

# 测试代码（验证模型功能，可选运行）
if __name__ == "__main__":
    # 初始化模型
    rddm = ResidualDiffusionModel(num_steps=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rddm.to(device)

    # 模拟输入
    batch_size = 1
    x_0 = torch.randn(batch_size, 3, 256, 256, device=device).float()  # 清晰图
    y = torch.randn(batch_size, 3, 256, 256, device=device).float()    # 噪声图
    t = torch.tensor([500], device=device).float()                     # 时间步（模拟之前的float类型）

    # 测试前向扩散
    x_t, noise_target, residual_target = rddm.add_noise(x_0, y, t)
    print("前向扩散测试:")
    print(f"  加噪图形状: {x_t.shape}, 类型: {x_t.dtype}")
    print(f"  噪声目标形状: {noise_target.shape}, 类型: {noise_target.dtype}")
    print(f"  残差目标形状: {residual_target.shape}, 类型: {residual_target.dtype}")

    # 测试x0恢复
    x_0_pred = rddm.predict_x0_from_noise(x_t, t.long(), noise_target)
    print(f"\nx0恢复测试:")
    print(f"  恢复x0形状: {x_0_pred.shape}, 类型: {x_0_pred.dtype}")

    # 测试反向扩散单步
    x_t_prev = rddm.reverse_step(x_t, noise_target, residual_target, t.long())
    print(f"\n反向扩散单步测试:")
    print(f"  x_t-1形状: {x_t_prev.shape}, 类型: {x_t_prev.dtype}")

    print("\nRDDM模型初始化和测试完成！")