# inference.py
import torch
import argparse
from PIL import Image
from torchvision import transforms
import os

from models.unet import ConditionalUNet
from models.rddm import ResidualDiffusionModel

def denoise(noisy_image_path, model_path, output_path, num_steps=50):
    """使用训练好的模型进行去噪推理"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = ConditionalUNet(
        in_channels=3,
        out_channels=6,
        time_dim=32,
        hidden_dim=32
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载噪声图像
    print(f"加载图像: {noisy_image_path}")
    img = Image.open(noisy_image_path).convert('RGB')
    
    # 预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    
    noisy = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # 初始化扩散模型
    diffusion = ResidualDiffusionModel(num_steps=1000)
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.alphas_cumprod_prev = diffusion.alphas_cumprod_prev.to(device)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    
    print(f"开始去噪 (步数: {num_steps})...")
    
    # 反向扩散过程（DDIM采样）
    with torch.no_grad():
        x = torch.randn_like(noisy)  # 随机初始化
        
        # 使用��间隔的时间步
        timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t] * x.shape[0], device=device)
            
            # 模型预测
            pred = model(x, noisy, t_tensor)
            pred_noise = pred[:, :3, :, :]
            pred_residual = pred[:, 3:, :, :]
            
            # 简单的反向步骤
            sqrt_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
            x = x - sqrt_one_minus * pred_noise * 0.1
            
            if (i + 1) % max(1, num_steps // 10) == 0:
                print(f"  进度: {i+1}/{num_steps}")
    
    # 后处理并保存
    denoised = x.squeeze(0).cpu()
    
    # 反归一化
    denoised = denoised * 0.5 + 0.5
    denoised = torch.clamp(denoised, 0, 1)
    
    # 转换为PIL Image
    denoised = transforms.ToPILImage()(denoised)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    denoised.save(output_path)
    
    print(f"✓ 去噪完成! 结果已保存到: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='扩散模型去噪推理')
    parser.add_argument('--input', type=str, required=True, help='输入噪声图像路径')
    parser.add_argument('--model', type=str, default='checkpoints/model_epoch_10.pt', help='模型路径')
    parser.add_argument('--output', type=str, default='results/denoised.png', help='输出路径')
    parser.add_argument('--steps', type=int, default=50, help='采样步数')
    
    args = parser.parse_args()
    
    denoise(args.input, args.model, args.output, args.steps)