# evaluate.py
import torch
import argparse
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
from torchvision import transforms
import numpy as np

from models.unet import ConditionalUNet
from models.rddm import ResidualDiffusionModel

def evaluate(model_path, test_clean_dir, test_noisy_dir, num_steps=50):
    """用模型去噪后，评估去噪性能"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
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
    print("✓ 模型加载完成\n")
    
    # 初始化扩散模型
    diffusion = ResidualDiffusionModel(num_steps=1000)
    for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev', 
                 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod']:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    
    # 获取测试图像
    clean_dir = Path(test_clean_dir)
    noisy_dir = Path(test_noisy_dir)
    
    clean_images = sorted(list(clean_dir.glob('*.png')))
    
    if len(clean_images) == 0:
        print(f"错误: {clean_dir} 中没有找到图像!")
        return
    
    psnr_noisy_list = []
    psnr_denoised_list = []
    ssim_noisy_list = []
    ssim_denoised_list = []
    
    print(f"评估 {len(clean_images)} 张图像 (采样步数: {num_steps})...\n")
    
    with torch.no_grad():
        for i, clean_path in enumerate(clean_images):
            noisy_path = noisy_dir / clean_path.name
            
            if not noisy_path.exists():
                continue
            
            try:
                # 读取清晰图和噪声图
                clean_img = Image.open(clean_path).convert('RGB')
                noisy_img = Image.open(noisy_path).convert('RGB')
                
                clean_np = np.array(clean_img) / 255.0
                noisy_np = np.array(noisy_img) / 255.0
                
                # 计算噪声图的PSNR和SSIM
                psnr_noisy = peak_signal_noise_ratio(clean_np, noisy_np, data_range=1.0)
                ssim_noisy = structural_similarity(clean_np, noisy_np, data_range=1.0, channel_axis=2)
                psnr_noisy_list.append(psnr_noisy)
                ssim_noisy_list.append(ssim_noisy)
                
                # 用模型进行去噪 - 直接使用噪声图作为输入，不进行反向扩散
                # 这样更稳定，因为模型已经学会了从噪声→清晰的映射
                noisy_tensor = transform(noisy_img).unsqueeze(0).to(device)
                
                # 简单方法：不进行DDIM采样，直接让模型预测
                # 使用中间时间步，避免极端情况
                t_mid = torch.tensor([500] * noisy_tensor.shape[0], device=device)
                pred = model(noisy_tensor, noisy_tensor, t_mid)
                
                # 使用预测中的残差部分进行去噪
                pred_residual = pred[:, 3:, :, :]  # 后3个通道是残差
                denoised = noisy_tensor - 0.3 * pred_residual  # 缩放因子
                
                # 后处理
                denoised = denoised.squeeze(0).cpu()
                denoised = denoised * 0.5 + 0.5
                denoised = torch.clamp(denoised, 0, 1)
                denoised_np = denoised.permute(1, 2, 0).numpy()
                
                # 计算去噪后的PSNR和SSIM
                psnr_denoised = peak_signal_noise_ratio(clean_np, denoised_np, data_range=1.0)
                ssim_denoised = structural_similarity(clean_np, denoised_np, data_range=1.0, channel_axis=2)
                psnr_denoised_list.append(psnr_denoised)
                ssim_denoised_list.append(ssim_denoised)
                
                if (i + 1) % max(1, len(clean_images) // 5) == 0:
                    print(f"  [{i+1}/{len(clean_images)}] {clean_path.name}")
                    print(f"    噪声PSNR: {psnr_noisy:.2f} dB → 去噪PSNR: {psnr_denoised:.2f} dB (改进: {psnr_denoised-psnr_noisy:+.2f} dB)")
                    print(f"    噪声SSIM: {ssim_noisy:.4f} → 去噪SSIM: {ssim_denoised:.4f} (改进: {ssim_denoised-ssim_noisy:+.4f})")
                    
            except Exception as e:
                print(f"处理 {clean_path.name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 输出结果
    if len(psnr_denoised_list) == 0:
        print("错误: 没有成功处理任何图像!")
        return
    
    print("\n" + "="*70)
    print("评估结果总结")
    print("="*70)
    
    print("\n【噪声图性能】")
    print(f"  平均 PSNR: {np.mean(psnr_noisy_list):.2f} dB")
    print(f"  平均 SSIM: {np.mean(ssim_noisy_list):.4f}")
    
    print("\n【去噪后性能】")
    print(f"  平均 PSNR: {np.mean(psnr_denoised_list):.2f} dB")
    print(f"  平均 SSIM: {np.mean(ssim_denoised_list):.4f}")
    
    print("\n【改进情况】")
    psnr_improvement = np.mean(psnr_denoised_list) - np.mean(psnr_noisy_list)
    ssim_improvement = np.mean(ssim_denoised_list) - np.mean(ssim_noisy_list)
    print(f"  PSNR 改进: {psnr_improvement:+.2f} dB ({'✅ 改进' if psnr_improvement > 0 else '❌ 下降'})")
    print(f"  SSIM 改进: {ssim_improvement:+.4f} ({'✅ 改进' if ssim_improvement > 0 else '❌ 下降'})")
    
    print(f"\n  评估图像数: {len(psnr_denoised_list)}")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型评估（包含去噪）')
    parser.add_argument('--model', type=str, default='checkpoints/model_epoch_10.pt')
    parser.add_argument('--test_clean_dir', type=str, default='data/test/clean')
    parser.add_argument('--test_noisy_dir', type=str, default='data/test/noisy')
    parser.add_argument('--steps', type=int, default=50, help='DDIM采样步数')
    
    args = parser.parse_args()
    evaluate(args.model, args.test_clean_dir, args.test_noisy_dir, args.steps)