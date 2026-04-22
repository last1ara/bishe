# setup_data.py
import os
import shutil
from pathlib import Path
import numpy as np
# 补全缺失的PIL Image导入
from PIL import Image  

def setup_data_structure():
    """
    适配已有目录结构：
    原始路径：D:\BISHE\data\BSD68\noise15/25/50/original
    目标：拆分训练/测试集，复用已有噪声图，新增泊松/混合噪声生成（可选）
    """
    print("="*60)
    print("开始准备训练数据（复用已有噪声图）...")
    print("="*60)
    
    # 根目录（适配你的路径）
    root_dir = Path(r'D:\BISHE\data\BSD68')
    # 已有噪声目录
    existing_noise_dirs = {
        'gaussian15': root_dir / 'noise15',
        'gaussian25': root_dir / 'noise25',
        'gaussian50': root_dir / 'noise50'
    }
    # 原始清晰图目录
    clean_dir = root_dir / 'original'
    
    # 检查目录是否存在
    for noise_name, noise_path in existing_noise_dirs.items():
        if not noise_path.exists():
            print(f"✗ 错误: {noise_path} 不存在!")
            return False
    if not clean_dir.exists():
        print(f"✗ 错误: {clean_dir} 不存在!")
        return False
    
    # 输出目录（拆分训练/测试）
    train_root = Path(r'D:\BISHE\data\train')
    test_root = Path(r'D:\BISHE\data\test')
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)
    
    # 读取清晰图列表（按名称排序，保证拆分一致）
    clean_images = sorted(list(clean_dir.glob('*.png')))
    if len(clean_images) == 0:
        print("✗ 错误: original目录下无png图片!")
        return False
    print(f"\n找到 {len(clean_images)} 张清晰图像")
    
    # 拆分训练/测试（50张训练，18张测试，BSD68标准拆分）
    train_split = 50
    train_clean = clean_images[:train_split]
    test_clean = clean_images[train_split:]
    print(f"\n分割数据:")
    print(f"  - 训练集: 前 {train_split} 张")
    print(f"  - 测试集: 后 {len(test_clean)} 张")
    
    # 1. 复制清晰图到训练/测试目录
    print(f"\n复制清晰图...")
    # 训练清晰图
    train_clean_dest = train_root / 'original'
    os.makedirs(train_clean_dest, exist_ok=True)
    for i, img_path in enumerate(train_clean):
        dest = train_clean_dest / img_path.name
        if not dest.exists():
            shutil.copy(str(img_path), str(dest))
    print(f"  ✓ 训练清晰图: {len(train_clean)} 张")
    
    # 测试清晰图
    test_clean_dest = test_root / 'original'
    os.makedirs(test_clean_dest, exist_ok=True)
    for i, img_path in enumerate(test_clean):
        dest = test_clean_dest / img_path.name
        if not dest.exists():
            shutil.copy(str(img_path), str(dest))
    print(f"  ✓ 测试清晰图: {len(test_clean)} 张")
    
    # 2. 复制已有高斯噪声图到训练/测试目录
    for noise_name, noise_path in existing_noise_dirs.items():
        print(f"\n处理 {noise_name} 噪声图...")
        # 训练噪声图
        train_noise_dest = train_root / noise_name
        os.makedirs(train_noise_dest, exist_ok=True)
        for img_path in train_clean:
            noisy_img_path = noise_path / img_path.name
            if noisy_img_path.exists():
                dest = train_noise_dest / img_path.name
                if not dest.exists():
                    shutil.copy(str(noisy_img_path), str(dest))
        
        # 测试噪声图
        test_noise_dest = test_root / noise_name
        os.makedirs(test_noise_dest, exist_ok=True)
        for img_path in test_clean:
            noisy_img_path = noise_path / img_path.name
            if noisy_img_path.exists():
                dest = test_noise_dest / img_path.name
                if not dest.exists():
                    shutil.copy(str(noisy_img_path), str(dest))
        
        print(f"  ✓ 训练{noise_name}噪声图: {len(list(train_noise_dest.glob('*.png')))} 张")
        print(f"  ✓ 测试{noise_name}噪声图: {len(list(test_noise_dest.glob('*.png')))} 张")
    
    # 3. 生成泊松/混合噪声（补充泛化能力）
    print(f"\n生成泊松/混合噪声图（补充数据）...")
    # 定义噪声生成函数（避免依赖外部utils）
    def generate_poisson_noise(image_np, peak=100):
        img = np.clip(image_np * peak, 0, peak)
        noisy = np.random.poisson(img) / peak
        return np.clip(noisy, 0, 1)
    
    def generate_mixed_noise(image_np, gauss_sigma=25, poisson_peak=100, ratio=0.5):
        # 高斯噪声
        gauss_noise = np.random.normal(0, gauss_sigma/255.0, image_np.shape)
        gauss_noisy = image_np + gauss_noise
        # 泊松噪声
        poisson_noisy = generate_poisson_noise(image_np, poisson_peak)
        # 混合
        mixed = ratio * gauss_noisy + (1 - ratio) * poisson_noisy
        return np.clip(mixed, 0, 1)
    
    # 训练集泊松/混合噪声
    for noise_type in ['poisson', 'mixed']:
        train_noise_dest = train_root / noise_type
        os.makedirs(train_noise_dest, exist_ok=True)
        for img_path in train_clean:
            clean_img = np.array(Image.open(img_path).convert('RGB')) / 255.0
            if noise_type == 'poisson':
                noisy_img = generate_poisson_noise(clean_img)
            else:  # mixed
                noisy_img = generate_mixed_noise(clean_img)
            # 保存
            noisy_img = Image.fromarray((noisy_img * 255).astype(np.uint8))
            noisy_img.save(train_noise_dest / img_path.name)
        
        # 测试集泊松/混合噪声
        test_noise_dest = test_root / noise_type
        os.makedirs(test_noise_dest, exist_ok=True)
        for img_path in test_clean:
            clean_img = np.array(Image.open(img_path).convert('RGB')) / 255.0
            if noise_type == 'poisson':
                noisy_img = generate_poisson_noise(clean_img)
            else:  # mixed
                noisy_img = generate_mixed_noise(clean_img)
            noisy_img = Image.fromarray((noisy_img * 255).astype(np.uint8))
            noisy_img.save(test_noise_dest / img_path.name)
        
        print(f"  ✓ 训练{noise_type}噪声图: {len(list(train_noise_dest.glob('*.png')))} 张")
        print(f"  ✓ 测试{noise_type}噪声图: {len(list(test_noise_dest.glob('*.png')))} 张")
    
    print("\n" + "="*60)
    print("✓ 数据准备完成!")
    print("最终目录结构:")
    print("  D:\BISHE\data\train/")
    print("    ├── original (训练清晰图)")
    print("    ├── gaussian15/25/50 (训练高斯噪声图)")
    print("    ├── poisson (训练泊松噪声图)")
    print("    ├── mixed (训练混合噪声图)")
    print("  D:\BISHE\data\test/ (结构同train)")
    print("="*60)
    
    return True

if __name__ == '__main__':
    setup_data_structure()