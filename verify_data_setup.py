# verify_data.py
from pathlib import Path

def verify_data():
    """验证数据目录和文件数量"""
    print("="*60)
    print("验证BSD68数据结构...")
    print("="*60)
    
    # 原始目录
    original_dirs = {
        '原始清晰图': Path(r'D:\BISHE\data\BSD68\original'),
        '高斯15噪声': Path(r'D:\BISHE\data\BSD68\noise15'),
        '高斯25噪声': Path(r'D:\BISHE\data\BSD68\noise25'),
        '高斯50噪声': Path(r'D:\BISHE\data\BSD68\noise50')
    }
    
    # 检查原始目录
    print("\n【原始数据检查】")
    for name, path in original_dirs.items():
        if path.exists():
            cnt = len(list(path.glob('*.png')))
            print(f"✓ {name}: {path} (数量: {cnt})")
        else:
            print(f"✗ {name}: {path} 不存在！")
    
    # 检查拆分后的目录
    split_dirs = {
        '训练清晰图': Path(r'D:\BISHE\data\train\original'),
        '训练高斯15': Path(r'D:\BISHE\data\train\gaussian15'),
        '训练高斯25': Path(r'D:\BISHE\data\train\gaussian25'),
        '训练高斯50': Path(r'D:\BISHE\data\train\gaussian50'),
        '训练泊松': Path(r'D:\BISHE\data\train\poisson'),
        '训练混合': Path(r'D:\BISHE\data\train\mixed'),
        '测试清晰图': Path(r'D:\BISHE\data\test\original'),
        '测试高斯15': Path(r'D:\BISHE\data\test\gaussian15'),
        '测试高斯25': Path(r'D:\BISHE\data\test\gaussian25'),
        '测试高斯50': Path(r'D:\BISHE\data\test\gaussian50'),
        '测试泊松': Path(r'D:\BISHE\data\test\poisson'),
        '测试混合': Path(r'D:\BISHE\data\test\mixed'),
    }
    
    print("\n【拆分后数据检查】")
    all_ok = True
    for name, path in split_dirs.items():
        if path.exists():
            cnt = len(list(path.glob('*.png')))
            print(f"✓ {name}: {path} (数量: {cnt})")
        else:
            print(f"✗ {name}: {path} 不存在！")
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ 所有数据目录验证通过！")
    else:
        print("✗ 部分目录缺失，请先运行setup_data.py！")
    print("="*60)

if __name__ == '__main__':
    verify_data()