import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    """时间步嵌入层（无inplace操作）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),  # 移除inplace=True
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        # 生成正弦时间嵌入（兼容任意设备）
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # MLP映射 + 强制float32
        emb = self.mlp(emb).float()
        return emb

class ResBlock(nn.Module):
    """残差块（彻底移除inplace操作）"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        # 卷积层：严格匹配输入输出通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()  # 核心修复：移除inplace=True
        
        # 时间步投影：time_dim → out_channels
        self.time_proj = nn.Linear(time_dim, out_channels)
        
        # 残差连接：确保输入通道数=输出通道数
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        """
        x: [B, in_channels, H, W]
        t: [B, time_dim]
        """
        # 时间嵌入广播到空间维度
        t_emb = self.time_proj(t)[:, :, None, None].float()
        
        # 主干路径（无inplace操作）
        h = self.relu(self.norm1(self.conv1(x)))  # 先conv→norm→relu，无inplace
        h = h + t_emb  # 避免inplace加法（h += t_emb）
        h = self.relu(self.norm2(self.conv2(h)))
        
        # 残差连接 + 强制float32
        res = self.residual_conv(x)
        return (h + res).float()  # 避免inplace加法

class ConditionalUNet(nn.Module):
    """
    条件U-Net（最终稳定版，无inplace操作）
    固定通道数流程：6→32→64→128→128→64→32→6
    """
    def __init__(self, in_channels=3, out_channels=6, time_dim=32, hidden_dim=32):
        super().__init__()
        self.time_dim = time_dim
        
        # 1. 时间嵌入层
        self.time_emb = TimeEmbedding(time_dim)
        
        # 2. 输入层：6通道（3+3）→ 32通道
        self.input_conv = nn.Conv2d(in_channels * 2, hidden_dim, 3, padding=1)
        
        # 3. 下采样层（通道数翻倍，尺寸减半）
        # 32 → 64（尺寸H×W → H/2×W/2）
        self.down1 = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim * 2, time_dim),
            nn.MaxPool2d(2)
        )
        # 64 → 128（尺寸H/2×W/2 → H/4×W/4）
        self.down2 = nn.Sequential(
            ResBlock(hidden_dim * 2, hidden_dim * 4, time_dim),
            nn.MaxPool2d(2)
        )
        
        # 4. 瓶颈层（128 → 128，尺寸不变）
        self.bottleneck = ResBlock(hidden_dim * 4, hidden_dim * 4, time_dim)
        
        # 5. 上采样层（通道数减半，尺寸翻倍）
        # 128 → 64（尺寸H/4×W/4 → H/2×W/2）
        self.up1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 2, stride=2)
        self.res_up1 = ResBlock(hidden_dim * 2, hidden_dim * 2, time_dim)  # 输入64通道
        
        # 64 → 32（尺寸H/2×W/2 → H×W）
        self.up2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 2, stride=2)
        self.res_up2 = ResBlock(hidden_dim, hidden_dim, time_dim)  # 输入32通道
        
        # 6. 输出层：32 → 6通道
        self.out_conv = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)

    def forward(self, x, cond, t):
        """
        Args:
            x: 加噪图 [B, 3, H, W]
            cond: 噪声图 [B, 3, H, W]
            t: 时间步 [B] (float32)
        Returns:
            out: 预测结果 [B, 6, H, W]
        """
        # ========== 1. 输入拼接 + 初始卷积 ==========
        # 拼接加噪图+噪声图 → 6通道（无inplace）
        x_cat = torch.cat([x, cond], dim=1).float()
        # 6 → 32通道
        h = self.input_conv(x_cat).float()
        
        # ========== 2. 下采样（保存中间特征） ==========
        # 32 → 64，尺寸H×W → H/2×W/2
        h1 = self.down1[0](h, self.time_emb(t))
        h1_pool = self.down1[1](h1)
        
        # 64 → 128，尺寸H/2×W/2 → H/4×W/4
        h2 = self.down2[0](h1_pool, self.time_emb(t))
        h2_pool = self.down2[1](h2)
        
        # ========== 3. 瓶颈层 ==========
        bottleneck = self.bottleneck(h2_pool, self.time_emb(t))
        
        # ========== 4. 上采样 + 跳跃连接 ==========
        # 上采样1：128 → 64，尺寸H/4×W/4 → H/2×W/2
        up1 = self.up1(bottleneck)
        up1 = self.res_up1(up1, self.time_emb(t))
        
        # 上采样2：64 → 32，尺寸H/2×W/2 → H×W
        up2 = self.up2(up1)
        up2 = self.res_up2(up2, self.time_emb(t))
        
        # ========== 5. 输出层 ==========
        out = self.out_conv(up2).float()
        
        return out

# 验证模型（无inplace操作，梯度计算正常）
if __name__ == "__main__":
    # 初始化模型
    model = ConditionalUNet().float()
    
    # 模拟输入
    x = torch.randn(1, 3, 256, 256).float()
    cond = torch.randn(1, 3, 256, 256).float()
    t = torch.tensor([500], dtype=torch.float32)
    
    # 开启梯度检测
    torch.autograd.set_detect_anomaly(True)
    
    # 前向+反向传播测试
    try:
        out = model(x, cond, t)
        loss = out.sum()
        loss.backward()
        print("✅ 梯度计算正常！无inplace操作错误")
    except Exception as e:
        print(f"❌ 梯度计算错误: {e}")