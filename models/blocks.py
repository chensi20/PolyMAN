import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def unpack_tuple(x):
    if isinstance(x, (tuple, list)):
        return unpack_tuple(x[0])
    return x

# 1. Sobel Operator 
class SobelOperator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.conv_x = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.conv_y = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        
        self.conv_x.weight.data = sobel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.conv_y.weight.data = sobel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = unpack_tuple(x)
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

# 2. SGAM 
class SGAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.sobel = SobelOperator(in_channels)
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = unpack_tuple(x)
        b, c, _, _ = x.size()
        y_chan = self.avg_pool(x).view(b, c)
        y_chan = self.fc(y_chan).view(b, c, 1, 1)
        edge_map = self.sobel(x)
        spatial_map = self.spatial_conv(x)
        y_spatial = self.sigmoid(spatial_map + torch.mean(edge_map, dim=1, keepdim=True))
        return x * y_chan * y_spatial

#  Non-Local  (Embedded Gaussian) ---
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.inter_channels = inter_channels if inter_channels else in_channels // 2
        
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, 1)
        
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, H, W)
        
        return self.W(y) + x

# 3. GLFM (Mamba  +  Non-Local )
class GLFM(nn.Module):
    def __init__(self, high_channels, low_channels, out_channels, mamba_channels=512, use_non_local=False):
        super().__init__()
        self.use_non_local = use_non_local
        
        self.conv_global = nn.Sequential(nn.Conv2d(high_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        self.conv_local = nn.Sequential(nn.Conv2d(low_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        
        self.conv_mamba = nn.Sequential(nn.Conv2d(mamba_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        
        if self.use_non_local:
            self.non_local = NonLocalBlock(out_channels)
            self.branch_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True)
            )
          
            fusion_in = out_channels * 4
        else:

            fusion_in = out_channels * 3

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

    def forward(self, x_global, x_local, mamba_feat):
        x_global = unpack_tuple(x_global)
        x_local = unpack_tuple(x_local)
        mamba_feat = unpack_tuple(mamba_feat) 
        

        g = self.conv_global(x_global)
        l = self.conv_local(x_local)
        m = self.conv_mamba(mamba_feat)
        
        target_size = l.shape[2:]
        g = F.interpolate(g, size=target_size, mode='bilinear', align_corners=True)
        m = F.interpolate(m, size=target_size, mode='bilinear', align_corners=True)
        
        if self.use_non_local:
            g_nl = self.non_local(g) 
            g_br = self.branch_conv(g)  
            out = torch.cat([g_nl, g_br, l, m], dim=1)
        else:
            out = torch.cat([g, l, m], dim=1)
            
        return self.fusion(out)

# 4. OmniScaleBlock 
class OmniScaleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        mid_dim = dim // 4 
        self.branch3x3 = nn.Sequential(nn.Conv2d(dim, mid_dim, 1),nn.BatchNorm2d(mid_dim),nn.ReLU(True),nn.Conv2d(mid_dim, mid_dim, 3, padding=1, groups=mid_dim),nn.BatchNorm2d(mid_dim),nn.ReLU(True))
        self.branch5x5 = nn.Sequential(nn.Conv2d(dim, mid_dim, 1),nn.BatchNorm2d(mid_dim),nn.ReLU(True),nn.Conv2d(mid_dim, mid_dim, 5, padding=2, groups=mid_dim),nn.BatchNorm2d(mid_dim),nn.ReLU(True))
        self.branch1x1 = nn.Sequential(nn.Conv2d(dim, mid_dim, 1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.fusion = nn.Conv2d(mid_dim * 3, dim, 1)
        self.bn = nn.BatchNorm2d(dim); self.act = nn.ReLU(True)
    def forward(self, x):
        x = unpack_tuple(x)
        return self.act(self.bn(self.fusion(torch.cat([self.branch1x1(x), self.branch3x3(x), self.branch5x5(x)], dim=1)) + x))

# 5. LargeKernelBlock
class LargeKernelBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim) 
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = unpack_tuple(x)
        return x + self.attn(x) 

# 6. VisionSSMBlock 
def selective_scan_seq(u, delta, A, B, C, D):
    batch_size, length, d_inner = u.shape
    d_state = A.shape[1]
    deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
    deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
    h = torch.zeros(batch_size, d_inner, d_state, device=u.device)
    ys = []
    for i in range(length):
        h = deltaA[:, i] * h + deltaB_u[:, i]
        y = torch.einsum('bdn,bn->bd', h, C[:, i])
        ys.append(y)
    y = torch.stack(ys, dim=1)
    return y + u * D

class VisionSSMBlock(nn.Module):
    def __init__(self, dim, d_state=16, expand=1.5, dt_rank="auto"):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == "auto" else dt_rank
        self.d_state = d_state
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 3, padding=1, groups=self.d_inner)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner // d_state + 1)[:self.d_inner]
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim)

    def forward(self, x):
        x = unpack_tuple(x)
        B, C, H, W = x.shape
        x_in = x.flatten(2).transpose(1, 2)
        x_in = self.norm(x_in)
        x_and_z = self.in_proj(x_in)
        x_src, z = x_and_z.chunk(2, dim=-1)
        x_src = x_src.transpose(1, 2)
        x_src = self.conv1d(x_src)
        x_src = self.act(x_src).transpose(1, 2)
        x_dbl = self.x_proj(x_src)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = selective_scan_seq(x_src, dt, A.unsqueeze(1).repeat(1, self.d_state), B_ssm, C_ssm, self.D)
        y = y * self.act(z)
        y = self.out_norm(y)
        out = self.out_proj(y)
        return out.transpose(1, 2).view(B, C, H, W) + x

# 7. Hybrid Bottleneck
class HybridBottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local_pre = OmniScaleBlock(dim)
        self.local_stream = LargeKernelBlock(dim)
        self.global_stream = VisionSSMBlock(dim)
        self.fusion = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(True))

    def forward(self, x):
        x = unpack_tuple(x)
        x_base = self.local_pre(x)
        x_local = self.local_stream(x_base)
        x_global = self.global_stream(x)
        out = torch.cat([x_local, x_global], dim=1)
        return self.fusion(out)
