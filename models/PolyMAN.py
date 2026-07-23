import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .blocks import HybridBottleneck, SGAM, GLFM, unpack_tuple

# PolyMAN (Final Version: Mamba Broadcast + Hybrid Non-Local)
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
    def forward(self, x): 
        return self.conv(x)

class PolyMAN_UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        
        # --- 1. Encoder: ResNet34 ---
        print(" Building PolyMAN (Hybrid): Loading ResNet34 pretrained weights...")
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1 # 64
        self.enc3 = resnet.layer2 # 128
        self.enc4 = resnet.layer3 # 256
        self.enc5 = resnet.layer4 # 512
        
        # --- 2. Bottleneck (Mamba) ---
        self.bottleneck = HybridBottleneck(512)
        
        # --- 3. Skip Connections (SGAM) ---
        self.sgam4 = SGAM(256)
        self.sgam3 = SGAM(128)
        self.sgam2 = SGAM(64)
        
        # --- 4. Decoder: Hybrid GLFM ---
        self.glfm4 = GLFM(high_channels=512, low_channels=256, out_channels=256, 
                          mamba_channels=512, use_non_local=True)
        
        self.glfm3 = GLFM(high_channels=256, low_channels=128, out_channels=128, 
                          mamba_channels=512, use_non_local=True)
        
        self.glfm2 = GLFM(high_channels=128, low_channels=64, out_channels=64, 
                          mamba_channels=512, use_non_local=False)
        
        # Decoder 1: Standard Upsample 
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2) 
        self.dec1 = ConvBlock(32 + 64, 32)
        
        # --- 5. Deep Supervision ---
        self.head1 = nn.Conv2d(32, n_classes, 1)
        self.head2 = nn.Conv2d(64, n_classes, 1)
        self.head3 = nn.Conv2d(128, n_classes, 1)
        self.head4 = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        # --- Encoder ---
        x = unpack_tuple(x)
        e1 = self.enc1(x)       # 64, H/2
        e_pool = self.pool(e1)  # 64, H/4
        e2 = self.enc2(e_pool)  # 64, H/4
        e3 = self.enc3(e2)      # 128, H/8
        e4 = self.enc4(e3)      # 256, H/16
        e5 = self.enc5(e4)      # 512, H/32
        
        # --- Bottleneck ---
        # b: Mamba (Global Context)
        b = self.bottleneck(e5) 
        
        # --- Decoder + Skip (With Global Broadcasting) ---
        
        # Stage 4
        e4_enhanced = self.sgam4(e4)        
        d4 = self.glfm4(x_global=b, x_local=e4_enhanced, mamba_feat=b)
        
        # Stage 3
        e3_enhanced = self.sgam3(e3)
        d3 = self.glfm3(x_global=d4, x_local=e3_enhanced, mamba_feat=b) 
        
        # Stage 2
        e2_enhanced = self.sgam2(e2)
        d2 = self.glfm2(x_global=d3, x_local=e2_enhanced, mamba_feat=b) 
        
        # Stage 1
        d1 = self.up1(d2)              
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # --- Output ---
        out1 = self.head1(d1)
        out1 = F.interpolate(out1, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        if self.training:
            out2 = F.interpolate(self.head2(d2), size=x.shape[2:], mode='bilinear', align_corners=True)
            out3 = F.interpolate(self.head3(d3), size=x.shape[2:], mode='bilinear', align_corners=True)
            out4 = F.interpolate(self.head4(d4), size=x.shape[2:], mode='bilinear', align_corners=True)
            return out1, out2, out3, out4
            
        return out1
