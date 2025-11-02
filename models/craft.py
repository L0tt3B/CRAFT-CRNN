import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .vgg_backbone import MyVGG16BN

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()
        self.basenet = MyVGG16BN(pretrained=False, freeze=freeze)
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)
        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1)
        )
        
        def init_block(modules):
            for m in modules:
                if isinstance(m, nn.Conv2d):
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    
        init_block(self.upconv1.modules())
        init_block(self.upconv2.modules())
        init_block(self.upconv3.modules())
        init_block(self.upconv4.modules())
        init_block(self.conv_cls.modules())
        
    def forward(self, x):
        sources = self.basenet(x)
        y = torch.cat([sources.fc7, sources.relu5_3], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(y, size=sources.relu4_3.size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources.relu4_3], dim=1)
        y = self.upconv2(y)
        y = F.interpolate(y, size=sources.relu3_2.size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources.relu3_2], dim=1)
        y = self.upconv3(y)
        y = F.interpolate(y, size=sources.relu2_2.size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources.relu2_2], dim=1)
        feature = self.upconv4(y)
        y = self.conv_cls(feature)
        return y.permute(0, 2, 3, 1), feature

if __name__ == "__main__":
    model = CRAFT(pretrained=False)
    print(model)
    
    dummy_input = torch.randn(1, 3, 768, 768)
    output, features = model(dummy_input)
    print("Output shape:", output.shape)
