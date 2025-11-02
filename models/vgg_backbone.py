import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from collections import namedtuple

def my_init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class MyVGG16BN(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(MyVGG16BN, self).__init__()
        vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())
        
        self.slice1 = nn.Sequential()  
        self.slice2 = nn.Sequential()  
        self.slice3 = nn.Sequential()  
        self.slice4 = nn.Sequential()  
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )
        
        for i in range(12):  
            self.slice1.add_module(str(i), features[i])
        for i in range(12, 19):  
            self.slice2.add_module(str(i), features[i])
        for i in range(19, 29):  
            self.slice3.add_module(str(i), features[i])
        for i in range(29, 39):  
            self.slice4.add_module(str(i), features[i])
        
        my_init_weights(self.slice5.modules())
        
        if not pretrained:
            my_init_weights(self.slice1.modules())
            my_init_weights(self.slice2.modules())
            my_init_weights(self.slice3.modules())
            my_init_weights(self.slice4.modules())
        
        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)     
        h2 = self.slice2(h1)     
        h3 = self.slice3(h2)     
        h4 = self.slice4(h3)     
        h5 = self.slice5(h4)     
        VggOutputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = VggOutputs(h5, h4, h3, h2, h1)
        return out

if __name__ == "__main__":
    backbone = MyVGG16BN(pretrained=False)
    dummy = torch.randn(1, 3, 768, 768)
    outputs = backbone(dummy)
    print("Backbone outputs:")
    for key, value in outputs._asdict().items():
        print(key, value.shape)
