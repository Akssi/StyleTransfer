import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vgg16
from collections import namedtuple

# class Vgg16(torch.nn.Module):
#     def __init__(self, device = "cpu"):
#         super(Vgg16, self).__init__()
#         features = list(vgg16(pretrained = True).features)[:23]
#         self.features = nn.ModuleList(features).to(device).eval() 
#         self.layers = [3,8,15,22]
        
#     def forward(self, x):
#         results = []
#         for ii,model in enumerate(self.features):
#             x = model(x)
#             if ii in self.layers:
#                 results.append(x.detach())
#         return results

class Vgg16(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x].to(device))
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class SelectiveLoadModule(torch.nn.Module):
    """Only load layers in trained models with the same name."""
    def __init__(self):
        super(SelectiveLoadModule, self).__init__()

    def forward(self, x):
        return x

    def load_state_dict(self, state_dict):
        """Override the function to ignore redundant weights."""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)


class ConvLayer(torch.nn.Module):
    """Reflection padded convolution layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ConvTanh(ConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTanh, self).__init__(in_channels, out_channels, kernel_size, stride)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out = super(ConvTanh, self).forward(x)
        return self.tanh(out/255) * 150 + 255/2


class ConvInstRelu(ConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvInstRelu, self).__init__(in_channels, out_channels, kernel_size, stride)
        self.instance = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = super(ConvInstRelu, self).forward(x)
        out = self.instance(out)
        out = self.relu(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """Upsamples the input and then does a convolution.
    This method gives better results compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class UpsampleConvInstRelu(UpsampleConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvInstRelu, self).__init__(in_channels, out_channels, kernel_size, stride, upsample)
        self.instance = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = super(UpsampleConvInstRelu, self).forward(x)
        out = self.instance(out)
        out = self.relu(out)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.in1 = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)
        self.in2 = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class ReCoNet(SelectiveLoadModule):
    def __init__(self):
        super(ReCoNet, self).__init__()

        self.style_conv1 = ConvInstRelu(3, 32, kernel_size=9, stride=1)
        self.style_conv2 = ConvInstRelu(32, 64, kernel_size=3, stride=2)
        self.style_conv3 = ConvInstRelu(64, 128, kernel_size=3, stride=2)

        self.style_res1 = ResidualBlock(128, 128)
        self.style_res2 = ResidualBlock(128, 128)
        self.style_res3 = ResidualBlock(128, 128)
        self.style_res4 = ResidualBlock(128, 128)
        self.style_res5 = ResidualBlock(128, 128)

        self.style_deconv1 = UpsampleConvInstRelu(128, 64, kernel_size=3, stride=1, upsample=2)
        self.style_deconv2 = UpsampleConvInstRelu(64, 32, kernel_size=3, stride=1, upsample=2)
        self.style_deconv3 = ConvTanh(32, 3, kernel_size=9, stride=1)

    def forward(self, x):
        return self.style_deconv3(self.style_deconv2(self.style_deconv1(
            self.style_res5(self.style_res4(self.style_res3(self.style_res2(self.style_res1(
                self.style_conv3(self.style_conv2(self.style_conv1(x)))
            ))))))))


class ReCoNet2(SelectiveLoadModule):
    def __init__(self):
        super(ReCoNet2, self).__init__()

        self.style_conv1 = ConvInstRelu(3, 48, kernel_size=9, stride=1)
        self.style_conv2 = ConvInstRelu(48, 96, kernel_size=3, stride=2)
        self.style_conv3 = ConvInstRelu(96, 192, kernel_size=3, stride=2)

        self.style_res1 = ResidualBlock(192, 192)
        self.style_res2 = ResidualBlock(192, 192)
        self.style_res3 = ResidualBlock(192, 192)
        self.style_res4 = ResidualBlock(192, 192)

        self.style_deconv1 = UpsampleConvInstRelu(192, 96, kernel_size=3, stride=1, upsample=2)
        self.style_deconv2 = UpsampleConvInstRelu(96, 48, kernel_size=3, stride=1, upsample=2)
        self.style_deconv3 = ConvTanh(48, 3, kernel_size=9, stride=1)

    def forward(self, x):
        return self.style_deconv3(self.style_deconv2(self.style_deconv1(
            self.style_res4(self.style_res3(self.style_res2(self.style_res1(
                self.style_conv3(self.style_conv2(self.style_conv1(x)))
            )))))))


class ReCoNetMin(SelectiveLoadModule):
    def __init__(self):
        super(ReCoNetMin, self).__init__()

        self.style_conv1 = ConvInstRelu(3, 24, kernel_size=9, stride=1)
        self.style_conv2 = ConvInstRelu(24, 48, kernel_size=3, stride=2)
        self.style_conv3 = ConvInstRelu(48, 96, kernel_size=3, stride=2)

        self.style_res1 = ResidualBlock(96, 96)
        self.style_res2 = ResidualBlock(96, 96)
        self.style_res3 = ResidualBlock(96, 96)

        self.style_deconv1 = UpsampleConvInstRelu(96, 48, kernel_size=3, stride=1, upsample=2)
        self.style_deconv2 = UpsampleConvInstRelu(48, 24, kernel_size=3, stride=1, upsample=2)
        self.style_deconv3 = ConvTanh(24, 3, kernel_size=9, stride=1)

    def forward(self, x):
        return self.style_deconv3(self.style_deconv2(self.style_deconv1(
            self.style_res3(self.style_res2(self.style_res1(
                self.style_conv3(self.style_conv2(self.style_conv1(x))))
            )))))