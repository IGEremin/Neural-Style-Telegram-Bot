import numpy as np
import torch
import torch.nn as nn


class GramMatrix(nn.Module):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, X):
        (b, ch, h, w) = X.shape
        features = X.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        conv_block = (norm_layer(inplanes),
                      nn.LeakyReLU(0.1),
                      ConvLayer(inplanes, planes, kernel_size=3, stride=stride),
                      norm_layer(planes),
                      nn.LeakyReLU(0.1),
                      ConvLayer(planes, planes, kernel_size=3, stride=1),
                      norm_layer(planes))
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, X):
        if self.downsample is not None:
            residual = self.residual_layer(X)
        else:
            residual = X
        return residual + self.conv_block(X)


class UpResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpResBlock, self).__init__()
        self.residual_layer = UpsampleConvLayer(inplanes, planes,
                                                kernel_size=1, stride=1, upsample=stride)
        conv_block = (norm_layer(inplanes),
                      nn.LeakyReLU(0.1),
                      UpsampleConvLayer(inplanes, planes, kernel_size=3, stride=1, upsample=stride),
                      norm_layer(planes),
                      nn.LeakyReLU(0.1),
                      ConvLayer(planes, planes, kernel_size=3, stride=1))
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, X):
        return self.residual_layer(X) + self.conv_block(X)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
                                            kernel_size=1, stride=stride)
        conv_block = (norm_layer(inplanes),
                      nn.LeakyReLU(0.1),
                      nn.Conv2d(inplanes, planes, kernel_size=1, stride=1),
                      norm_layer(planes),
                      nn.LeakyReLU(0.1),
                      ConvLayer(planes, planes, kernel_size=3, stride=stride),
                      norm_layer(planes),
                      nn.LeakyReLU(0.1),
                      nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1))
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, X):
        if self.downsample is not None:
            residual = self.residual_layer(X)
        else:
            residual = X
        return residual + self.conv_block(X)


class UpBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
                                                kernel_size=1, stride=1, upsample=stride)
        conv_block = (norm_layer(inplanes),
                      nn.LeakyReLU(0.1),
                      nn.Conv2d(inplanes, planes, kernel_size=1, stride=1),
                      norm_layer(planes),
                      nn.LeakyReLU(0.1),
                      UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride),
                      norm_layer(planes),
                      nn.LeakyReLU(0.1),
                      nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1))
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, X):
        return self.residual_layer(X) + self.conv_block(X)


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, X):
        X = self.reflection_pad(X)
        X = self.conv2d(X)
        return X


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, X):
        if self.upsample:
            X = self.upsample_layer(X)
        if self.reflection_padding != 0:
            X = self.reflection_pad(X)
        out = self.conv2d(X)
        return out


class Inspiration(nn.Module):
    def __init__(self, C, B=10):
        super(Inspiration, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)

        self.G = torch.Tensor(B, C, C).clone().detach().requires_grad_(True)
        self.C = C
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0.0, 0.02)

    def setTarget(self, target):
        self.G = target

    def forward(self, X):
        self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
        out = torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
                         X.view(X.size(0), X.size(1), -1)).view_as(X)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'N x ' + str(self.C) + ')'


class Net(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=128, norm_layer=nn.InstanceNorm2d, n_blocks=6):
        super(Net, self).__init__()
        self.gram = GramMatrix()

        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        model1 = (ConvLayer(input_nc, 64, kernel_size=7, stride=1),
                  norm_layer(64),
                  nn.LeakyReLU(0.1),
                  block(64, 32, 2, 1, norm_layer),
                  block(32 * expansion, ngf, 2, 1, norm_layer))
        self.model1 = nn.Sequential(*model1)

        self.ins = Inspiration(ngf * expansion)
        model = [self.model1, self.ins]

        for i in range(n_blocks):
            model += [block(ngf * expansion, ngf, 1, None, norm_layer)]

        model += [upblock(ngf * expansion, 32, 2, norm_layer),
                  upblock(32 * expansion, 16, 2, norm_layer),
                  norm_layer(16 * expansion),
                  nn.LeakyReLU(0.1),
                  ConvLayer(16 * expansion, output_nc, kernel_size=7, stride=1)]

        self.model = nn.Sequential(*model)

    def set_target(self, xs):
        xs = self.model1(xs)
        xs = self.gram(xs)
        self.ins.setTarget(xs)

    def forward(self, xs, xc):
        self.set_target(xs)
        res = self.model(xc)
        return res
