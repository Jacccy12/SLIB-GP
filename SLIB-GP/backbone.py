# copied from: https://github.com/wyharveychen/CloserLookFewShot/blob/master/backbone.py
# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import copy
import ipdb

# Basic ResNet model实现了一个用于小样本学习（Few-Shot Learning）的神经网络框架

f_softplus = nn.functional.softplus


def init_layer(L):
    # 初始化函数，用于对不同类型的神经网络层进行特定初始化
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels#n 是卷积核的参数量
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))#初始化权重
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)#初始化为全 1，表示初始时不缩放批归一化的输出。
        L.bias.data.fill_(0)


class distLinear(nn.Module):
    # 初始化这是一个自定义的线性层，实现了权重归一化的余弦距离计算
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        # 应用WeightNorm（权重归一化）将权重分解为方向和范数两部分
        WeightNorm.apply(self.L, 'weight', dim=0)
        if outdim <= 200:
            # 输出维度≤200时，使用较小的缩放因子2
            self.scale_factor = 2
        else:
            # 输出维度>200时（如Omniglot数据集），使用较大的缩放因子10
            self.scale_factor = 10

    def forward(self, x):
        # 前向传播部分(forward方法)
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)#计算x的L2范数
        x_normalized = x.div(x_norm + 0.00001)#x_normalized是归一化后的x（除以范数+小常数防止除零）
        L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(
            1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)  # matrix product by forward function余弦相似度计算
        scores = self.scale_factor * (cos_dist)

        return scores


class Flatten(nn.Module):#多维张量展平为一维（用于全连接层输入）通常用于卷积神经网络（CNN）到全连接层（Dense Layer）的过渡
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear):
    # 用于 MAML算法的自定义线性层，支持**快速权重（Fast Weights）**机制
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)#继承：基于 nn.Linear，保留原始线性层的所有功能
        self.weight.fast = None  # Lazy hack to add fast weight link# 快速权重占位符
        self.bias.fast = None    # 快速偏置占位符


    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) # 使用快速权重
        else:
            out = super(Linear_fw, self).forward(x)# 使用原始权重
        return out


class Linear_fw_bbb(nn.Linear):#线性层
    # 结合了 MAML快速权重 和 贝叶斯神经网络不确定性建模 的自定义线性层
    def __init__(self, in_features, out_features):
        super(Linear_fw_bbb, self).__init__(in_features, out_features)
        #MAML快速权重
        self.weight.fast = None
        self.bias.fast = None

        # 贝叶斯参数初始化
        self.weight_std = nn.Parameter(-10 *
                                       torch.ones_like(self.weight).to(self.weight.device))
        self.weight_std.fast = None
        self.bias_std = nn.Parameter(-10 *
                                     torch.ones_like(self.bias).to(self.weight.device))
        self.bias_std.fast = None

    def get_sample_stats(self):
        # 用于获取当前卷积层的贝叶斯采样参数和相关统计量。
        return (
            [self.sampled_w, self.sampled_b],#当前前向传播中实际使用的卷积核权重偏置项（经过随机采样
            [self.weight, self.bias],#权重偏置的高斯分布均值（原始权重参数）
            [self.weight_std, self.bias_std]#偏置权重分布的标准差参数（通过softplus转换为正值）
        )

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            # 使用快速权重 + 采样
            self.sampled_w.fast = self.weight.fast + \
                                  torch.randn_like(self.weight) * \
                                  f_softplus(self.weight_std.fast)
            self.sampled_b.fast = self.bias.fast + \
                                  torch.randn_like(self.bias) * f_softplus(self.bias_std.fast)
            out = F.linear(x, self.sampled_w.fast, self.sampled_b.fast)
        else:
            self.sampled_w = self.weight + \
                             torch.randn_like(self.weight) * f_softplus(self.weight_std)
            self.sampled_b = self.bias + \
                             torch.randn_like(self.bias) * f_softplus(self.bias_std)
            out = F.linear(x, self.sampled_w, self.sampled_b)
        return out


class Conv2d_fw(nn.Conv2d):
    # 用于 MAML 的自定义卷积层，支持快速权重（Fast Weights）机制
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels,
                                        kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None # 快速权重占位符
        if not self.bias is None:
            self.bias.fast = None # 快速偏置占位符（如果存在偏置）

    def forward(self, x):
        if self.bias is None:# 情况1：无偏置项
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None,# 使用快速权重
                               stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)# 使用原始权重
        else:# 情况3：偏置项
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast,
                               stride=self.stride, padding=self.padding)# 使用快速权重偏置
            else:
                out = super(Conv2d_fw, self).forward(x)# 使用原始权重偏置

        return out


class Conv2d_fw_bbb(nn.Conv2d): #PyTorch 中实现 二维卷积操作 的类
    # 结合了 MAML快速权重机制和贝叶斯神经网络特性的高级卷积层实现
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw_bbb, self).__init__(in_channels, out_channels,
                                            kernel_size, stride=stride, padding=padding, bias=bias)
        # 调用父类nn.Conv2d的初始化方法保持标准卷积层的所有基础参数
        assert bias
        # 为权重和偏置添加 fast 属性 初始化为 None，在MAML 内循环中会被赋予临时值
        self.weight.fast = None
        self.bias.fast = None
        # self.weight_std = copy.deepcopy(self.weight)
        # self.bias_std = copy.deepcopy(self.bias)
        self.weight_std = nn.Parameter(-10 *
                                       torch.ones_like(self.weight).to(self.weight.device))
        self.weight_std.fast = None
        self.bias_std = nn.Parameter(-10 *
                                     torch.ones_like(self.bias).to(self.weight.device))
        self.bias_std.fast = None

    def get_sample_stats(self):
        return (
            [self.sampled_w, self.sampled_b],
            [self.weight, self.bias],
            [self.weight_std, self.bias_std]
        )

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            self.sampled_w.fast = self.weight.fast + \
                                  torch.randn_like(self.weight.fast) * \
                                  f_softplus(self.weight_std.fast)
            self.sampled_b.fast = self.bias.fast + \
                                  torch.randn_like(self.bias.fast) * \
                                  f_softplus(self.bias_std.fast)
            out = F.conv2d(x, self.sampled_w.fast, self.sampled_b.fast,
                           stride=self.stride, padding=self.padding)
        else:

            self.sampled_w = self.weight + \
                             torch.randn_like(self.weight) * f_softplus(self.weight_std)

            self.sampled_b = self.bias + \
                             torch.randn_like(self.bias) * f_softplus(self.bias_std)
            out = F.conv2d(x, self.sampled_w, self.sampled_b,
                           stride=self.stride, padding=self.padding)
        return out


# used in MAML to forward input with fast weight批量归一化层
class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).type_as(x)
        running_var = torch.ones(x.data.size()[1]).type_as(x)

        if self.training:
            if self.weight.fast is not None and self.bias.fast is not None:
                # print("[Learner: ] updating")
                out = F.batch_norm(x, running_mean, running_var, self.weight.fast,
                                   self.bias.fast, training=True, momentum=1)
                # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
            else:
                # print("[Learner: ] 1st step")
                out = F.batch_norm(x, running_mean, running_var,
                                   self.weight, self.bias, training=True, momentum=1)
            self.running_var = running_var
            self.running_mean = running_mean
        else:  # this basically is used after "inner-loop" is done on the support set, and we're "evaluating" on the support set
            # print("[Learner: ] predicting")
            out = F.batch_norm(x, self.running_mean, self.running_var,
                               self.weight.fast, self.bias.fast, training=False)
        return out


class BBBConvBlock(nn.Module):
    def __init__(self, indim, outdim, pool=True, padding=1, relu=True):
        super(BBBConvBlock, self).__init__()
        self.indim = indim # 输入通道数
        self.outdim = outdim # 输出通道数
        self.C = Conv2d_fw_bbb(indim, outdim, 3, padding=padding) # 贝叶斯卷积
        self.BN = BatchNorm2d_fw(outdim)# MAML兼容的BN
        self.relu = nn.ReLU(inplace=True)# 可选激活函数

        # WARNING: having BN here pretty much means this is not Bayesian..# 基础层
        self.parametrized_layers = [self.C, self.BN]
        if relu:# 可选ReLU
            self.parametrized_layers += [self.relu]

        if pool:# 可选pool
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)# 自定义初始化

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def get_sample_stats(self):
        return self.C.get_sample_stats()# 从贝叶斯卷积层获取采样统计量

    def forward(self, x):
        out = self.trunk(x) # 按顺序执行各层
        return out


# Simple Conv Block


class ConvBlock(nn.Module):
    maml = False  # Default


    # 一个灵活的卷积块实现，可以根据需要选择是否支持MAML框架
    def __init__(self, indim, outdim, pool=True, padding=1, relu=True):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:#maml
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:#标准
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN]
        if relu:
            self.parametrized_layers += [self.relu]

        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


# Simple ResNet Block


class SimpleBlock(nn.Module):
    maml = False  # Default默认不使用MAML模式

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3,
                                stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(
                outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
            # Conv2d_fw和BatchNorm2d_fw是MAML专用的卷积和批归一化层，支持元学习中的参数快速适应。
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3,
                                stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(
                outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # i输入输出维度不同时           #通过1x1卷积调整通道数和分辨率（若half_res=True则步长为2
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(#卷积层
                    indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)#批归一化层
            else:
                self.shortcut = nn.Conv2d(
                    indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            # 将捷径分支的卷积层和BN层添加到 parametrized_layers列表中
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)
            # init_layer函数对卷积和BN层进行初始化

    def forward(self, x):
        # 主分支
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        # 捷径分支，确保捷径分支的输出与主分支的维度匹配
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(
            self.shortcut(x))
        # 当输入 / 输出维度相同时（shortcut_type = 'identity'），直接传递输入x，否则使用1x1卷积调整维度（含BN处理）
        out = out + short_out# 主分支 + 捷径分支
        out = self.relu2(out)# 最终激活
        return out


# Bottleneck block
class BottleneckBlock(nn.Module):#经典的 瓶颈残差块（Bottleneck Residual Block），属于 ResNet 中更高效的变体结构
    #通过 1×1 卷积降维再升维 来减少计算量
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim / 4)## 瓶颈层维度（通常是输出通道的1/4）
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim,
                                kernel_size=1, bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim,
                                kernel_size=3, stride=2 if half_res else 1, padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim,
                                kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim,
                                kernel_size=1, bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim,
                                kernel_size=3, stride=2 if half_res else 1, padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim,
                                kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [
            self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(
                    indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(
                    indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ConvNetLL(nn.Module):
    """ Linear Last layer带有线性最后一层（Linear Last Layer）的卷积神经网络，通常用于特征提取任务（如小样本学习）
    """

    def __init__(self, depth, flatten=True):
        super(ConvNetLL, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64# 输入通道：首层为3（RGB），其他层为64
            outdim = 64 # 输出通道固定为64
            if i < depth - 1: # 前n-1层

                B = ConvBlock(indim, outdim, pool=(i < 4))# 前4层带池化
            else:# 最后一层
                B = ConvBlock(indim, outdim, pool=(i < 4), relu=False)# 无ReLU
            trunk.append(B)
        # 展平与输出维度
        if flatten:
            trunk.append(Flatten())
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 256#final_feat_dim 硬编码为256，说明展平后特征长度为256

    def forward(self, x):
        out = self.trunk(x)
        return out


class WeightedFiLM(nn.Module):
    """
    加权特征线性调制层，
    FiLM 层的作用是通过外部条件（gammas 和 betas）对输入特征 x 进行逐通道的仿射变换（缩放和平移），并通过权重 film_w 控制调制强度。
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def forward(self, x, gammas, betas, film_w):
        # 如果 film_w=0，直接返回原始输入（不调制
        if film_w == 0:
            return x
        # 扩展 gammas 和 betas 到与 x 相同的形状 [B, C, H, W]
        gammas = 1 + gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        # 加权混合调制
        return (film_w * ((gammas * x) + betas)
                + (1 - film_w) * x)


class Conv4FiLM(nn.Module):
    # 实现了一个 4层卷积网络，并在每层卷积后插入了一个加权FiLM层（WeightedFiLM），用于动态调整特征
    def __init__(self, bw=False, flatten=True):
        super(Conv4FiLM, self).__init__()
        if bw:# 黑白图像，输入通道=1
            idim = 1
            fdim = 64
        else:# 彩色图像，输入通道=3（RGB）
            idim = 3
            fdim = 256
            # 4层卷积（每层后接池化）
        self.conv1 = ConvBlock(idim, 64, pool=True)
        self.conv2 = ConvBlock(64, 64, pool=True)
        self.conv3 = ConvBlock(64, 64, pool=True)
        self.conv4 = ConvBlock(64, 64, pool=True)
        # 展平层（可选）
        self.flatten = Flatten()
        # FiLM调制层（共享权重）
        self.film1 = WeightedFiLM()

        self.final_feat_dim = fdim

    def forward(self, x, gammas=0, betas=0, film_w=0):
        if isinstance(gammas, list):
            g1, g2, g3, g4 = gammas
            b1, b2, b3, b4 = betas
        else:
            g1, g2, g3, g4 = [gammas] * 4
            b1, b2, b3, b4 = [betas] * 4
        x = self.conv1(x)
        x = self.film1(x, g1, b1, film_w)
        x = self.conv2(x)
        x = self.film1(x, g2, b2, film_w)
        x = self.conv3(x)
        x = self.film1(x, g3, b3, film_w)
        x = self.conv4(x)
        x = self.film1(x, g4, b4, film_w)
        # x = self.film1(x, g4, b4, 0)
        x = self.flatten(x)
        return x


def Conv4FiLMBW():#快速创建一个黑白图像输入的 Conv4FiLM 实例。
    return Conv4FiLM(True)


class ConvNet(nn.Module):
    # 灵活可配置的通用卷积网络，支持贝叶斯推理和梯度控制，适用于需要不确定性估计的任务彩色图像
    def __init__(self, depth, flatten=True, bbb=False):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            if bbb:
                B = BBBConvBlock(indim, outdim, pool=(i < 4))## 使用贝叶斯卷积块
            else:
                # only pooling for fist 4 layers
                B = ConvBlock(indim, outdim, pool=(i < 4))## 普通卷积块
            trunk.append(B)

        self._trunk = trunk # 保存原始块列表（用于统计采样）

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        # self.final_feat_dim = 1600
        self.final_feat_dim = 256

#收集所有 BBBConvBlock 的统计信息（均值、方差等）。
    def get_sample_stats(self):
        ret = [[], [], []]
        for block in self._trunk:
            if not isinstance(block, BBBConvBlock):
                continue
            tmp = block.get_sample_stats()# 获取每个贝叶斯块的统计量
            ret[0] += tmp[0] # 均值列表
            ret[1] += tmp[1] # 方差列表
            ret[2] += tmp[2] # 其他统计量
        return ret

    def forward(self, x, no_grad=False):
        with torch.set_grad_enabled(not no_grad):
            out = self.trunk(x)
        return out


class ConvNetBW(nn.Module):
    # 专为 ** 黑白图像（单通道） ** 设计的卷积神经网络
    def __init__(self, depth, flatten=True, bbb=False):
        super(ConvNetBW, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            if bbb:#贝叶斯卷积选项（bbb=True 时启用 BBBConvBlock）
                B = BBBConvBlock(indim, outdim, pool=(i < 4))
            else:
                # only pooling for fist 4 layers
                B = ConvBlock(indim, outdim, pool=(i < 4))
            trunk.append(B)

        self._trunk = trunk

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    # 贝叶斯统计方法（get_sample_stats）
    def get_sample_stats(self):
        ret = [[], [], []]
        for block in self._trunk:
            if not isinstance(block, BBBConvBlock):
                continue
            tmp = block.get_sample_stats()
            ret[0] += tmp[0]
            ret[1] += tmp[1]
            ret[2] += tmp[2]
        return ret

    def forward(self, x, no_grad=False):
        with torch.set_grad_enabled(not no_grad):
            out = self.trunk(x)
        return out


# Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
class ConvNetNopool(nn.Module):#无全局池化的卷积网络
    def __init__(self, depth):
        super(ConvNetNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]), padding=0 if i in [ # 仅第0、1层使用池化
                0, 1] else 1)  # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 19, 19]

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNetS(nn.Module):  # 专为 Omniglot（手写字符数据集）优化的轻量卷积网络
    def __init__(self, depth, flatten=True):
        super(ConvNetS, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            # only pooling for fist 4 layers
            B = ConvBlock(indim, outdim, pool=(i < 4))# 前4层带池化
            trunk.append(B)

        if flatten:
            trunk.append(Flatten()) # 可选展平层

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self, x):
        out = x[:, 0:1, :, :]  # only use the first dimension
        out = self.trunk(out)
        return out


# Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
class ConvNetSNopool(nn.Module):#专为 Omniglot 等小尺寸图像设计的卷积网络
    def __init__(self, depth):
        super(ConvNetSNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]), padding=0 if i in [
                0, 1] else 1)  # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 5, 5]

    def forward(self, x):
        out = x[:, 0:1, :, :]  # only use the first dimension
        out = self.trunk(out)
        return out


class ResNet(nn.Module):#实现了一个灵活的残差网络架构，4个阶段（stage），每阶段可自定义层数和输出通道。通过 maml 标志切换第一层为MAML适配的快速权重层
    maml = False  # Default

    def __init__(self, nc, block, list_of_num_layers, list_of_out_dims, flatten=True, final_fmap_size=7):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        # 第一层（MAML模式或普通模式）
        if self.maml:
            conv1 = Conv2d_fw(nc, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(nc, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 初始化层参数
        init_layer(conv1)
        init_layer(bn1)
        # 构建初始层序列
        trunk = [conv1, bn1, relu, pool1]
        # 残差阶段构建
        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]
        # 输出处理
        if flatten:
            avgpool = nn.AvgPool2d(final_fmap_size)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, final_fmap_size, final_fmap_size]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out


class ResNetH(nn.Module):
    maml = False  # Default，ResNetH 是 ResNet 的改进版本

    def __init__(self, nc, block, list_of_num_layers, list_of_out_dims, flatten=True, final_fmap_size=7):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNetH, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(nc, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(nc, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        self.block1 = nn.Sequential(conv1, bn1, relu, pool1)
        # 残差阶段（block2-block5）
        indim = 64
        # for i in range(4):
        #     inner_block = []
        #     for j in range(list_of_num_layers[i]):
        #         half_res = (i >= 1) and (j == 0)
        #         B = block(indim, list_of_out_dims[i], half_res)
        #         inner_block.append(B)
        #         indim = list_of_out_dims[i]
        #
        # trunk.append(nn.Sequential(*inner_block))
        # ResNet的4个残差阶段（stage）的构建，每个阶段包含若干残差块（block
        i = 0
        inner_block = []
        for j in range(list_of_num_layers[i]):
            half_res = (i >= 1) and (j == 0)
            B = block(indim, list_of_out_dims[i], half_res)
            inner_block.append(B)
            indim = list_of_out_dims[i]
        self.block2 = nn.Sequential(*inner_block)

        i = 1
        inner_block = []
        for j in range(list_of_num_layers[i]):
            half_res = (i >= 1) and (j == 0)
            B = block(indim, list_of_out_dims[i], half_res)
            inner_block.append(B)
            indim = list_of_out_dims[i]
        self.block3 = nn.Sequential(*inner_block)

        i = 2
        inner_block = []
        for j in range(list_of_num_layers[i]):
            half_res = (i >= 1) and (j == 0)
            B = block(indim, list_of_out_dims[i], half_res)
            inner_block.append(B)
            indim = list_of_out_dims[i]
        self.block4 = nn.Sequential(*inner_block)

        i = 3
        inner_block = []
        for j in range(list_of_num_layers[i]):
            half_res = (i >= 1) and (j == 0)
            B = block(indim, list_of_out_dims[i], half_res)
            inner_block.append(B)
            indim = list_of_out_dims[i]
        self.block5 = nn.Sequential(*inner_block)

        if flatten:
            avgpool = nn.AvgPool2d(final_fmap_size)
            self.block6 = nn.Sequential(avgpool, Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, final_fmap_size, final_fmap_size]

    def forward(self, x):
        hiddens = []
        x = self.block1(x)
        hiddens.append(x)

        x = self.block2(x)
        hiddens.append(x)

        x = self.block3(x)
        hiddens.append(x)

        x = self.block4(x)
        hiddens.append(x)

        x = self.block5(x)
        hiddens.append(x)

        x = self.block6(x)
        torch.cuda.empty_cache()

        return x, hiddens


def ResNetL_IH(L, imgSize, nc, flatten=True):


    # 函数是ResNet的工厂函数，根据输入参数动态生成不同配置ResNetH的实例
    if imgSize == 32:
        ffs = 1
    elif imgSize == 64:
        ffs = 2
    elif imgSize == 128:
        ffs = 4
    elif imgSize == 256:
        ffs = 8
    else:
        raise

    if L == 10:
        net = ResNetH(nc, SimpleBlock, [1, 1, 1, 1], [
            64, 128, 256, 512], flatten, final_fmap_size=ffs)
    elif L == 34:
        net = ResNetH(nc, SimpleBlock, [3, 4, 6, 3], [
            64, 128, 256, 512], flatten, final_fmap_size=ffs)
    elif L == 50:
        net = ResNetH(nc, BottleneckBlock, [3, 4, 6, 3], [
            256, 512, 1024, 2048], flatten, final_fmap_size=ffs)
    return net

# 函数是不同卷积网络架构的工厂函数
def Conv4BWBBB():
    return ConvNetBW(4, bbb=True)


def Conv4BW():
    # for Omniglot
    return ConvNetBW(4)


def Conv4LL():
    return ConvNetLL(4)


def Conv4BBB():
    return ConvNet(4, bbb=True)


def Conv4():
    return ConvNet(4)


def Conv6():
    return ConvNet(6)


def Conv4NP():
    return ConvNetNopool(4)


def Conv6NP():
    return ConvNetNopool(6)


def Conv4S():
    return ConvNetS(4)


def Conv4SNP():
    return ConvNetSNopool(4)


def ResNet10(nc=3, flatten=True):
    return ResNet(nc, SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten, final_fmap_size=1)


def ResNet10_64(nc=3, flatten=True):
    return ResNetL_I(10, 64, nc, flatten=flatten)


def ResNetL_I(L, imgSize, nc, flatten=True):

    # 根据输入的图片尺寸 imgSize和层数L，选择合适的网络结构，并返回相应的ResNet模型。
    if imgSize == 32:
        ffs = 1
    elif imgSize == 64:
        ffs = 2
    elif imgSize == 128:
        ffs = 4
    elif imgSize == 256:
        ffs = 8
    else:
        raise

    if L == 10:
        net = ResNet(nc, SimpleBlock, [1, 1, 1, 1], [
            64, 128, 256, 512], flatten, final_fmap_size=ffs)
    elif L == 34:
        net = ResNet(nc, SimpleBlock, [3, 4, 6, 3], [
            64, 128, 256, 512], flatten, final_fmap_size=ffs)
    elif L == 50:
        net = ResNet(nc, BottleneckBlock, [3, 4, 6, 3], [
            256, 512, 1024, 2048], flatten, final_fmap_size=ffs)
    return net

# def ResNet18( flatten = True):
#     return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

# def ResNet34( flatten = True):
#     return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

# def ResNet50( flatten = True):
#     return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

# def ResNet101( flatten = True):
#     return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)
