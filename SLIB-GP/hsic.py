import torch
import numpy as np
from torch.autograd import Variable, grad
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

import torch.nn.functional as F
def sigma_estimation(X, Y):#估计一个核函数的参数
    """ sigma from median distance
    """
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med


def distmat(X):#计算输入矩阵 X 中每对样本之间的欧氏距离矩阵
    """ distance matrix
    """
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def coco_kernelmat(X, sigma, ktype='gaussian'):#根据输入数据 X 计算核矩阵
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))

    ## Adding linear kernel
    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(H, torch.mm(Kx, H))

    return Kxc


def coco_normalized_cca(x, y, sigma, ktype='gaussian'):#计算数据 x 和 y 之间的规范化的核相关性coco（CCA）
    m = int(x.size()[0])
    K = coco_kernelmat(x, sigma=sigma)
    L = coco_kernelmat(y, sigma=sigma, ktype=ktype)

    res = torch.sqrt(torch.norm(torch.mm(K, L))) / m
    return res


def coco_objective(hidden, h_target, h_data, sigma, ktype='gaussian'):#计算coco依赖性
    coco_hx_val = coco_normalized_cca(hidden, h_data, sigma=sigma)#lxz
    coco_hy_val = coco_normalized_cca(hidden, h_target, sigma=sigma, ktype=ktype)#lzy

    return coco_hx_val, coco_hy_val


def kernelmat(X, sigma, ktype='gaussian'):#计算核函数
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))


    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(Kx, H)

    return Kxc


def hsic_normalized_cca(x, y, sigma, ktype='gaussian'):#计算xy之间的hsic值
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype)

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy


def hsic_objective(hidden, h_target, h_data, sigma, ktype='gaussian'):#计算hsic依赖性
    hsic_hx_val = hsic_normalized_cca(hidden, h_data, sigma=sigma)
    hsic_hy_val = hsic_normalized_cca(hidden, h_target, sigma=sigma, ktype=ktype)

    return hsic_hx_val, hsic_hy_val







# 成功版本
# def mi_normalized(x, y):
#     """使用分块计算的方式计算互信息
#     Args:
#         x: 输入特征 [batch_size, feature_dim]
#         y: 目标变量 [batch_size, class_dim] 或 [batch_size]
#     Returns:
#         标准化的互信息值
#     """
#     # 确保所有输入都在同一个设备上
#     device = x.device
#
#     # 对输入进行归一化
#     x = F.normalize(x, p=2, dim=1)
#     if y.dim() > 1:
#         y = F.normalize(y, p=2, dim=1)
#
#     # 计算相似度矩阵
#     sim_matrix = torch.mm(x, x.t())
#
#     # 对相似度矩阵进行缩放，避免数值溢出
#     temperature = 0.1  # 温度参数
#     sim_matrix = sim_matrix / temperature
#
#     # 使用更稳定的softmax实现
#     sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0]  # 减去最大值以提高数值稳定性
#     sim_matrix = F.softmax(sim_matrix, dim=1)
#
#     # 计算互信息
#     if y.dim() > 1:
#         # 对于one-hot编码的标签
#         y_sim = torch.mm(y, y.t())
#         y_sim = (y_sim > 0).float()
#     else:
#         # 对于普通标签
#         y_sim = (y.unsqueeze(0) == y.unsqueeze(1)).float()
#
#     # 计算互信息，添加数值稳定性检查
#     mi = torch.sum(sim_matrix * y_sim) / x.size(0)
#
#     # 确保返回值在合理范围内
#     mi = torch.clamp(mi, min=0.0, max=1.0)
#
#     # 检查是否为NaN或Inf
#     if torch.isnan(mi) or torch.isinf(mi):
#         print("Warning: MI calculation resulted in NaN or Inf")
#         mi = torch.tensor(0.0, device=device)
#
#     return mi


# def mi_objective(hidden, h_target, h_data):
#     """计算基于互信息的依赖性目标
#     Args:
#         hidden: 隐藏层表示
#         h_target: 目标变量
#         h_data: 输入数据
#     Returns:
#         mi_hx_val: 隐藏层与输入之间的互信息
#         mi_hy_val: 隐藏层与目标之间的互信息
#     """
#     # 确保所有输入都在同一个设备上
#     device = hidden.device
#     h_target = h_target.to(device)
#     h_data = h_data.to(device)
#
#     # 对隐藏层进行降维
#     if hidden.size(1) > 512:
#         with torch.no_grad():
#             proj_matrix = torch.randn(hidden.size(1), 512, device=device) / (hidden.size(1) ** 0.5)
#             hidden = hidden @ proj_matrix
#         # 添加数值稳定性检查
#         if torch.isnan(hidden).any() or torch.isinf(hidden).any():
#             print("Warning: NaN or Inf in hidden features")
#             hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=-1.0)
#
#         if torch.isnan(h_data).any() or torch.isinf(h_data).any():
#             print("Warning: NaN or Inf in input data")
#             h_data = torch.nan_to_num(h_data, nan=0.0, posinf=1.0, neginf=-1.0)
#
#         if torch.isnan(h_target).any() or torch.isinf(h_target).any():
#             print("Warning: NaN or Inf in target data")
#             h_target = torch.nan_to_num(h_target, nan=0.0, posinf=1.0, neginf=-1.0)
#
#     mi_hx_val = mi_normalized(hidden, h_data)
#     mi_hy_val = mi_normalized(hidden, h_target)
#
#     return mi_hx_val, mi_hy_val


# def mi_normalized(x, y):
#     """使用改进的核函数方法计算互信息，添加更多的数值稳定性措施
#     Args:
#         x: 输入特征 [batch_size, feature_dim]
#         y: 目标变量 [batch_size, class_dim] 或 [batch_size]
#     Returns:
#         标准化的互信息值
#     """
#     # 确保所有输入都在同一个设备上
#     device = x.device
#
#     # 对输入进行归一化，使用更稳定的方式
#     x = F.normalize(x, p=2, dim=1)
#     if y.dim() > 1:
#         y = F.normalize(y, p=2, dim=1)
#
#     # 计算核矩阵
#     def compute_kernel_matrix(data, sigma=None):
#         # 计算欧氏距离矩阵，使用更稳定的方式
#         data = data.float()  # 确保数据类型一致
#         data = torch.clamp(data, min=-10.0, max=10.0)  # 限制数值范围
#
#         # 使用更稳定的距离计算
#         data_norm = (data ** 2).sum(1).view(-1, 1)
#         dist = data_norm + data_norm.t() - 2.0 * torch.mm(data, data.t())
#         dist = torch.clamp(dist, min=0.0)  # 确保距离非负
#
#         # 如果没有指定sigma，使用中位数距离
#         if sigma is None:
#             sigma = torch.median(dist[dist > 0])
#             if sigma <= 0:
#                 sigma = torch.mean(dist[dist > 0])
#             if sigma < 1E-2:
#                 sigma = torch.tensor(1E-2, device=device)
#
#         # 使用更稳定的RBF核函数计算
#         kernel = torch.exp(-dist / (2 * sigma * sigma))
#         kernel = torch.clamp(kernel, min=1e-6, max=1.0)  # 限制核函数值范围
#         return kernel
#
#     try:
#         # 计算x和y的核矩阵
#         Kx = compute_kernel_matrix(x)
#         if y.dim() > 1:
#             Ky = compute_kernel_matrix(y)
#         else:
#             # 对于标签，使用delta核
#             Ky = (y.unsqueeze(0) == y.unsqueeze(1)).float()
#
#         # 中心化核矩阵，使用更稳定的方式
#         n = x.size(0)
#         H = torch.eye(n, device=device) - (1.0 / n) * torch.ones((n, n), device=device)
#         Kx = torch.mm(torch.mm(H, Kx), H)
#         Ky = torch.mm(torch.mm(H, Ky), H)
#
#         # 计算HSIC统计量，使用更稳定的方式
#         hsic = torch.trace(torch.mm(Kx, Ky)) / (n * n)
#
#         # 归一化HSIC值到[0,1]区间
#         hsic = torch.clamp(hsic, min=0.0, max=1.0)
#
#         # 数值稳定性检查
#         if torch.isnan(hsic) or torch.isinf(hsic):
#             print("Warning: MI calculation resulted in NaN or Inf")
#             hsic = torch.tensor(0.0, device=device)
#
#         return hsic
#
#     except RuntimeError as e:
#         print(f"Error in MI calculation: {str(e)}")
#         return torch.tensor(0.0, device=device)
#
#
# def mi_objective(hidden, h_target, h_data):
#     """计算基于互信息的依赖性目标，添加更多的数值稳定性措施
#     Args:
#         hidden: 隐藏层表示
#         h_target: 目标变量
#         h_data: 输入数据
#     Returns:
#         mi_hx_val: 隐藏层与输入之间的互信息
#         mi_hy_val: 隐藏层与目标之间的互信息
#     """
#     # 确保所有输入都在同一个设备上
#     device = hidden.device
#     h_target = h_target.to(device)
#     h_data = h_data.to(device)
#
#     # 对隐藏层进行降维和归一化
#     if hidden.size(1) > 512:
#         with torch.no_grad():
#             proj_matrix = torch.randn(hidden.size(1), 512, device=device) / (hidden.size(1) ** 0.5)
#             hidden = hidden @ proj_matrix
#
#     # 添加数值稳定性检查
#     if torch.isnan(hidden).any() or torch.isinf(hidden).any():
#         print("Warning: NaN or Inf in hidden features")
#         hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=-1.0)
#
#     if torch.isnan(h_data).any() or torch.isinf(h_data).any():
#         print("Warning: NaN or Inf in input data")
#         h_data = torch.nan_to_num(h_data, nan=0.0, posinf=1.0, neginf=-1.0)
#
#     if torch.isnan(h_target).any() or torch.isinf(h_target).any():
#         print("Warning: NaN or Inf in target data")
#         h_target = torch.nan_to_num(h_target, nan=0.0, posinf=1.0, neginf=-1.0)
#
#     try:
#         # 计算互信息
#         mi_hx_val = mi_normalized(hidden, h_data)
#         mi_hy_val = mi_normalized(hidden, h_target)
#
#         # 确保返回值在合理范围内
#         mi_hx_val = torch.clamp(mi_hx_val, min=0.0, max=1.0)
#         mi_hy_val = torch.clamp(mi_hy_val, min=0.0, max=1.0)
#
#         return mi_hx_val, mi_hy_val
#
#     except RuntimeError as e:
#         print(f"Error in MI objective calculation: {str(e)}")
#         return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

def mi_normalized(x, y):
    """使用改进的核函数方法计算互信息，添加更多的数值稳定性措施
    Args:
        x: 输入特征 [batch_size, feature_dim]
        y: 目标变量 [batch_size, class_dim] 或 [batch_size]
    Returns:
        标准化的互信息值
    """
    # 确保所有输入都在同一个设备上
    device = x.device

    # 对输入进行归一化，使用更稳定的方式
    x = F.normalize(x, p=2, dim=1)
    if y.dim() > 1:
        y = F.normalize(y, p=2, dim=1)

    # 计算核矩阵
    def compute_kernel_matrix(data, sigma=None):
        # 计算欧氏距离矩阵，使用更稳定的方式
        data = data.float()  # 确保数据类型一致
        data = torch.clamp(data, min=-10.0, max=10.0)  # 限制数值范围

        # 使用更稳定的距离计算
        data_norm = (data ** 2).sum(1).view(-1, 1)
        dist = data_norm + data_norm.t() - 2.0 * torch.mm(data, data.t())
        dist = torch.clamp(dist, min=0.0)  # 确保距离非负

        # 如果没有指定sigma，使用中位数距离
        if sigma is None:
            sigma = torch.median(dist[dist > 0])
            if sigma <= 0:
                sigma = torch.mean(dist[dist > 0])
            if sigma < 1E-2:
                sigma = torch.tensor(1E-2, device=device)

        # 使用更稳定的RBF核函数计算
        kernel = torch.exp(-dist / (2 * sigma * sigma))
        kernel = torch.clamp(kernel, min=1e-6, max=1.0)  # 限制核函数值范围
        return kernel

    try:
        # 计算x和y的核矩阵
        Kx = compute_kernel_matrix(x)
        if y.dim() > 1:
            Ky = compute_kernel_matrix(y)
        else:
            # 对于标签，使用delta核
            Ky = (y.unsqueeze(0) == y.unsqueeze(1)).float()

        # 中心化核矩阵，使用更稳定的方式
        n = x.size(0)
        H = torch.eye(n, device=device) - (1.0 / n) * torch.ones((n, n), device=device)
        Kxc = torch.mm(H, torch.mm(Kx, H))
        Kyc = torch.mm(H, torch.mm(Ky, H))

        # 计算HSIC，添加一个小的正则项以提高稳定性
        epsilon = 1e-5
        dKx = torch.diag(Kxc)
        dKy = torch.diag(Kyc)

        # 确保分母不为零
        Mi_val = torch.sum(dKx * dKy) / torch.sqrt(
            torch.sum(dKx * dKx) * torch.sum(dKy * dKy) + epsilon
        )

        # 再次检查最终值是否为NaN
        if torch.isnan(Mi_val):
            return torch.tensor(0.0, device=device)

        return Mi_val

    except Exception as e:
        # 捕获任何潜在的计算错误，返回一个安全的零值
        print(f"Error in mi_normalized: {e}")
        return torch.tensor(0.0, device=device)


def mi_objective(hidden, h_target, h_data):
    """计算基于互信息的依赖性目标
    Args:
        hidden: 隐藏层表示
        h_target: 目标变量
        h_data: 输入数据
    Returns:
        mi_hx_val: 隐藏层与输入之间的互信息
        mi_hy_val: 隐藏层与目标之间的互信息
    """
    mi_hx_val = mi_normalized(hidden, h_data)
    mi_hy_val = mi_normalized(hidden, h_target)

    return mi_hx_val, mi_hy_val