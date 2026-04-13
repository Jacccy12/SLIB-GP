# 这个
import utils
from utils import *
from generator import *
from discri import *

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch, time, time, os, logging, statistics
import numpy as np
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch.nn.functional as F
# from fid_score import calculate_fid_given_paths
from fid_score_raw import calculate_fid_given_paths
from fid_score_mnist import calculate_fid_given_paths as calculate_fid_mnist
from fid_score_raw import calculate_fid_given_paths as calculate_fid_raw
from fid_score_cifar import calculate_fid_given_paths as calculate_fid_cifar

import sys

import torch

torch.cuda.empty_cache()

sys.path.append('../BiDO')
import model


# vib使用

def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu


# LS使用，输出out为（60，1000）
# LS使用，输出out为（60，1000）



def dist_inversion_cifar(args, G, D, T, E, iden, lr=2e-2, lamda=100, iter_times=1500, clip_range=1,
                         improved=True, num_seeds=50, verbose=False):
    iden = iden.view(-1).long().to('cuda')
    criterion = nn.CrossEntropyLoss().to('cuda')
    bs = iden.shape[0]

    G.eval()
    D.eval()
    T.eval()
    E.eval()

    # 1. 初始化 Latent Space (稍带随机性有助于跳出局部最优)
    mu = Variable(torch.randn(bs, 100).cuda() * 0.1, requires_grad=True)
    log_var = Variable(torch.randn(bs, 100).cuda() * 0.1, requires_grad=True)
    solver = optim.Adam([mu, log_var], lr=lr)

    # CIFAR-10 标准化函数 (仅用于分类器 T 和 评估模型 E)
    def normalize_for_vgg(x):
        # 1. 映射 Tanh 输出 [-1, 1] -> [0, 1]
        x_0_1 = (x + 1.0) / 2.0
        # 2. 标准化
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to('cuda')
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to('cuda')
        return (x_0_1 - mean) / std

        # TV Loss 专门用来杀掉雪花噪声
    def get_tv_loss(img):
        w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
        return (h_variance + w_variance) / img.size(0)

    def _vib_expected_input_size(net):
        m = net.module if hasattr(net, 'module') else net
        if hasattr(m, 'st_layer') and hasattr(m.st_layer, 'in_features'):
            return 64 if m.st_layer.in_features == 2048 else 32
        return 32


    # --- 优化阶段 ---
    for i in range(iter_times):
        z = reparameterize(mu, log_var)
        fake = G(z)  # 原始输出，范围通常在 [-1, 1]

        # # A. 判别器损失 (Prior Loss) - 保持原始值域
        # if improved:
        #     _, label = D(fake)  # D 期望看到的是 GAN 原始域的图片
        #     prior_loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
        # else:
        #     label = D(fake)
        #     prior_loss = - label.mean()
        #
        # # B. 分类器损失 (Identity Loss) - 必须标准化
        # if fake.shape[-1] != 32:
        #     fake_input = F.interpolate(fake, size=(32, 32), mode='bilinear', align_corners=False)
        # else:
        #     fake_input = fake
        #
        # # 核心：只有分类器 T 接收标准化后的输入
        # out = T(normalize_for_vgg(fake_input))
        # if isinstance(out, tuple):
        #     out = out[-1]
        #
        # iden_loss = criterion(out, iden)
        #
        # # 组合损失
        # total_loss = prior_loss + lamda * iden_loss

        # A. Prior Loss (加大 D 的权重)
        if improved:
            _, label = D(fake)
            prior_loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
        else:
            label = D(fake)
            prior_loss = - label.mean()

        # B. Identity Loss
        target_hw = _vib_expected_input_size(T)
        fake_input = F.interpolate(fake, size=(target_hw, target_hw), mode='bilinear', align_corners=False) if fake.shape[-1] != target_hw else fake
        out = T(normalize_for_vgg(fake_input))
        if isinstance(out, (list, tuple)):
            out = out[-1]
        iden_loss = criterion(out, iden)

        # C. TV Loss (平滑度约束)
        tv_loss = get_tv_loss(fake)

        # --- 这里的权重组合是关键 ---
        # 1. 降低 lamda 到 100
        # 2. 增加 tv_loss 权重 (0.1 - 1.0 之间尝试)
        # 3. 甚至给 prior_loss 加个 2 倍系数
        total_loss = 2.0 * prior_loss + lamda * iden_loss + 0.5 * tv_loss

        solver.zero_grad()
        total_loss.backward()
        solver.step()

        # 截断与约束
        with torch.no_grad():
            mu.data = torch.clamp(mu.data, -clip_range, clip_range)
            log_var.data = torch.clamp(log_var.data, -clip_range, clip_range)

        if (i + 1) % 500 == 0 and verbose:
            with torch.no_grad():
                cur_pred = torch.argmax(out, dim=1)
                cur_acc = iden.eq(cur_pred).sum().item() * 100.0 / bs
                print(
                    f"Iter {i + 1:4d} | Prior: {prior_loss.item():.2f} | Iden: {iden_loss.item():.2f} | Acc: {cur_acc:6.2f}%")

    # --- 评估阶段 ---
    res_list, res5_list = [], []
    actual_seeds = max(1, num_seeds)

    # 建立保存路径
    os.makedirs(args.save_img_dir, exist_ok=True)

    for seed in range(actual_seeds):
        with torch.no_grad():
            z = reparameterize(mu, log_var)
            fake = G(z)

            # 这里的 fake_save 必须转回 [0, 1] 才能正确计算 FID
            fake_save = (fake + 1.0) / 2.0
            fake_save = torch.clamp(fake_save, 0, 1)

            target_hw = _vib_expected_input_size(E)
            eval_input = F.interpolate(fake, size=(target_hw, target_hw), mode='bilinear', align_corners=False) if fake.shape[-1] != target_hw else fake
            eval_out = E(normalize_for_vgg(eval_input))
            eval_prob = eval_out[-1] if isinstance(eval_out, (list, tuple)) else eval_out
            eval_iden = torch.argmax(eval_prob, dim=1)

            cnt, cnt5 = 0, 0
            for j in range(bs):
                gt = iden[j].item()
                # 保存图片供 FID 计算使用
                save_tensor_images(fake_save[j], os.path.join(args.save_img_dir, f"iden_{gt}_seed_{seed}.png"))

                if eval_iden[j].item() == gt:
                    cnt += 1
                _, top5_idx = torch.topk(eval_prob[j], 5)
                if gt in top5_idx:
                    cnt5 += 1

            res_list.append(cnt * 100.0 / bs)
            res5_list.append(cnt5 * 100.0 / bs)

    if len(res_list) > 0:
        avg_acc = statistics.mean(res_list)
        avg_acc5 = statistics.mean(res5_list)
        var_acc = statistics.stdev(res_list) if len(res_list) > 1 else 0.0
        var_acc5 = statistics.stdev(res5_list) if len(res5_list) > 1 else 0.0
    else:
        avg_acc, avg_acc5, var_acc, var_acc5 = 0.0, 0.0, 0.0, 0.0

    return avg_acc, avg_acc5, var_acc, var_acc5


if __name__ == "__main__":
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='cifar', help='celeba | mnist')
    parser.add_argument('--defense', default='vib', help='reg | vib')
    parser.add_argument('--iter', default=5000, type=int)
    parser.add_argument('--improved_flag', action='store_true', default=True, help='use improved k+1 GAN')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../BiDOhsiccoco/target_model')
    parser.add_argument('--save_img_dir', default='./attack_res/')
    parser.add_argument('--success_dir', default='')
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--acc', default=0, type=float)
    parser.add_argument('--verbose', default=False, type=bool)

    args = parser.parse_args()

    z_dim = 100
    ############################# mkdirs ##############################
    log_path = os.path.join(args.root_path, "attack_logs")
    os.makedirs(log_path, exist_ok=True)

    log_file = f"attack_{args.dataset}_{args.defense}.txt"
    utils.Tee(os.path.join(log_path, log_file), 'a+')
    save_model_dir = os.path.join(args.root_path, args.dataset, args.defense)

    args.save_img_dir = os.path.join(args.save_img_dir, args.dataset, args.defense)
    args.success_dir = args.save_img_dir + "/res_success"
    os.makedirs(args.success_dir, exist_ok=True)

    args.save_img_dir = os.path.join(args.save_img_dir, 'all')
    os.makedirs(args.save_img_dir, exist_ok=True)

    ############################# mkdirs ##############################
    file = "./config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)
    stage = loaded_args["dataset"]["stage"]
    model_name = loaded_args["dataset"]["model_name"]



    if args.dataset == 'cifar':
        E = None  # 若未提供专用评估模型，则复用目标模型

        if args.defense == 'reg':
            ac = 91.20  # 可以根据需要调整
            T = model.VGG16(10, dataset='cifar')
            T = torch.nn.DataParallel(T).to('cuda')
            path_T = os.path.join(args.model_path, args.dataset,
                                  args.defense, f"{model_name}_reg_{ac:.2f}.tar")
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            path_G = os.path.join(save_model_dir, "{}_G_reg_{:.2f}.tar").format(model_name, ac)
            path_D = os.path.join(save_model_dir, "{}_D_reg_{:.2f}.tar").format(model_name, ac)

        elif args.defense == 'vib':
            # beta = args.beta if args.beta != 0 else 0.010
            # ac = args.acc if args.acc != 0 else 87.46
            configs = [
                (0.01, 87.46),
                # (0.005, 87.71),
                # (0.000, 88.12)

            ]
            for beta, ac in configs:
                print(f"Running VIB with beta={beta}, acc={ac}")
            path_T = os.path.join(args.model_path, args.dataset,
                                  f"{model_name}_beta{beta:.3f}_{ac:.2f}.tar")


            def _infer_vgg16_vib_dataset_from_ckpt(state_dict):
                key = None
                for k in ("module.st_layer.weight", "st_layer.weight"):
                    if k in state_dict:
                        key = k
                        break
                if key is None:
                    return 'cifar'
                feat_dim = state_dict[key].shape[1]
                return 'celeba' if feat_dim == 2048 else 'cifar'


            def _load_state_dict_shape_safe(net, state_dict):
                net_state = net.state_dict()
                filtered = {}
                for k, v in state_dict.items():
                    if k in net_state and hasattr(v, "shape") and net_state[k].shape == v.shape:
                        filtered[k] = v
                missing, unexpected = net.load_state_dict(filtered, strict=False)
                return missing, unexpected


            ckp_T = torch.load(path_T, map_location='cpu')
            ckpt_sd = ckp_T['state_dict'] if isinstance(ckp_T, dict) and 'state_dict' in ckp_T else ckp_T
            vib_dataset = _infer_vgg16_vib_dataset_from_ckpt(ckpt_sd)

            T = model.VGG16_vib(10, dataset=vib_dataset)
            T = torch.nn.DataParallel(T).to('cuda')

            try:
                T.load_state_dict(ckpt_sd, strict=True)
            except RuntimeError:
                _load_state_dict_shape_safe(T, ckpt_sd)

            path_G = os.path.join(save_model_dir, "{}_G_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)
            path_D = os.path.join(save_model_dir, "{}_D_beta_{:.3f}_{:.2f}.tar").format(model_name, beta, ac)
        else:
            raise ValueError(f"Unsupported defense {args.defense} for CIFAR")

        if E is None:
            E = T

        G = Generator_CIFAR(z_dim)
        G = torch.nn.DataParallel(G).to('cuda')
        D = MinibatchDiscriminator_CIFAR(n_classes=10)
        D = torch.nn.DataParallel(D).to('cuda')

        ckp_G = torch.load(path_G)
        G.load_state_dict(ckp_G['state_dict'], strict=False)
        ckp_D = torch.load(path_D)
        D.load_state_dict(ckp_D['state_dict'], strict=False)

        aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
        fid = []
        res_all = []

        K = 5
        for i in range(K):
            if args.verbose:
                print('-------------------------')
            iden = torch.from_numpy(np.arange(5))
            acc, acc5, var, var5 = dist_inversion_cifar(
                args, G, D, T, E, iden,
                lr=2e-2, lamda=100, iter_times=args.iter,
                clip_range=1, improved=True, num_seeds=50, verbose=args.verbose
            )

            fid_value = calculate_fid_cifar(
                args.dataset,
                [f'attack_res/{args.dataset}/trainset/',
                 f'attack_res/{args.dataset}/{args.defense}/all/'],
                50, 1, 2048
            )
            print(f'Round {i + 1}: Acc={acc:.2f}, Acc5={acc5:.2f}, '
                  f'Acc_var={var:.4f}, Acc5_var={var5:.4f}, FID={fid_value:.4f}')

            fid.append(fid_value)
            res_all.append([acc, acc5, var, var5])

        res = np.array(res_all).mean(0)
        avg_fid = statistics.mean(fid)
        var_fid = statistics.stdev(fid) if len(fid) > 1 else 0.0
        print(f"[Average over {K} runs] "
              f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), "
              f"Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
        print(f"[Average over {K} runs] FID:{avg_fid:.4f} (+/- {var_fid:.4f})")


