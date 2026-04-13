#
import utils
# 90,38
from utils import *
from generator import *
from discri import *
import torch.nn as nn
import torch, time, time, os, logging, statistics
import numpy as np
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from recover_vib import reparameterize, dist_inversion
from fid_score import calculate_fid_given_paths
from fid_score_raw import calculate_fid_given_paths
from fid_score_cifar import calculate_fid_given_paths as calculate_fid_cifar

import sys

sys.path.append('../BiDO')
import model

if __name__ == "__main__":
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='mnist', help='mnist | mnist | cifar')
    # 默认优先加载 DP+MI 防御模型（与你之前脚本保持一致）
    parser.add_argument('--defense', default='MI', help='MI+DP | MI | HSIC | COCO')
    parser.add_argument('--iter', default=5000, type=int)
    # 评估期防御：在将攻击样本送入目标模型前添加轻微噪声，降低可逆性
    # ⚠ 注意：原来这里是 default=True，相当于“永远加噪声”，导致所有 (a1,a2) 的攻击/ FID 都被同一种噪声防御洗平了。
    # 现在改成默认不加噪声，只有你显式传 --defend_eval 时才启用。
    parser.add_argument(
        '--defend_eval',
        action='store_true',
        default=False,
        help='是否在评估阶段对目标模型输入加噪声（默认关闭，传 --defend_eval 才开启）',
    )
    parser.add_argument(
        '--defend_noise_std',
        type=float,
        default=0.05,
        help='Gaussian noise std for eval-time defense (0~0.1)',
    )
    parser.add_argument('--improved_flag', action='store_true', default=True, help='use improved k+1 GAN')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--save_img_dir', default='./attack_res/')
    parser.add_argument('--success_dir', default='')
    parser.add_argument('--verbose', default=False, type=bool)

    args = parser.parse_args()

    z_dim = 100
    ############################# mkdirs ##############################
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

    if args.dataset == 'celeba':
        hp_ac_list = [


            # 最终的实验数据
            # # 有噪声三组数据MI+DP
            (0.05, 0.05, 82.85),  # 87.60(82.85)前面那个有噪声的attack acc不满意
            (0.05, 0.5, 64.06),  # 中等权重82.15(64.06)
            (0.2, 2.0, 54.95),  # 大权重73.3(54.95)
            # 无噪声MI
            # (0.05, 0.05, 83.54),  # 87.60(82.85)
            # (0.05, 0.5, 68.28),  # 中等权重82.15(64.06)
            # (0.2, 2.0, 52.49),  # 大权重73.3(54.95)

        ]

        for (a1, a2, ac) in hp_ac_list:
            hp_set = "a1 = {:.3f}|a2 = {:.3f}, test_acc={:.2f}".format(a1, a2, ac)
            print(hp_set)

            G = Generator(z_dim)
            G = torch.nn.DataParallel(G).cuda()
            D = MinibatchDiscriminator()
            D = torch.nn.DataParallel(D).cuda()

            path_G = os.path.join(save_model_dir, "{}_G_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
            path_D = os.path.join(save_model_dir, "{}_D_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)

            ckp_G = torch.load(path_G)
            G.load_state_dict(ckp_G['state_dict'], strict=False)
            ckp_D = torch.load(path_D)
            D.load_state_dict(ckp_D['state_dict'], strict=False)

            # 使用防御性模型加载器
            from load_defensive_model import load_defensive_model

            # 路径构造 + 回退：优先尝试 MI+DP，其次 MI
            base_dir = os.path.join(args.model_path, f"{args.dataset}")
            candidate_dirs = [args.defense]
            if args.defense == 'MI':
                candidate_dirs = ['MI+DP', 'MI']
            elif args.defense == 'MI+DP':
                candidate_dirs = ['MI+DP', 'MI']

            path_T = None
            for sub in candidate_dirs:
                candidate = os.path.join(base_dir, sub, "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))
                if os.path.exists(candidate):
                    path_T = candidate
                    if sub != args.defense:
                        print(f"ℹ️ 未找到期望的防御目录 {args.defense}，改为加载 {sub}")
                    break
            if path_T is None:
                # 退化为原路径（可能不存在），以便报错提示
                path_T = os.path.join(base_dir, args.defense,
                                      "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))

            print(f"🛡️ 加载防御性模型: {path_T}")
            print(f"🔒 模型包含DP和MI防御机制")

            # 设置身份映射范围，解决输出维度不匹配问题
            # 攻击使用60个身份，但模型输出1000维，需要映射
            identity_range = 60  # 攻击使用的身份数量
            # 使用MI训练时启用hsic_training
            hsic_training = True if (('MI' in args.defense) or ('HSIC' in args.defense)) else False

            T = load_defensive_model(path_T, model_name, n_classes=1000, dataset=args.dataset,
                                     identity_range=identity_range, hsic_training=hsic_training)

            # 评估期随机化防御封装（轻微高斯噪声），降低反演稳定性
            if args.defend_eval:
                import torch.nn as nn


                class EvalDefenseWrapper(nn.Module):
                    def __init__(self, net, noise_std=0.03):
                        super().__init__()
                        self.net = net
                        self.noise_std = float(max(0.0, min(noise_std, 0.1)))

                    def forward(self, x):
                        if self.training:
                            return self.net(x)
                        if self.noise_std > 0:
                            noise = torch.randn_like(x) * self.noise_std
                            x = torch.clamp(x + noise, -1.0, 1.0)
                        return self.net(x)


                # 仅当你显式指定 --defend_eval 时，才对 T 包一层噪声防御
                T = EvalDefenseWrapper(T, noise_std=args.defend_noise_std)
                print(f"[DefendEval] enabled, noise_std={args.defend_noise_std}")
            else:
                print("[DefendEval] disabled (no extra noise in recover)")

            T = torch.nn.DataParallel(T).cuda()

            # 确保防御性模型设置为推理模式
            T.eval()
            print("🛡️ 防御性模型已设置为推理模式")

            E = model.FaceNet(1000)
            E = torch.nn.DataParallel(E).cuda()
            path_E = './eval_ckp/FaceNet_95.88.tar'
            ckp_E = torch.load(path_E)
            E.load_state_dict(ckp_E['state_dict'], strict=False)

            ############         attack     ###########
            aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
            # evaluate on the first 300 identities only
            ids = 300
            times = 5
            ids_per_time = ids // times
            iden = torch.from_numpy(np.arange(ids_per_time))
            for idx in range(times):
                if args.verbose:
                    print("--------------------- Attack batch [%s]------------------------------" % idx)

                # 使用标准攻击设置，避免“弱化/强化”攻击导致评估偏差
                acc, acc5, var, var5 = dist_inversion(args, G, D, T, E, iden, lr=2e-2, lamda=100,
                                                      iter_times=args.iter, clip_range=1, improved=args.improved_flag,
                                                      num_seeds=10, verbose=args.verbose)

                iden = iden + ids_per_time
                aver_acc += acc / times
                aver_acc5 += acc5 / times
                aver_var += var / times
                aver_var5 += var5 / times

            fid_value = calculate_fid_given_paths(args.dataset,
                                                  [f'attack_res/{args.dataset}/trainset/',
                                                   f'attack_res/{args.dataset}/{args.defense}/all/'],
                                                  50, 1, 2048)
            print(f'FID:{fid_value:.4f}')
            print("Avg acc:{:.2f}\tAvg acc5:{:.2f}\tAvg acc_var:{:.4f}\tAvg acc_var5:{:.4f}".format(
                aver_acc,
                aver_acc5,
                aver_var,
                aver_var5))


    elif args.dataset == 'mnist':
        hp_ac_list = [
            # # mnist-coco
            # # MI+DP
            (0.01, 0.05, 99.92),  # 小权重84.24,,84.08，，79.82
            (0.05, 0.5, 99.90),  # 中等权重67.52
            (0.5, 5.0, 99.86),  # 很大权重41.39
            # # # MI
            # (0.01, 0.05, 99.96),  #
            # (0.05, 0.5, 99.90),  #
            # (0.5, 5.0, 99.92),  #
        ]
        for (a1, a2, ac) in hp_ac_list:
            hp_set = "a1 = {:.3f}|a2 = {:.3f}, test_acc={:.2f}".format(a1, a2, ac)
            print(hp_set)
            G = GeneratorMNIST(z_dim)
            G = torch.nn.DataParallel(G).cuda()
            D = MinibatchDiscriminator_MNIST()
            D = torch.nn.DataParallel(D).cuda()

            path_G = os.path.join(save_model_dir, "{}_G_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
            path_D = os.path.join(save_model_dir, "{}_D_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)

            ckp_G = torch.load(path_G)
            G.load_state_dict(ckp_G['state_dict'], strict=False)
            ckp_D = torch.load(path_D)
            D.load_state_dict(ckp_D['state_dict'], strict=False)

            T = model.MCNN(5)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, f"{args.dataset}", args.defense,
                                  "{}_{:.3f}&{:.3f}_{:.2f}_microdp.tar".format(model_name, a1, a2, ac))
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            E = model.SCNN(10)
            path_E = './eval_ckp/SCNN_99.90.tar'
            ckp_E = torch.load(path_E)
            E = nn.DataParallel(E).cuda()
            E.load_state_dict(ckp_E['state_dict'])

            aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
            fid = []
            res_all = []

            K = 5
            for i in range(K):
                if args.verbose:
                    print('-------------------------')
                iden = torch.from_numpy(np.arange(5))
                acc, acc5, var, var5 = dist_inversion(args, G, D, T, E, iden, lr=2e-2, lamda=100, iter_times=args.iter,
                                                      clip_range=1, improved=True, num_seeds=100, verbose=args.verbose)

                fid_value = calculate_fid_given_paths(args.dataset,
                                                      [f'attack_res/{args.dataset}/trainset/',
                                                       f'attack_res/{args.dataset}/{args.defense}/all/'],
                                                      50, 1, 2048)
                print(f'FID:{fid_value:.4f}')

                fid.append(fid_value)
                res_all.append([acc, acc5, var, var5])
                fid.append(fid_value)

            res = np.array(res_all).mean(0)
            avg_fid, var_fid = statistics.mean(fid), statistics.stdev(fid)
            print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
            print(f'FID:{avg_fid:.4f} (+/- {var_fid:.4f})')
    elif args.dataset == 'cifar':
        hp_ac_list = [
            # cifar - MI
            (0.5, 5, 64.51),
            (0.2, 2.0, 75.92),
            (0.01, 0.05, 89.49),

            # cifar - MI+DP

            # (0.2, 2.0, 60.59),
            # (0.05, 0.5, 75.34),
            # (0.01, 0.05, 88.19),
        ]
        if len(hp_ac_list) == 0:
            print("[WARN] CIFAR hp_ac_list 为空，请补充实际的 (a1, a2, acc) 与检查点文件。")

        for (a1, a2, ac) in hp_ac_list:
            hp_set = "a1 = {:.3f}|a2 = {:.3f}, test_acc={:.2f}".format(a1, a2, ac)
            print(hp_set)
            G = Generator_CIFAR(z_dim)
            G = torch.nn.DataParallel(G).cuda()
            D = MinibatchDiscriminator_CIFAR(n_classes=10)
            D = torch.nn.DataParallel(D).cuda()

            path_G = os.path.join(save_model_dir, "{}_G_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
            path_D = os.path.join(save_model_dir, "{}_D_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)

            ckp_G = torch.load(path_G)
            G.load_state_dict(ckp_G['state_dict'], strict=False)
            ckp_D = torch.load(path_D)
            D.load_state_dict(ckp_D['state_dict'], strict=False)

            # 目标模型：优先加载 MI+DP / MI，若不存在需补充
            from load_defensive_model import load_defensive_model

            base_dir = os.path.join(args.model_path, f"{args.dataset}")
            candidate_dirs = [args.defense]
            if args.defense == 'MI':
                candidate_dirs = ['MI+DP', 'MI']
            elif args.defense == 'MI+DP':
                candidate_dirs = ['MI+DP', 'MI']

            path_T = None
            for sub in candidate_dirs:
                candidate = os.path.join(base_dir, sub, "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))
                if os.path.exists(candidate):
                    path_T = candidate
                    if sub != args.defense:
                        print(f"ℹ️ 未找到期望的防御目录 {args.defense}，改为加载 {sub}")
                    break
            if path_T is None:
                path_T = os.path.join(base_dir, args.defense,
                                      "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))

            print(f"🛡️ 加载防御性模型: {path_T}")
            identity_range = 10
            hsic_training = True if (('MI' in args.defense) or ('HSIC' in args.defense)) else False
            T = load_defensive_model(path_T, model_name, n_classes=10, dataset=args.dataset,
                                     identity_range=identity_range, hsic_training=hsic_training)
            T = torch.nn.DataParallel(T).cuda()
            T.eval()

            # 若无单独的评估模型，则复用 T
            E = T

            aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
            fid = []
            res_all = []

            K = 5
            for i in range(K):
                if args.verbose:
                    print('-------------------------')
                iden = torch.from_numpy(np.arange(5))
                acc, acc5, var, var5 = dist_inversion(
                    args, G, D, T, E, iden, lr=2e-2, lamda=100, iter_times=args.iter,
                    clip_range=1, improved=True, num_seeds=100, verbose=args.verbose)

                fid_value = calculate_fid_cifar(
                    args.dataset,
                    [f'attack_res/{args.dataset}/trainset/',
                     f'attack_res/{args.dataset}/{args.defense}/all/'],
                    50, 1, 2048)
                print(f'FID:{fid_value:.4f}')

                fid.append(fid_value)
                res_all.append([acc, acc5, var, var5])
                fid.append(fid_value)

            res = np.array(res_all).mean(0)
            avg_fid = statistics.mean(fid)
            var_fid = statistics.stdev(fid) if len(fid) > 1 else 0.0
            print(f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
            print(f'FID:{avg_fid:.4f} (+/- {var_fid:.4f})')

