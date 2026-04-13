import utils
from utils import *
from generator import *
from discri import *
import torch.nn as nn
import torch, time, time, os, logging, statistics
import numpy as np
from generator import Generator
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from recover_mnist_reg import reparameterize, dist_inversion,dist_inversion_cifar
# from recover_vib import reparameterize, dist_inversion,dist_inversion_cifar
# from fid_score import calculate_fid_given_paths
# from fid_score_raw import calculate_fid_given_paths
from fid_score_mnist import calculate_fid_given_paths as calculate_fid_mnist
from fid_score_raw import calculate_fid_given_paths as calculate_fid_raw
from fid_score_cifar import calculate_fid_given_paths as calculate_fid_cifar

import sys

sys.path.append('../BiDO')
import model

if __name__ == "__main__":
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='cifar', help='celeba | mnist | cifar')
    parser.add_argument('--defense', default='COCO', help='HSIC | COCO')
    parser.add_argument('--iter', default=5000, type=int)
    parser.add_argument('--improved_flag', action='store_true', default=True, help='use improved k+1 GAN')
    parser.add_argument('--root_path', default="./improvedGAN")
    parser.add_argument('--model_path', default='../BiDOhsiccoco/target_model')
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
            #COCO

            (15, 75, 44.85),
            (10, 50, 73.17),
            (5, 10, 84.38),



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

            T = model.VGG16(1000, True)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, f"{args.dataset}", args.defense,
                                  "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

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

                acc, acc5, var, var5 = dist_inversion(args, G, D, T, E, iden, lr=2e-2, lamda=100,
                                                      iter_times=args.iter, clip_range=1, improved=args.improved_flag,
                                                      num_seeds=5, verbose=args.verbose)

                iden = iden + ids_per_time
                aver_acc += acc / times
                aver_acc5 += acc5 / times
                aver_var += var / times
                aver_var5 += var5 / times

            fid_value = calculate_fid_raw(
                args.dataset,
                [f'attack_res/{args.dataset}/trainset/',
                 f'attack_res/{args.dataset}/{args.defense}/all/'],
                50, 1, 2048
            )
            print(f'FID:{fid_value:.4f}')
            print("Avg acc:{:.2f}\tAvg acc5:{:.2f}\tAvg acc_var:{:.4f}\tAvg acc_var5:{:.4f}".format(
                    aver_acc,
                    aver_acc5,
                    aver_var,
                    aver_var5))


    elif args.dataset == 'mnist':
        hp_ac_list = [
            # # mnist-coco
            # (1, 50, 99.51),
            # (2, 20, 99.61),
            (15, 75, 99.80),
            (10, 50, 99.84),
            (5, 10, 99.84),
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
                                  "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))
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
                acc, acc5, var, var5 = dist_inversion(
                    args, G, D, T, E, iden,
                    lr=2e-2, lamda=100, iter_times=args.iter,
                    clip_range=1, improved=True, num_seeds=100, verbose=args.verbose
                )

                # MNIST-COCO 使用专门的 fid_score_mnist
                fid_value = calculate_fid_mnist(
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
            avg_fid, var_fid = statistics.mean(fid), statistics.stdev(fid)
            print(f"[Average over {K} runs] "
                  f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), "
                  f"Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
            print(f"[Average over {K} runs] FID:{avg_fid:.4f} (+/- {var_fid:.4f})")


    elif args.dataset == 'cifar':
        # 这里填入你的模型配置 (a1, a2, test_acc)
        # 如果是 Regular 模型，可以将 a1, a2 设为 0，并在逻辑中特殊处理
        hp_ac_list = [
            # # cifar - COCO
            # (15, 75, 42.49),
            # (10, 50, 53.69),
            # (5, 10, 79.95),
            (0.1, 5, 79.11),
            (0.5, 5, 88.06),
            (0.1, 2, 83.23),
            # # # cifar - hsic

            # (1, 0.5, 20.15),
            # (0.5, 1, 66.63),
            # (0.05, 0.5, 88.45),
        ]

        if len(hp_ac_list) == 0:
            print("[WARN] CIFAR hp_ac_list 为空，请填入实际 (a1, a2, acc) 以运行恢复。")

        for (a1, a2, ac) in hp_ac_list:
            defense_type = 'reg' if (a1 == 0 and a2 == 0) else args.defense
            hp_set = f"Defense: {defense_type} | a1={a1:.3f}, a2={a2:.3f}, test_acc={ac:.2f}"
            print(f"\n>>>> {hp_set}")

            # 1. 加载生成器 G 和 判别器 D
            G = Generator_CIFAR(z_dim)  # 确保使用 CIFAR 专用结构
            G = torch.nn.DataParallel(G).cuda()
            D = MinibatchDiscriminator_CIFAR(n_classes=10)
            D = torch.nn.DataParallel(D).cuda()

            if defense_type == 'reg':
                path_G = os.path.join(save_model_dir, f"{model_name}_G_reg_{ac:.2f}.tar")
                path_D = os.path.join(save_model_dir, f"{model_name}_D_reg_{ac:.2f}.tar")
            else:
                path_G = os.path.join(save_model_dir, "{}_G_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
                path_D = os.path.join(save_model_dir, "{}_D_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)

            G.load_state_dict(torch.load(path_G)['state_dict'], strict=False)
            D.load_state_dict(torch.load(path_D)['state_dict'], strict=False)

            # 2. 加载目标分类器 T
            T = model.VGG16(10,hsic_training=True, dataset='cifar')
            T = torch.nn.DataParallel(T).cuda()

            if defense_type == 'reg':
                path_T = os.path.join(args.model_path, args.dataset, 'reg', f"{model_name}_reg_{ac:.2f}.tar")
            else:
                path_T = os.path.join(args.model_path, args.dataset, args.defense,
                                      "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))

            T.load_state_dict(torch.load(path_T)['state_dict'], strict=False)
            E = T  # 评估模型复用 T

            # 3. 实验统计准备
            fid_list = []
            res_all = []
            K = 5

            for i in range(K):
                if args.verbose:
                    print(f'--- Round {i + 1} ---')

                # 攻击前 5 类
                iden = torch.from_numpy(np.arange(5)).cuda()

                # --- 调用修改后的 CIFAR 专用函数 ---
                # 注意：lamda 设为 500 或更高以解决 20% 准确率问题
                acc, acc5, var, var5 = dist_inversion_cifar(
                    args, G, D, T, E, iden,
                    lr=2e-2, lamda=500, iter_times=args.iter,
                    clip_range=1, improved=True, num_seeds=50, verbose=args.verbose
                )

                # 计算 FID
                fid_value = calculate_fid_cifar(
                    args.dataset,
                    [f'attack_res/{args.dataset}/trainset/',
                     f'attack_res/{args.dataset}/{args.defense}/all/'],
                    50, 1, 2048
                )

                print(f'Round {i + 1}: Acc={acc:.2f}, Acc5={acc5:.2f}, FID={fid_value:.4f}')

                fid_list.append(fid_value)
                res_all.append([acc, acc5, var, var5])

            # 4. 输出该配置下的平均结果
            res_avg = np.array(res_all).mean(0)
            avg_fid = statistics.mean(fid_list)
            var_fid = statistics.stdev(fid_list) if len(fid_list) > 1 else 0.0

            print(f"\n[Final Results for {hp_set}]")
            print(f"Average Acc: {res_avg[0]:.4f} (+/- {res_avg[2]:.4f})")
            print(f"Average Acc5: {res_avg[1]:.4f} (+/- {res_avg[3]:.4f})")
            print(f"Average FID: {avg_fid:.4f} (+/- {var_fid:.4f})")
 # 新的
# elif args.dataset == 'cifar':
    #     hp_ac_list = [
    #         # # # cifar - COCO
    #         # (15, 75, 42.49),
    #         # (10, 50, 53.69),
    #         # (5, 10, 79.95),
    #
    #         # # # cifar - hsic
    #
    #
    #         (1, 0.5, 20.15),
    #         (0.5, 1, 66.63),
    #         (0.05, 0.5, 88.45),
    #
    #     ]
    #     if len(hp_ac_list) == 0:
    #         print("[WARN] CIFAR hp_ac_list 为空，请填入实际 (a1, a2, acc) 以运行恢复。")
    #     for (a1, a2, ac) in hp_ac_list:
    #         hp_set = "a1 = {:.3f}|a2 = {:.3f}, test_acc={:.2f}".format(a1, a2, ac)
    #         print(hp_set)
    #         G = Generator_CIFAR(z_dim)
    #         G = torch.nn.DataParallel(G).cuda()
    #         D = MinibatchDiscriminator_CIFAR(n_classes=10)
    #         D = torch.nn.DataParallel(D).cuda()
    #
    #         path_G = os.path.join(save_model_dir, "{}_G_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
    #         path_D = os.path.join(save_model_dir, "{}_D_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
    #
    #         ckp_G = torch.load(path_G)
    #         G.load_state_dict(ckp_G['state_dict'], strict=False)
    #         ckp_D = torch.load(path_D)
    #         D.load_state_dict(ckp_D['state_dict'], strict=False)
    #
    #         # T = model.VGG16(10, dataset='cifar')
    #         T = model.VGG16(10, hsic_training=True, dataset='cifar')
    #
    #         T = torch.nn.DataParallel(T).cuda()
    #         path_T = os.path.join(args.model_path, f"{args.dataset}", args.defense,
    #                               "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))
    #         ckp_T = torch.load(path_T)
    #         T.load_state_dict(ckp_T['state_dict'], strict=False)
    #
    #         # CIFAR 没有单独评估模型时，直接复用目标模型
    #         E = T
    #
    #         aver_acc, aver_acc5, aver_var, aver_var5 = 0, 0, 0, 0
    #         fid = []
    #         res_all = []
    #
    #         K = 5
    #         for i in range(K):
    #             if args.verbose:
    #                 print('-------------------------')
    #             iden = torch.from_numpy(np.arange(5))
    #             acc, acc5, var, var5 = dist_inversion_cifar(
    #                 args, G, D, T, E, iden,
    #                 lr=2e-2, lamda=100, iter_times=args.iter,
    #                 clip_range=1, improved=True, num_seeds=50, verbose=args.verbose
    #             )
    #
    #             fid_value = calculate_fid_cifar(
    #                 args.dataset,
    #                 [f'attack_res/{args.dataset}/trainset/',
    #                  f'attack_res/{args.dataset}/{args.defense}/all/'],
    #                 50, 1, 2048
    #             )
    #             print(f'Round {i + 1}: Acc={acc:.2f}, Acc5={acc5:.2f}, '
    #                   f'Acc_var={var:.4f}, Acc5_var={var5:.4f}, FID={fid_value:.4f}')
    #
    #             fid.append(fid_value)
    #             res_all.append([acc, acc5, var, var5])
    #
    #         res = np.array(res_all).mean(0)
    #         avg_fid = statistics.mean(fid)
    #         var_fid = statistics.stdev(fid) if len(fid) > 1 else 0.0
    #         print(f"[Average over {K} runs] "
    #               f"Acc:{res[0]:.4f} (+/- {res[2]:.4f}), "
    #               f"Acc5:{res[1]:.4f} (+/- {res[3]:.4f})")
    #         print(f"[Average over {K} runs] FID:{avg_fid:.4f} (+/- {var_fid:.4f})")