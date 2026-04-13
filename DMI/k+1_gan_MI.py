import os
import time
import utils
import torch
from utils import *
from torch.autograd import grad
import torch.nn.functional as F
from discri import DGWGAN, Discriminator, MinibatchDiscriminator
from discri import *
from generator import Generator
from generator import *
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import sys
#基于生成对抗网络（GAN）的目标恢复攻击
sys.path.append('../BiDO')
import model


def freeze(net):# 冻结生成器
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net): # 解冻判别器
    for p in net.parameters():
        p.requires_grad_(True)


def gradient_penalty(x, y):#计算梯度惩罚
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x) # 在x和y之间线性插值
    z = z.cuda()
    z.requires_grad = True# 开启梯度计算
    # 2. 计算判别器输出
    o = DG(z)
    # 3. 计算梯度
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


def log_sum_exp(x, axis=1):
    m = torch.max(x, dim=1)[0]

    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim=axis))


if __name__ == "__main__":
    #参数解析与初始化
    parser = ArgumentParser(description='Step2: targeted recovery')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist')
    parser.add_argument('--defense', default='MI+DP', help='HSIC | vib | MI')
    parser.add_argument('--root_path', default="./improvedGAN")

    parser.add_argument('--model_path', default='../BiDO/target_model')
    parser.add_argument('--beta', default=0, type=float)
    parser.add_argument('--acc', default=0, type=float)
    args = parser.parse_args()
    #加载配置文件与创建目录
    file = "./config/" + args.dataset + ".json"
    loaded_args = load_json(json_file=file)

    ############################# mkdirs ##############################
    os.makedirs(args.root_path, exist_ok=True)
    save_model_dir = os.path.join(args.root_path, args.dataset, args.defense)
    os.makedirs(save_model_dir, exist_ok=True)

    ############################# mkdirs ##############################
    #数据集与模型配置
    file_path = loaded_args['dataset']['gan_file_path']
    stage = loaded_args['dataset']['stage']
    lr = loaded_args[stage]['lr']
    batch_size = loaded_args[stage]['batch_size']
    z_dim = loaded_args[stage]['z_dim']
    epochs = loaded_args[stage]['epochs']
    n_critic = loaded_args[stage]['n_critic']

    utils.print_params(loaded_args["dataset"], loaded_args[stage])
    model_name = loaded_args["dataset"]["model_name"]

    if args.dataset == 'celeba':
        hp_ac_list = [
            # celeba - hsic
            # (0.05, 0.05, 84.64),
            # (0.01, 0.01, 83.01),
            # (0.03, 0.03, 83.98),
             # (0.06, 0.06, 83.34),
            # (0.04, 0.04, 83.28),#新的
            # (0.04, 0.04, 83.24),
            # (0.05, 0.5,83.71),
            # (0.01, 0.05, 83.34),

            # (0.03, 0.03, 83.41),#原始版加了DP，engine＋DP版，效果非常好
            # (0.04, 0.04 ,83.21),#效果好的一版加了DP，engine＋DP版，没有多行数据，效果不好
            # (0.01, 0.03,83.38),#原始版加了DP，engine＋DP版，效果非常好
            # (0.03, 0.03, 83.98),  # 效果好的一版加了DP，用的engine＋DPold版，有多行数据,效果不好
            # (0.03, 0.04, 83.78),#原始版加了DP，engine＋DP版，没有多行数据，效果好
            # (0.1, 0.2,83.94),#原始版加了DP，engine＋DP版，没有多行数据,
            # (0.05, 0.05,83.11),  # 用的engine＋DP 版，没有很多行效果不好的一版修改之后尝试对齐效果好的一版
            # (0.05, 0.05, 85.70),




#重新训练
            # (0.05, 0.5, 86.14),#mutual+DP，
            # (0.05, 0.5, 85.34),#mutualnew
            # (0.05, 0.5, 85.67),#mutual上
            # (0.05, 0.05, 86.27),#mutual下

            # (0.05, 0.5, 86.00),#改变了一行代码的mutual+DP，两行都存在，效果正常

            # (0.05, 0.5, 85.47),# 只存在了代码net = torch.nn.DataParallel(net)  # Opacus 不支持 DataParallel，推荐使用单卡或 DDP

            # (0.05, 0.5, 85.51),  # mutual加了dp操作
            # (0.05, 0.5, 85.57),#mutual加了dp操作，渐进式
            # (0.05, 0.5, 85.24),  # mutual加了dp操作，渐进式，又改进了
            # (0.05, 0.5, 85.80),  # mutual加了dp操作，渐进式，又改进了,又改进了
            # (0.05, 0.5, 85.37),  # mutual加了dp操作，渐进式，又改进了,又改进了,又改进了
            # (0.05, 0.5, 85.50),  # mutual加了dp操作，渐进式，改进了

            # (0.05, 0.5, 85.41),  # mutual,dp_t ual



# 这三组是新的
             # (0.05, 0.5, 89.69),  # train_mutual_dp_test
            # (0.06, 0.65, 90.16),  # train_mutual_dp_test
            # (0.05, 0.1, 90.03),  # train_mutual_dp_test，改小了差分隐私值

            # (0.5, 5.0, 40.86),  # 较大权重
            # (0.1, 1.0, 61.07),      # 大权重

# # gail,mi
#
#             (0.01, 0.1,80.22),  # 小权重80.22
#             (0.01, 0.05,84.24),  # 小权重84.24
#    # mi+dp
#             (0.05, 0.5,82.15),     # 中等权重82.15
#             (0.05, 0.05,87.60),#87.60
#             (0.1, 1.0,79.22),      # 稍大权重79.22
#             (0.2, 2.0,73.3),  # 大权重73.3
#             (0.5, 5.0,62.43),  # 很大权重\62.43

# mi+dp,MI++
#             (0.05, 0.5,64.06),  # 中等权重82.15(64.06)
#             (0.05, 0.05,82.85),  # 87.60(82.85)
#             (0.1, 1.0,59.44),  # 稍大权重79.22(59.44)
#             (0.2, 2.0,54.95),  # 大权重73.3(54.95)
#             (0.5, 5.0,48.34),  # 很大权重\62.43(48.34)
            (0.05, 0.05, 82.25),  # 87.60(82.85)前面那个有噪声的attack acc不满意

#             # 无噪声
#             (0.05, 0.5, 68.28),  # 中等权重82.15(64.06)
#             (0.05, 0.05, 83.54),  # 87.60(82.85)
#             (0.1, 1.0, 60.57),  # 稍大权重79.22(59.44)
#             (0.2, 2.0, 52.49),  # 大权重73.3(54.95)
#             (0.5, 5.0, 41.66),  # 很大权重\62.43(48.34)

# 最终的实验数据
# 有噪声三组数据
            (0.05, 0.05, 82.25),  # 87.60(82.85)前面那个有噪声的attack acc不满意
            (0.05, 0.5, 64.06),  # 中等权重82.15(64.06)
            (0.2, 2.0, 54.95),  # 大权重73.3(54.95)
# 无噪声
            (0.05, 0.05, 83.54),  # 87.60(82.85)
            (0.05, 0.5, 68.28),  # 中等权重82.15(64.06)
            (0.2, 2.0, 52.49),  # 大权重73.3(54.95)

        ]
        #CelebA训练流程
        for (a1, a2, ac) in hp_ac_list:
            print("a1:", a1, "a2:", a2, "test_acc:", ac)
            #目标模型（T）加载
            T = model.VGG16(1000, True)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, f"{args.dataset}/",args.defense,
                                  "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))

            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)

            print("---------------------Training [%s]------------------------------" % stage)

            dataset, dataloader = utils.init_dataloader(loaded_args, file_path, batch_size, mode="gan")
            #生成器（G）与判别器（DG）初始化
            G = Generator(z_dim)
            DG = MinibatchDiscriminator()
            G = torch.nn.DataParallel(G).cuda()
            DG = torch.nn.DataParallel(DG).cuda()

            dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
            g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

            entropy = HLoss()

            step = 0
            #训练循环
            for epoch in range(0, epochs):
                start = time.time()
                _, unlabel_loader1 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)
                _, unlabel_loader2 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)

                for i, imgs in enumerate(dataloader):
                    current_iter = epoch * len(dataloader) + i + 1

                    step += 1
                    imgs = imgs.cuda()
                    bs = imgs.size(0)
                    # x_unlabel = unlabel_loader1.next()
                    # x_unlabel2 = unlabel_loader2.next()
                    x_unlabel = next(unlabel_loader1)
                    x_unlabel2 = next(unlabel_loader2)

                    freeze(G)
                    unfreeze(DG)

                    z = torch.randn(bs, z_dim).cuda()
                    f_imgs = G(z)

                    y_prob = T(imgs)[-1]

                    targetprobs = nn.functional.softmax(y_prob, dim=1)
                    # print(entropy(targetprobs))

                    y = torch.argmax(y_prob, dim=1).view(-1)

                    _, output_label = DG(imgs)
                    _, output_unlabel = DG(x_unlabel)
                    _, output_fake = DG(f_imgs)

                    loss_lab = softXEnt(output_label, y_prob)
                    loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel)))
                                        - torch.mean(log_sum_exp(output_unlabel))
                                        + torch.mean(F.softplus(log_sum_exp(output_fake))))
                    dg_loss = loss_lab + loss_unlab

                    acc = torch.mean((output_label.max(1)[1] == y).float())

                    dg_optimizer.zero_grad()
                    dg_loss.backward()
                    dg_optimizer.step()

                    # train G
                    if step % n_critic == 0:
                        freeze(DG)
                        unfreeze(G)
                        z = torch.randn(bs, z_dim).cuda()
                        f_imgs = G(z)
                        mom_gen, output_fake = DG(f_imgs)
                        mom_unlabel, _ = DG(x_unlabel2)

                        mom_gen = torch.mean(mom_gen, dim=0)
                        mom_unlabel = torch.mean(mom_unlabel, dim=0)

                        Hloss = entropy(output_fake)
                        g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                        g_optimizer.zero_grad()
                        g_loss.backward()
                        g_optimizer.step()

                end = time.time()
                interval = end - start

                print("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (
                    epoch, interval, dg_loss, g_loss,
                    acc))

                if epoch + 1 >= 100 and (epoch + 1) % 10 == 0:
                    Gpath = os.path.join(save_model_dir, "{}_G_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
                    Dpath = os.path.join(save_model_dir, "{}_D_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)

                    torch.save({'state_dict': G.state_dict()}, Gpath)
                    torch.save({'state_dict': DG.state_dict()}, Dpath)

    elif args.dataset == 'mnist':
        hp_ac_list = [
            # # mnist-coco
            (1, 50, 99.51),
        ]
        for (a1, a2, ac) in hp_ac_list:
            print("a1:", a1, "a2:", a2, "test_acc:", ac)
            T = model.MCNN(5)
            T = torch.nn.DataParallel(T).cuda()
            path_T = os.path.join(args.model_path, f"{args.dataset}",args.defense,
                                  "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, ac))
            ckp_T = torch.load(path_T)
            T.load_state_dict(ckp_T['state_dict'], strict=False)
            dataset, dataloader = utils.init_dataloader(loaded_args, file_path, batch_size, mode="gan")

            G = GeneratorMNIST(z_dim)
            G = torch.nn.DataParallel(G).cuda()
            DG = MinibatchDiscriminator_MNIST()
            DG = torch.nn.DataParallel(DG).cuda()

            dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
            g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

            entropy = HLoss()

            step = 0
            for epoch in range(0, epochs):
                start = time.time()
                _, unlabel_loader1 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)
                _, unlabel_loader2 = init_dataloader(loaded_args, file_path, batch_size, mode="gan", iterator=True)

                for i, imgs in enumerate(dataloader):
                    current_iter = epoch * len(dataloader) + i + 1

                    step += 1
                    imgs = imgs.cuda()
                    bs = imgs.size(0)
                    x_unlabel = unlabel_loader1.next()
                    x_unlabel2 = unlabel_loader2.next()

                    freeze(G)
                    unfreeze(DG)

                    z = torch.randn(bs, z_dim).cuda()
                    f_imgs = G(z)

                    y_prob = T(imgs)[-1]

                    targetprobs = nn.functional.softmax(y_prob, dim=1)
                    # print(entropy(targetprobs))

                    y = torch.argmax(y_prob, dim=1).view(-1)

                    _, output_label = DG(imgs)
                    _, output_unlabel = DG(x_unlabel)
                    _, output_fake = DG(f_imgs)

                    loss_lab = softXEnt(output_label, y_prob)
                    loss_unlab = 0.5 * (torch.mean(F.softplus(log_sum_exp(output_unlabel)))
                                        - torch.mean(log_sum_exp(output_unlabel))
                                        + torch.mean(F.softplus(log_sum_exp(output_fake))))
                    dg_loss = loss_lab + loss_unlab

                    acc = torch.mean((output_label.max(1)[1] == y).float())

                    dg_optimizer.zero_grad()
                    dg_loss.backward()
                    dg_optimizer.step()

                    # train G
                    if step % n_critic == 0:
                        freeze(DG)
                        unfreeze(G)
                        z = torch.randn(bs, z_dim).cuda()
                        f_imgs = G(z)
                        mom_gen, output_fake = DG(f_imgs)
                        mom_unlabel, _ = DG(x_unlabel2)

                        mom_gen = torch.mean(mom_gen, dim=0)
                        mom_unlabel = torch.mean(mom_unlabel, dim=0)

                        Hloss = entropy(output_fake)
                        g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                        g_optimizer.zero_grad()
                        g_loss.backward()
                        g_optimizer.step()

                end = time.time()
                interval = end - start

                print("Epoch:%d \tTime:%.2f\tD_loss:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (
                    epoch, interval, dg_loss, g_loss,
                    acc))
                #模型保存
                if epoch + 1 >= 100 and (epoch + 1) % 10 == 0:
                    Gpath = os.path.join(save_model_dir, "{}_G_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)
                    Dpath = os.path.join(save_model_dir, "{}_D_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, ac)

                    torch.save({'state_dict': G.state_dict()}, Gpath)
                    torch.save({'state_dict': DG.state_dict()}, Dpath)




