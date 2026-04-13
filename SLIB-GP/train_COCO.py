# # import torch, os, engine, model, utils, sys
# # import torch.nn as nn
# # from torch.optim.lr_scheduler import MultiStepLR
# # from copy import deepcopy
# # import numpy as np
# # import collections
# # #用于训练深度学习模型的脚本，特别是针对一个特定的约束优化算法HSIC
# # import model
# # from util import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
# #
# # device = "cuda"
# #
# # #加载预训练模型的权重
# # def load_my_state_dict(net, state_dict):
# #     print("load nature model!!!")
# #     net_state = net.state_dict()
# #     for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
# #         # print(name, '---', old_name)
# #         net_state[name].copy_(old_param.data)
# #
# #
# # def load_feature_extractor(net, state_dict):#加载预训练的特征提取部分的权重
# #     print("load_pretrained_feature_extractor!!!")
# #     net_state = net.state_dict()
# #
# #     new_state_dict = collections.OrderedDict()
# #     for name, param in state_dict.items():
# #         if "running_var" in name:
# #             new_state_dict[name] = param#权重
# #             new_item = name.replace("running_var", "num_batches_tracked")
# #             new_state_dict[new_item] = torch.tensor(0)
# #         else:
# #             new_state_dict[name] = param
# #
# #     for ((name, param), (new_name, mew_param)) in zip(net_state.items(), new_state_dict.items()):
# #         if "classifier" in new_name:
# #             break
# #         if "num_batches_tracked" in new_name:
# #             continue
# #         # print(name, '---', new_name)
# #         net_state[name].copy_(mew_param.data)
# #
# #
# # def main(args, loaded_args, trainloader, testloader):
# #     n_classes = loaded_args["dataset"]["n_classes"]
# #     model_name = loaded_args["dataset"]["model_name"]
# #     weight_decay = loaded_args[model_name]["weight_decay"]
# #     momentum = loaded_args[model_name]["momentum"]
# #     n_epochs = loaded_args[model_name]["epochs"]
# #     lr = loaded_args[model_name]["lr"]
# #     milestones = loaded_args[model_name]["adjust_epochs"]
# #
# #     hp_list = [
# #         # (15, 75),
# #         (10, 50),
# #         (5, 50),
# #
# #     ]
# #
# #     criterion = nn.CrossEntropyLoss().cuda()
# #
# #     for i, (a1, a2) in enumerate(hp_list):
# #         print("a1:", a1, "a2:", a2)
# #
# #         if model_name == "VGG16" or model_name == "reg":
# #             net = model.VGG16(n_classes, hsic_training=args.hsic_training, dataset=args.dataset)
# #
# #             # load_pretrained_feature_extractor = True
# #             load_pretrained_feature_extractor = args.load_pretrained
# #             if load_pretrained_feature_extractor:
# #                 pretrained_model_ckpt = "target_model/vgg16_bn-6c64b313.pth"
# #                 checkpoint = torch.load(pretrained_model_ckpt)
# #                 load_feature_extractor(net, checkpoint)
# #
# #         elif model_name == "ResNet":
# #             net = model.ResNetClsH(nclass=n_classes, resnetl=10)
# #             # net = model.ResNet18(n_classes=n_classes)
# #
# #         elif model_name == "MCNN":
# #             net = model.MCNN(n_classes)
# #         elif model_name == "LeNet":
# #             net = model.LeNet3(n_classes)
# #
# #         elif model_name == "SimpleCNN":
# #             net = model.Classifier(1, 128, n_classes)
# #
# #         optimizer = torch.optim.Adam(net.parameters(), lr)
# #
# #         net = torch.nn.DataParallel(net).to(device)
# #         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
# #
# #         best_ACC = -1
# #         for epoch in range(n_epochs):
# #             print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
# #             train_loss, train_acc = engine.train_HSIC(net, criterion, optimizer, trainloader, a1, a2, n_classes,
# #                                                       ktype=args.ktype,
# #                                                       hsic_training=args.hsic_training,measure=args.measure)
# #             test_loss, test_acc = engine.test_HSIC(net, criterion, testloader, a1, a2, n_classes, ktype=args.ktype,
# #                                                    hsic_training=args.hsic_training,measure=args.measure)
# #
# #             if test_acc > best_ACC:
# #                 best_ACC = test_acc
# #                 best_model = deepcopy(net)
# #             scheduler.step()
# #
# #         print("best acc:", best_ACC)
# #         utils.save_checkpoint({
# #             'state_dict': best_model.state_dict(),
# #         }, model_path, "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, best_ACC))
# #
# #
# # if __name__ == '__main__':
# #     from argparse import ArgumentParser
# #
# #     parser = ArgumentParser(description='train with BiDO')
# #     parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
# #     parser.add_argument('--measure', default='COCO', help='HSIC | COCO')
# #     parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
# #     parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
# #     parser.add_argument('--load_pretrained', default=False, help='load pretrained feature extractor', type=bool)
# #     parser.add_argument('--root_path', default='./', help='')
# #     parser.add_argument('--config_dir', default='./config', help='')
# #     parser.add_argument('--model_dir', default='./target_model', help='')
# #     args = parser.parse_args()
# #
# #     model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.measure)
# #     os.makedirs(model_path, exist_ok=True)
# #
# #     file = os.path.join(args.config_dir, args.dataset + ".json")
# #
# #     loaded_args = utils.load_json(json_file=file)
# #     model_name = loaded_args["dataset"]["model_name"]
# #
# #     train_file = loaded_args['dataset']['train_file']
# #     test_file = loaded_args['dataset']['test_file']
# #
# #     trainloader = utils.init_dataloader(loaded_args, train_file, mode="train")
# #     testloader = utils.init_dataloader(loaded_args, test_file, mode="test")
# #
# #     main(args, loaded_args, trainloader, testloader)
# #
#
#
# # 断点修改
# import torch, os, engine, model, utils, sys
# import atexit
# import torch.nn as nn
# from torch.optim.lr_scheduler import MultiStepLR
# from copy import deepcopy
# import numpy as np
# import collections
# #用于训练深度学习模型的脚本，特别是针对一个特定的约束优化算法HSIC
# import model
# from util import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
#
# device = "cuda"
#
# #加载预训练模型的权重
# def load_my_state_dict(net, state_dict):
#     print("load nature model!!!")
#     net_state = net.state_dict()
#     for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
#         # print(name, '---', old_name)
#         net_state[name].copy_(old_param.data)
#
#
# def load_feature_extractor(net, state_dict):#加载预训练的特征提取部分的权重
#     print("load_pretrained_feature_extractor!!!")
#     net_state = net.state_dict()
#
#     new_state_dict = collections.OrderedDict()
#     for name, param in state_dict.items():
#         if "running_var" in name:
#             new_state_dict[name] = param#权重
#             new_item = name.replace("running_var", "num_batches_tracked")
#             new_state_dict[new_item] = torch.tensor(0)
#         else:
#             new_state_dict[name] = param
#
#     for ((name, param), (new_name, mew_param)) in zip(net_state.items(), new_state_dict.items()):
#         if "classifier" in new_name:
#             break
#         if "num_batches_tracked" in new_name:
#             continue
#         # print(name, '---', new_name)
#         net_state[name].copy_(mew_param.data)
#
#
# def main(args, loaded_args, trainloader, testloader):
#     n_classes = loaded_args["dataset"]["n_classes"]
#     model_name = loaded_args["dataset"]["model_name"]
#     weight_decay = loaded_args[model_name]["weight_decay"]
#     momentum = loaded_args[model_name]["momentum"]
#     n_epochs = loaded_args[model_name]["epochs"]
#     lr = loaded_args[model_name]["lr"]
#     milestones = loaded_args[model_name]["adjust_epochs"]
#
#     hp_list = [
#         (15, 75),
#         (10, 50),
#         (5, 50),
#         # (0.05, 0.50),  # 轻度正则（目标≈50%场景）
#         # (0.10, 0.90),  # 中等正则（目标≈70%场景）
#         # (0.20, 1.50),  # 强正则（目标≈80%场景）
#
#     ]
#
#     criterion = nn.CrossEntropyLoss().cuda()
#
#     # 进程结束自动保存
#     autosave_state = {"epoch": 0, "best_acc": -1.0, "model_state": None}
#     def _autosave_on_exit():
#         try:
#             if autosave_state["model_state"] is not None:
#                 utils.save_checkpoint({
#                     'state_dict': autosave_state["model_state"],
#                     'epoch': autosave_state["epoch"],
#                     'best_acc': autosave_state["best_acc"],
#                     'args': vars(args),
#                 }, model_path, 'last_autosave.tar')
#                 print(f"\n⚠️  进程结束触发自动保存: last_autosave.tar (epoch={autosave_state['epoch']}, best_acc={autosave_state['best_acc']:.2f})")
#         except Exception:
#             pass
#     atexit.register(_autosave_on_exit)
#
#     for i, (a1, a2) in enumerate(hp_list):
#         print("a1:", a1, "a2:", a2)
#
#         if model_name == "VGG16" or model_name == "reg":
#             net = model.VGG16(n_classes, hsic_training=args.hsic_training, dataset=args.dataset)
#
#             load_pretrained_feature_extractor = bool(getattr(args, 'pretrained', False) or getattr(args, 'load_pretrained', False))
#             if load_pretrained_feature_extractor:
#                 pretrained_model_ckpt = "target_model/vgg16_bn-6c64b313.pth"
#                 checkpoint = torch.load(pretrained_model_ckpt)
#                 load_feature_extractor(net, checkpoint)
#
#         elif model_name == "ResNet":
#             net = model.ResNetClsH(nclass=n_classes, resnetl=10)
#             # net = model.ResNet18(n_classes=n_classes)
#
#         elif model_name == "MCNN":
#             net = model.MCNN(n_classes)
#         elif model_name == "LeNet":
#             net = model.LeNet3(n_classes)
#
#         elif model_name == "SimpleCNN":
#             net = model.Classifier(1, 128, n_classes)
#
#         optimizer = torch.optim.Adam(net.parameters(), lr)
#
#         net = torch.nn.DataParallel(net).to(device)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
#
#         best_ACC = -1
#         start_epoch = 0
#
#         # 解析恢复训练
#         resume_path = None
#         did_resume = False
#         if getattr(args, 'resume', ''):
#             if args.resume == 'auto':
#                 last_path = os.path.join(model_path, 'last.tar')
#                 auto_path = os.path.join(model_path, 'last_autosave.tar')
#                 if os.path.isfile(last_path):
#                     resume_path = last_path
#                 elif os.path.isfile(auto_path):
#                     resume_path = auto_path
#             else:
#                 resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(model_path, args.resume)
#         else:
#             # 未指定 --resume，尝试自动检测最近保存点
#             last_path = os.path.join(model_path, 'last.tar')
#             auto_path = os.path.join(model_path, 'last_autosave.tar')
#             if os.path.isfile(last_path):
#                 resume_path = last_path
#                 print(f"ℹ️ 未指定 --resume，自动选择: {os.path.basename(last_path)}")
#             elif os.path.isfile(auto_path):
#                 resume_path = auto_path
#                 print(f"ℹ️ 未指定 --resume，自动选择: {os.path.basename(auto_path)}")
#
#         if resume_path and os.path.isfile(resume_path):
#             try:
#                 ckpt = torch.load(resume_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
#                 if 'state_dict' in ckpt:
#                     net.load_state_dict(ckpt['state_dict'], strict=False)
#                 if 'epoch' in ckpt:
#                     start_epoch = int(ckpt['epoch'])
#                 if 'best_acc' in ckpt:
#                     best_ACC = float(ckpt['best_acc'])
#                 if 'optimizer' in ckpt:
#                     try:
#                         optimizer.load_state_dict(ckpt['optimizer'])
#                     except Exception:
#                         print('⚠️ 恢复优化器状态失败，使用新优化器')
#                 if 'scheduler' in ckpt:
#                     try:
#                         scheduler.load_state_dict(ckpt['scheduler'])
#                     except Exception:
#                         print('⚠️ 恢复调度器状态失败，使用新调度器')
#                 print(
#                     f"🔁 已从 {os.path.basename(resume_path)} 恢复: start_epoch={start_epoch}, best_acc={best_ACC:.2f}")
#                 did_resume = True
#             except Exception as e:
#                 print(f"⚠️ 恢复失败 ({resume_path}): {e}")
#             else:
#                 if did_resume:
#                     pass
#                 elif getattr(args, 'resume', ''):
#                     print(
#                         f"ℹ️ 未找到可用的恢复文件: resume='{args.resume}', 期望位置: {resume_path or model_path}. 将从 epoch 1 开始")
#                 else:
#                     print("ℹ️ 未检测到保存点，默认从 epoch 1 开始")
#
#         for epoch in range(start_epoch, n_epochs):
#             print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
#             try:
#                 train_loss, train_acc = engine.train_HSIC(net, criterion, optimizer, trainloader, a1, a2, n_classes,
#                                                           ktype=args.ktype,
#                                                           hsic_training=args.hsic_training,measure=args.measure)
#                 test_loss, test_acc = engine.test_HSIC(net, criterion, testloader, a1, a2, n_classes, ktype=args.ktype,
#                                                        hsic_training=args.hsic_training,measure=args.measure)
#             except RuntimeError as e:
#                 print(f"\n❌ 运行时错误 (epoch {epoch+1}): {e}")
#                 crash_name = f"crash_epoch_{epoch+1:03d}.tar"
#                 try:
#                     utils.save_checkpoint({
#                         'state_dict': net.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'scheduler': scheduler.state_dict(),
#                         'epoch': epoch + 1,
#                         'best_acc': best_ACC,
#                         'args': vars(args),
#                         'error': str(e),
#                     }, model_path, crash_name)
#                     print(f"💾 已保存崩溃快照: {crash_name}")
#                 except Exception:
#                     print("⚠️ 保存崩溃快照失败")
#                 raise
#             except KeyboardInterrupt:
#                 print(f"\n⛔ 捕获到中断信号 (epoch {epoch+1})，正在保存当前进度...")
#                 crash_name = f"interrupt_epoch_{epoch+1:03d}.tar"
#                 try:
#                     utils.save_checkpoint({
#                         'state_dict': net.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'scheduler': scheduler.state_dict(),
#                         'epoch': epoch + 1,
#                         'best_acc': best_ACC,
#                         'args': vars(args),
#                     }, model_path, crash_name)
#                     print(f"💾 已保存中断快照: {crash_name}")
#                 except Exception:
#                     print("⚠️ 保存中断快照失败")
#                 raise
#
#             if test_acc > best_ACC:
#                 best_ACC = test_acc
#                 best_model = deepcopy(net)
#             scheduler.step()
#
#             # 更新自动保存信息与周期性保存
#             autosave_state["epoch"] = epoch + 1
#             autosave_state["best_acc"] = best_ACC
#             autosave_state["model_state"] = net.state_dict()
#             try:
#                 if (epoch + 1) % getattr(args, 'checkpoint_interval', 5) == 0:
#                     utils.save_checkpoint({
#                         'state_dict': net.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'scheduler': scheduler.state_dict(),
#                         'epoch': epoch + 1,
#                         'best_acc': best_ACC,
#                         'args': vars(args),
#                     }, model_path, 'last.tar')
#                     print(f"💾 自动保存: last.tar (epoch={epoch+1}, best_acc={best_ACC:.2f})")
#             except Exception:
#                 print("⚠️ 自动保存失败 (last.tar)")
#
#         print("best acc:", best_ACC)
#         utils.save_checkpoint({
#             'state_dict': best_model.state_dict(),
#         }, model_path, "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, best_ACC))
#
#
# if __name__ == '__main__':
#     from argparse import ArgumentParser
#
#     parser = ArgumentParser(description='train with BiDO')
#     parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
#     parser.add_argument('--measure', default='COCO', help='HSIC | COCO')
#     parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
#     parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
#     # 预训练控制
#     parser.add_argument('--pretrained', action='store_true', default=False, help='Load pretrained feature extractor (default False)')
#     parser.add_argument('--pretrained_from_config', action='store_true', default=False, help='Use config file to decide pretrained loading')
#     parser.add_argument('--root_path', default='./', help='')
#     parser.add_argument('--config_dir', default='./config', help='')
#     parser.add_argument('--model_dir', default='./target_model', help='')
#     parser.add_argument('--checkpoint_interval', type=int, default=5, help='Epoch interval for periodic autosave')
#     parser.add_argument('--resume', default='', help="Resume from checkpoint path, or 'auto' to pick latest")
#     args = parser.parse_args()
#
#     model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.measure)
#     os.makedirs(model_path, exist_ok=True)
#
#     file = os.path.join(args.config_dir, args.dataset + ".json")
#
#     loaded_args = utils.load_json(json_file=file)
#     model_name = loaded_args["dataset"]["model_name"]
#
#     train_file = loaded_args['dataset']['train_file']
#     test_file = loaded_args['dataset']['test_file']
#
#     trainloader = utils.init_dataloader(loaded_args, train_file, mode="train")
#     testloader = utils.init_dataloader(loaded_args, test_file, mode="test")
#
#     # 跟随配置文件决定是否加载预训练
#     if args.pretrained_from_config:
#         cfg_model = loaded_args.get(model_name, {}) if isinstance(loaded_args.get(model_name, {}), dict) else {}
#         cfg_dataset = loaded_args.get('dataset', {}) if isinstance(loaded_args.get('dataset', {}), dict) else {}
#         cfg_pretrained = cfg_model.get('pretrained', cfg_dataset.get('pretrained', False))
#         args.pretrained = bool(cfg_pretrained)
#
#     main(args, loaded_args, trainloader, testloader)
#
#效果不好，只有10点几，怀疑是不是之前东西加多了，反复归真一下

import torch, os, engine, model, utils, sys
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from copy import deepcopy
import numpy as np
import collections
#用于训练深度学习模型的脚本，特别是针对一个特定的约束优化算法HSIC
import model
from util import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

device = "cuda"

#加载预训练模型的权重
def load_my_state_dict(net, state_dict):
    print("load nature model!!!")
    net_state = net.state_dict()
    for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
        # print(name, '---', old_name)
        net_state[name].copy_(old_param.data)


def load_feature_extractor(net, state_dict):#加载预训练的特征提取部分的权重
    print("load_pretrained_feature_extractor!!!")
    net_state = net.state_dict()

    new_state_dict = collections.OrderedDict()
    for name, param in state_dict.items():
        if "running_var" in name:
            new_state_dict[name] = param#权重
            new_item = name.replace("running_var", "num_batches_tracked")
            new_state_dict[new_item] = torch.tensor(0)
        else:
            new_state_dict[name] = param

    for ((name, param), (new_name, mew_param)) in zip(net_state.items(), new_state_dict.items()):
        if "classifier" in new_name:
            break
        if "num_batches_tracked" in new_name:
            continue
        # print(name, '---', new_name)
        net_state[name].copy_(mew_param.data)


def main(args, loaded_args, trainloader, testloader):
    n_classes = loaded_args["dataset"]["n_classes"]
    model_name = loaded_args["dataset"]["model_name"]
    weight_decay = loaded_args[model_name]["weight_decay"]
    momentum = loaded_args[model_name]["momentum"]
    n_epochs = loaded_args[model_name]["epochs"]
    lr = loaded_args[model_name]["lr"]
    milestones = loaded_args[model_name]["adjust_epochs"]

    hp_list = [
        # (1.5, 7.5),
        # (0.2, 1.5),
        # (10, 50),
        # (5, 50),
        (15, 75),
    ]


    criterion = nn.CrossEntropyLoss().cuda()



    for i, (a1, a2) in enumerate(hp_list):
        print("a1:", a1, "a2:", a2)

        if model_name == "VGG16" or model_name == "reg":
            net = model.VGG16(n_classes, hsic_training=args.hsic_training, dataset=args.dataset)

            load_pretrained_feature_extractor = True
            if load_pretrained_feature_extractor:
                pretrained_model_ckpt = "target_model/vgg16_bn-6c64b313.pth"
                checkpoint = torch.load(pretrained_model_ckpt)
                load_feature_extractor(net, checkpoint)

        elif model_name == "ResNet":
            net = model.ResNetClsH(nclass=n_classes, resnetl=10)
            # net = model.ResNet18(n_classes=n_classes)

        elif model_name == "MCNN":
            net = model.MCNN(n_classes)
        elif model_name == "LeNet":
            net = model.LeNet3(n_classes)

        elif model_name == "SimpleCNN":
            net = model.Classifier(1, 128, n_classes)

        optimizer = torch.optim.Adam(net.parameters(), lr)

        net = torch.nn.DataParallel(net).to(device)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

        best_ACC = -1
        for epoch in range(n_epochs):
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
            train_loss, train_acc = engine.train_HSIC(net, criterion, optimizer, trainloader, a1, a2, n_classes,
                                                      ktype=args.ktype,
                                                      hsic_training=args.hsic_training,measure='COCO')
            test_loss, test_acc = engine.test_HSIC(net, criterion, testloader, a1, a2, n_classes, ktype=args.ktype,
                                                   hsic_training=args.hsic_training,measure='COCO')

            if test_acc > best_ACC:
                best_ACC = test_acc
                best_model = deepcopy(net)
            scheduler.step()

        print("best acc:", best_ACC)
        utils.save_checkpoint({
            'state_dict': best_model.state_dict(),
        }, model_path, "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, best_ACC))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train with BiDO')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
    parser.add_argument('--measure', default='COCO', help='HSIC | COCO')
    parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
    parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
    parser.add_argument('--root_path', default='./', help='')
    parser.add_argument('--config_dir', default='./config', help='')
    parser.add_argument('--model_dir', default='./target_model', help='')
    args = parser.parse_args()

    model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.measure)
    os.makedirs(model_path, exist_ok=True)

    file = os.path.join(args.config_dir, args.dataset + ".json")

    loaded_args = utils.load_json(json_file=file)
    model_name = loaded_args["dataset"]["model_name"]

    train_file = loaded_args['dataset']['train_file']
    test_file = loaded_args['dataset']['test_file']

    trainloader = utils.init_dataloader(loaded_args, train_file, mode="train")
    testloader = utils.init_dataloader(loaded_args, test_file, mode="test")

    main(args, loaded_args, trainloader, testloader)

#加上断点



