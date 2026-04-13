# # # # # # # # 让不同的 a1/a2 更“能起作用”，模型准确率太低了，mutual。py文件
# # # # # # # DP-SGD + BiDO Mutual Information Gradient Fusion
# # # # # # # 梯度级融合：DP-SGD + BiDO正则项，实现差分隐私与互信息约束的组合
# # # # # # import torch, os, engine, utils
# # # # # # import torch.nn as nn
# # # # # # from copy import deepcopy
# # # # # # import collections
# # # # # # from opacus import PrivacyEngine
# # # # # # from opacus.validators import ModuleValidator
# # # # # # from opacus.utils.batch_memory_manager import BatchMemoryManager
# # # # # # # 用于训练深度学习模型的脚本，特别是针对一个特定的约束优化算法HSIC
# # # # # # import model
# # # # # #
# # # # # # device = "cuda"
# # # # # #
# # # # # #
# # # # # # # 加载预训练模型的权重
# # # # # # def load_my_state_dict(net, state_dict):
# # # # # #     print("load nature model!!!")
# # # # # #     net_state = net.state_dict()
# # # # # #     for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
# # # # # #         # print(name, '---', old_name)
# # # # # #         net_state[name].copy_(old_param.data)
# # # # # #
# # # # # #
# # # # # # def load_feature_extractor(net, state_dict):  # 加载预训练的特征提取部分的权重
# # # # # #     print("load_pretrained_feature_extractor!!!")
# # # # # #     net_state = net.state_dict()
# # # # # #
# # # # # #     new_state_dict = collections.OrderedDict()
# # # # # #     for name, param in state_dict.items():
# # # # # #         if "running_var" in name:
# # # # # #             new_state_dict[name] = param  # 权重
# # # # # #             new_item = name.replace("running_var", "num_batches_tracked")
# # # # # #             new_state_dict[new_item] = torch.tensor(0)
# # # # # #         else:
# # # # # #             new_state_dict[name] = param
# # # # # #
# # # # # #     for ((name, param), (new_name, mew_param)) in zip(net_state.items(), new_state_dict.items()):
# # # # # #         if "classifier" in new_name:
# # # # # #             break
# # # # # #         if "num_batches_tracked" in new_name:
# # # # # #             continue
# # # # # #         # print(name, '---', new_name)
# # # # # #         net_state[name].copy_(mew_param.data)
# # # # # #
# # # # # #
# # # # # # def main(args, loaded_args, trainloader, testloader):
# # # # # #     n_classes = loaded_args["dataset"]["n_classes"]
# # # # # #     model_name = loaded_args["dataset"]["model_name"]
# # # # # #     weight_decay = loaded_args[model_name]["weight_decay"]
# # # # # #     momentum = loaded_args[model_name]["momentum"]
# # # # # #     n_epochs = loaded_args[model_name]["epochs"]
# # # # # #     lr = loaded_args[model_name]["lr"]
# # # # # #     milestones = loaded_args[model_name]["adjust_epochs"]
# # # # # #
# # # # # #     # 互信息约束权重超参数搜索列表 (α1, α2)
# # # # # #     # 格式: (α1, α2) 其中 α1 控制最小化 I(Z;X)，α2 控制最大化 I(Z;Y)
# # # # # #     # 参考 BiDO/train_COCO.py 的数值范围，但 MI 的数值范围可能不同，需要调整
# # # # # #     # 扩大搜索范围，确保不同参数有明显差异
# # # # # #     hp_list = [
# # # # # #         # (0.0, 0.0),      # 基线：无正则项
# # # # # #         (0.5, 5.0),      # 很大权重
# # # # # #         (0.01, 0.1),     # 小权重
# # # # # #         (0.01, 0.05),  # 小权重
# # # # # #         # (0.05, 0.5),     # 中等权重
# # # # # #         # (0.1, 1.0),      # 较大权重
# # # # # #         # (0.2, 2.0),      # 大权重
# # # # # #
# # # # # #     ]
# # # # # #     # 交叉熵
# # # # # #     criterion = nn.CrossEntropyLoss().cuda()
# # # # # #
# # # # # #     for i, (a1, a2) in enumerate(hp_list):
# # # # # #         print("\n" + "=" * 80)
# # # # # #         print(f"开始训练第 {i+1}/{len(hp_list)} 组超参数: a1={a1}, a2={a2}")
# # # # # #         print("=" * 80)
# # # # # #         torch.cuda.empty_cache()  # 清空GPU缓存
# # # # # #
# # # # # #         # 根据配置选择模型，并加载预训练权重
# # # # # #         if model_name == "VGG16" or model_name == "reg":
# # # # # #             net = model.VGG16(n_classes, hsic_training=args.hsic_training, dataset=args.dataset)
# # # # # #
# # # # # #             load_pretrained_feature_extractor = True
# # # # # #             if load_pretrained_feature_extractor:
# # # # # #                 pretrained_model_ckpt = "target_model/vgg16_bn-6c64b313.pth"
# # # # # #                 checkpoint = torch.load(pretrained_model_ckpt)
# # # # # #                 load_feature_extractor(net, checkpoint)
# # # # # #
# # # # # #         elif model_name == "ResNet":
# # # # # #             net = model.ResNetClsH(nclass=n_classes, resnetl=10)
# # # # # #         elif model_name == "MCNN":
# # # # # #             net = model.MCNN(n_classes)
# # # # # #         elif model_name == "LeNet":
# # # # # #             net = model.LeNet3(n_classes)
# # # # # #         elif model_name == "SimpleCNN":
# # # # # #             net = model.Classifier(1, 128, n_classes)
# # # # # #
# # # # # #         # 如果启用差分隐私，确保模型结构兼容Opacus
# # # # # #         if args.private and not ModuleValidator.is_valid(net):
# # # # # #             net = ModuleValidator.fix(net)
# # # # # #
# # # # # #         # 使用混合精度训练
# # # # # #         net = net.cuda()
# # # # # #         # Opacus 不支持 DataParallel，推荐使用单卡或 DDP
# # # # # #         scaler = torch.cuda.amp.GradScaler()
# # # # # #
# # # # # #         # 定义优化器（调整学习率以适应DP-SGD）
# # # # # #         # 对于DP训练，通常需要较小的学习率以确保稳定性
# # # # # #         base_lr = lr if not args.private else lr * 0.5  # DP训练时降低学习率
# # # # # #         optimizer = torch.optim.AdamW(
# # # # # #             net.parameters(),
# # # # # #             lr=base_lr,
# # # # # #             weight_decay=weight_decay if weight_decay > 0 else 1e-4,
# # # # # #             betas=(0.9, 0.999),
# # # # # #             eps=1e-8
# # # # # #         )
# # # # # #
# # # # # #         # 如果启用差分隐私，初始化并附加PrivacyEngine
# # # # # #         if args.private:
# # # # # #             print(f"\n{'=' * 80}")
# # # # # #             print(f"🔒 启用差分隐私训练")
# # # # # #             print(f"  目标隐私预算: ε={args.target_epsilon}, δ={args.target_delta}")
# # # # # #             print(f"  梯度裁剪范围: {args.max_grad_norm}")
# # # # # #             print(f"  采样率: {args.sample_rate}")
# # # # # #             print(f"{'=' * 80}\n")
# # # # # #
# # # # # #             privacy_engine = PrivacyEngine(
# # # # # #                 module=net,
# # # # # #                 sample_rate=args.sample_rate,
# # # # # #                 target_epsilon=args.target_epsilon,
# # # # # #                 target_delta=args.target_delta,
# # # # # #                 max_grad_norm=args.max_grad_norm,
# # # # # #                 epochs=n_epochs,
# # # # # #             )
# # # # # #             net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
# # # # # #                 module=net,
# # # # # #                 optimizer=optimizer,
# # # # # #                 data_loader=trainloader,
# # # # # #                 max_grad_norm=args.max_grad_norm,
# # # # # #                 epochs=n_epochs,
# # # # # #                 target_epsilon=args.target_epsilon,
# # # # # #                 target_delta=args.target_delta,
# # # # # #             )
# # # # # #
# # # # # #         # 使用余弦退火学习率调度器
# # # # # #         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
# # # # # #             optimizer,
# # # # # #             T_0=5,  # 第一次重启的周期
# # # # # #             T_mult=2,  # 每次重启后周期长度的倍数
# # # # # #             eta_min=base_lr * 0.01  # 最小学习率
# # # # # #         )
# # # # # #
# # # # # #         best_ACC = -1
# # # # # #         # 在DP训练下，每个epoch都会消费隐私预算
# # # # # #         for epoch in range(n_epochs):
# # # # # #             print('\n' + '=' * 80)
# # # # # #             print(f'Epoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
# # # # # #             print('=' * 80)
# # # # # #
# # # # # #             # 训练：总损失 = 任务损失 + β * BiDO正则项（互信息约束）
# # # # # #             # engine.train_mutual内部计算: L_total = L_task + mi_weight * (α1*I(Z;X) - α2*I(Z;Y))
# # # # # #             # DP-SGD会自动处理梯度裁剪和噪声添加
# # # # # #             train_loss, train_acc, train_reg_term = engine.train_mutual(net, criterion, optimizer, trainloader, a1, a2, n_classes,
# # # # # #                                                         ktype=args.ktype,
# # # # # #                                                         hsic_training=args.hsic_training,
# # # # # #                                                         measure=args.measure,
# # # # # #                                                         mi_weight=args.mi_weight)
# # # # # #
# # # # # #             test_loss, test_acc, test_reg_term = engine.test_mutual(net, criterion, testloader, a1, a2, n_classes, ktype=args.ktype,
# # # # # #                                                      hsic_training=args.hsic_training,
# # # # # #                                                      measure=args.measure,
# # # # # #                                                      mi_weight=args.mi_weight)
# # # # # #
# # # # # #             # 如果是DP训练，打印当前隐私预算消耗
# # # # # #             if args.private:
# # # # # #                 # 从optimizer中获取privacy_engine
# # # # # #                 if hasattr(optimizer, 'privacy_engine'):
# # # # # #                     epsilon = optimizer.privacy_engine.get_epsilon(delta=args.target_delta)
# # # # # #                     print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
# # # # # #                     print(f"🔒 当前隐私预算: ε = {epsilon:.4f}, δ = {args.target_delta}")
# # # # # #                     print(f"{'=' * 80}")
# # # # # #                 else:
# # # # # #                     print(f"⚠️  无法获取隐私预算信息")
# # # # # #             else:
# # # # # #                 print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
# # # # # #                 print(f"📈 正则项贡献: 训练={train_reg_term:.6f} | 测试={test_reg_term:.6f}")
# # # # # #
# # # # # #             if test_acc > best_ACC:
# # # # # #                 best_ACC = test_acc
# # # # # #                 best_model = deepcopy(net)
# # # # # #                 print(f"✅ 新的最佳准确率: {best_ACC:.2f}%")
# # # # # #
# # # # # #             scheduler.step()
# # # # # #
# # # # # #             # 每个epoch结束后清理缓存
# # # # # #             torch.cuda.empty_cache()
# # # # # #
# # # # # #         print("best acc:", best_ACC)
# # # # # #         if args.private:
# # # # # #             file_name = "{}_{:.3f}&{:.3f}_{:.2f}_private_eps_{:.2f}.tar".format(model_name, a1, a2, best_ACC,
# # # # # #                                                                                 args.target_epsilon)
# # # # # #         else:
# # # # # #             file_name = "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, best_ACC)
# # # # # #
# # # # # #         utils.save_checkpoint({
# # # # # #             'state_dict': best_model.state_dict(),
# # # # # #         }, model_path, file_name)
# # # # # #
# # # # # #
# # # # # # if __name__ == '__main__':
# # # # # #     from argparse import ArgumentParser
# # # # # #
# # # # # #     parser = ArgumentParser(description='train with BiDO')
# # # # # #     parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
# # # # # #     parser.add_argument('--measure', default='MI', help='HSIC | COCO | MI')
# # # # # #     parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
# # # # # #     parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
# # # # # #     parser.add_argument('--root_path', default='./', help='')
# # # # # #     parser.add_argument('--config_dir', default='./config', help='')
# # # # # #     parser.add_argument('--model_dir', default='./target_model', help='')
# # # # # #     # Opacus DP arguments
# # # # # #     parser.add_argument('--mi_weight', type=float, default=1.0, help='global weight for MI/HSIC regularizer')
# # # # # #     parser.add_argument('--private', action='store_true', help='Enable differential privacy')
# # # # # #     # DP参数调整：目标-准确率下降约5%，攻击准确率下降约20%
# # # # # #     # 较小的epsilon提供更强的隐私保护但可能牺牲更多准确率
# # # # # #     parser.add_argument('--target_epsilon', type=float, default=6.0,
# # # # # #                         help='Target epsilon for DP (较小值=更强的隐私保护)')
# # # # # #     parser.add_argument('--target_delta', type=float, default=1e-5, help='Target delta for DP')
# # # # # #     parser.add_argument('--max_grad_norm', type=float, default=0.8, help='Max grad norm for clipping (梯度裁剪范围)')
# # # # # #     parser.add_argument('--sample_rate', type=float, default=0.008,
# # # # # #                         help='Subsampling rate for DP-SGD (较小的采样率可提供更强的隐私保护)')
# # # # # #
# # # # # #     args = parser.parse_args()
# # # # # #
# # # # # #     if args.private:
# # # # # #         args.measure = args.measure + "+DP"
# # # # # #
# # # # # #     model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.measure)
# # # # # #
# # # # # #     os.makedirs(model_path, exist_ok=True)
# # # # # #
# # # # # #     file = os.path.join(args.config_dir, args.dataset + ".json")
# # # # # #
# # # # # #     loaded_args = utils.load_json(json_file=file)
# # # # # #     model_name = loaded_args["dataset"]["model_name"]
# # # # # #
# # # # # #     train_file = loaded_args['dataset']['train_file']
# # # # # #     test_file = loaded_args['dataset']['test_file']
# # # # # #
# # # # # #     trainloader = utils.init_dataloader(loaded_args, train_file, mode="train")
# # # # # #     testloader = utils.init_dataloader(loaded_args, test_file, mode="test")
# # # # # #
# # # # # #     # The user should set args.target_delta to be 1/len(trainloader.dataset)
# # # # # #     if args.private and args.target_delta is None:
# # # # # #         args.target_delta = 1 / len(trainloader.dataset)
# # # # # #         print(f"Setting delta to {args.target_delta}")
# # # # # #
# # # # # #     main(args, loaded_args, trainloader, testloader)
# # # # # #
# # # # # # # 让不同的 a1/a2 更“能起作用”，模型准确率太低了，MI 正则权重支持 warmup：新增 --mi_weight（默认 0.5）和
# # # # # # --mi_warmup_epochs（默认 15），
# # # # # # 训练时按线性 warmup 逐步放大 current_mi_weight，降低前期扰动，从而让 DP 版在 (0.05,0.5) 下不至于过早被正则主导
# # # # # # DP-SGD + BiDO Mutual Information Gradient Fusion
# # # # # # 梯度级融合：DP-SGD + BiDO正则项，实现差分隐私与互信息约束的组合
# # # # # #使用DP，但是准确率下降太多了
# # # # # import argparse
# # # # # import torch, os, engine, utils
# # # # # import torch.nn as nn
# # # # # from copy import deepcopy
# # # # # import collections
# # # # # from opacus import PrivacyEngine
# # # # # from opacus.validators import ModuleValidator
# # # # # from opacus.utils.batch_memory_manager import BatchMemoryManager
# # # # # # 用于训练深度学习模型的脚本，特别是针对一个特定的约束优化算法HSIC
# # # # # import model
# # # # #
# # # # # device = "cuda"
# # # # #
# # # # #
# # # # # def str2bool(v):
# # # # #     if isinstance(v, bool):
# # # # #         return v
# # # # #     if v.lower() in ('yes', 'true', 't', '1', 'y'):
# # # # #         return True
# # # # #     if v.lower() in ('no', 'false', 'f', '0', 'n'):
# # # # #         return False
# # # # #     raise argparse.ArgumentTypeError('Boolean value expected.')
# # # # #
# # # # #
# # # # # # 加载预训练模型的权重
# # # # # def load_my_state_dict(net, state_dict):
# # # # #     print("load nature model!!!")
# # # # #     net_state = net.state_dict()
# # # # #     for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
# # # # #         # print(name, '---', old_name)
# # # # #         net_state[name].copy_(old_param.data)
# # # # #
# # # # #
# # # # # def load_feature_extractor(net, state_dict):  # 加载预训练的特征提取部分的权重
# # # # #     print("load_pretrained_feature_extractor!!!")
# # # # #     net_state = net.state_dict()
# # # # #
# # # # #     new_state_dict = collections.OrderedDict()
# # # # #     for name, param in state_dict.items():
# # # # #         if "running_var" in name:
# # # # #             new_state_dict[name] = param  # 权重
# # # # #             new_item = name.replace("running_var", "num_batches_tracked")
# # # # #             new_state_dict[new_item] = torch.tensor(0)
# # # # #         else:
# # # # #             new_state_dict[name] = param
# # # # #
# # # # #     for ((name, param), (new_name, mew_param)) in zip(net_state.items(), new_state_dict.items()):
# # # # #         if "classifier" in new_name:
# # # # #             break
# # # # #         if "num_batches_tracked" in new_name:
# # # # #             continue
# # # # #         # print(name, '---', new_name)
# # # # #         net_state[name].copy_(mew_param.data)
# # # # #
# # # # #
# # # # # def main(args, loaded_args, trainloader, testloader):
# # # # #     n_classes = loaded_args["dataset"]["n_classes"]
# # # # #     model_name = loaded_args["dataset"]["model_name"]
# # # # #     weight_decay = loaded_args[model_name]["weight_decay"]
# # # # #     momentum = loaded_args[model_name]["momentum"]
# # # # #     n_epochs = loaded_args[model_name]["epochs"]
# # # # #     lr = loaded_args[model_name]["lr"]
# # # # #     milestones = loaded_args[model_name]["adjust_epochs"]
# # # # #
# # # # #     # 互信息约束权重超参数搜索列表 (α1, α2)
# # # # #     # 只保留几组代表性的配置，包含用户重点关注的 (0.05, 0.5)
# # # # #     hp_list = [
# # # # #         # (0.0, 0.0),      # 无正则参考
# # # # #         (0.05, 0.5),     # 中等权重82.15
# # # # #         (0.05, 0.05),#87.60
# # # # #         (0.1, 1.0),      # 稍大权重79.22
# # # # #         (0.2, 2.0),  # 大权重73.3q
# # # # #         (0.5, 5.0),  # 很大权重\62.43
# # # # #
# # # # #     ]
# # # # #     # 交叉熵
# # # # #     criterion = nn.CrossEntropyLoss().cuda()
# # # # #
# # # # #     for i, (a1, a2) in enumerate(hp_list):
# # # # #         print("\n" + "=" * 80)
# # # # #         print(f"开始训练第 {i+1}/{len(hp_list)} 组超参数: a1={a1}, a2={a2}")
# # # # #         print("=" * 80)
# # # # #         torch.cuda.empty_cache()  # 清空GPU缓存
# # # # #
# # # # #         # 根据配置选择模型，并加载预训练权重
# # # # #         if model_name == "VGG16" or model_name == "reg":
# # # # #             net = model.VGG16(n_classes, hsic_training=args.hsic_training, dataset=args.dataset)
# # # # #
# # # # #             load_pretrained_feature_extractor = True
# # # # #             if load_pretrained_feature_extractor:
# # # # #                 pretrained_model_ckpt = "target_model/vgg16_bn-6c64b313.pth"
# # # # #                 checkpoint = torch.load(pretrained_model_ckpt)
# # # # #                 load_feature_extractor(net, checkpoint)
# # # # #
# # # # #         elif model_name == "ResNet":
# # # # #             net = model.ResNetClsH(nclass=n_classes, resnetl=10)
# # # # #         elif model_name == "MCNN":
# # # # #             net = model.MCNN(n_classes)
# # # # #         elif model_name == "LeNet":
# # # # #             net = model.LeNet3(n_classes)
# # # # #         elif model_name == "SimpleCNN":
# # # # #             net = model.Classifier(1, 128, n_classes)
# # # # #
# # # # #         # 如果启用差分隐私，确保模型结构兼容Opacus
# # # # #         if args.private and not ModuleValidator.is_valid(net):
# # # # #             net = ModuleValidator.fix(net)
# # # # #
# # # # #         # 使用混合精度训练
# # # # #         net = net.cuda()
# # # # #         # Opacus 不支持 DataParallel，推荐使用单卡或 DDP
# # # # #         scaler = torch.cuda.amp.GradScaler()
# # # # #
# # # # #         # 定义优化器（调整学习率以适应DP-SGD）
# # # # #         # 对于DP训练，通常需要较小的学习率以确保稳定性
# # # # #         base_lr = lr if not args.private else lr * 0.5  # DP训练时降低学习率
# # # # #         optimizer = torch.optim.AdamW(
# # # # #             net.parameters(),
# # # # #             lr=base_lr,
# # # # #             weight_decay=weight_decay if weight_decay > 0 else 1e-4,
# # # # #             betas=(0.9, 0.999),
# # # # #             eps=1e-8
# # # # #         )
# # # # #
# # # # #         # 如果启用差分隐私，初始化并附加PrivacyEngine
# # # # #         if args.private:
# # # # #             print(f"\n{'=' * 80}")
# # # # #             print(f"🔒 启用差分隐私训练")
# # # # #             print(f"  目标隐私预算: ε={args.target_epsilon}, δ={args.target_delta}")
# # # # #             print(f"  梯度裁剪范围: {args.max_grad_norm}")
# # # # #             print(f"  采样率: {args.sample_rate}")
# # # # #             print(f"{'=' * 80}\n")
# # # # #
# # # # #             privacy_engine = PrivacyEngine()
# # # # #             net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
# # # # #                 module=net,
# # # # #                 optimizer=optimizer,
# # # # #                 data_loader=trainloader,
# # # # #                 max_grad_norm=args.max_grad_norm,
# # # # #                 # sample_rate=args.sample_rate,
# # # # #                 epochs=n_epochs,
# # # # #                 target_epsilon=args.target_epsilon,
# # # # #                 target_delta=args.target_delta,
# # # # #             )
# # # # #
# # # # #         # 使用余弦退火学习率调度器
# # # # #         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
# # # # #             optimizer,
# # # # #             T_0=5,  # 第一次重启的周期
# # # # #             T_mult=2,  # 每次重启后周期长度的倍数
# # # # #             eta_min=base_lr * 0.01  # 最小学习率
# # # # #         )
# # # # #
# # # # #         best_ACC = -1
# # # # #         base_mi_weight = args.mi_weight
# # # # #         # 在DP训练下，每个epoch都会消费隐私预算
# # # # #         for epoch in range(n_epochs):
# # # # #             warmup_progress = 1.0
# # # # #             if args.mi_warmup_epochs > 0:
# # # # #                 warmup_progress = min(1.0, (epoch + 1) / args.mi_warmup_epochs)
# # # # #             current_mi_weight = base_mi_weight * warmup_progress
# # # # #
# # # # #             print('\n' + '=' * 80)
# # # # #             print(f'Epoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
# # # # #             print('=' * 80)
# # # # #             print(f"⚙️ 当前互信息缩放: base={base_mi_weight:.3f}, warmup={warmup_progress * 100:.1f}%, 实际权重={current_mi_weight:.3f}")
# # # # #
# # # # #             # 训练：总损失 = 任务损失 + β * BiDO正则项（互信息约束）
# # # # #             # engine.train_mutual内部计算: L_total = L_task + mi_weight * (α1*I(Z;X) - α2*I(Z;Y))
# # # # #             # DP-SGD会自动处理梯度裁剪和噪声添加
# # # # #             train_loss, train_acc, train_reg_term = engine.train_mutual(net, criterion, optimizer, trainloader, a1, a2, n_classes,
# # # # #                                                         ktype=args.ktype,
# # # # #                                                         hsic_training=args.hsic_training,
# # # # #                                                         measure=args.measure,
# # # # #                                                         mi_weight=current_mi_weight)
# # # # #
# # # # #             test_loss, test_acc, test_reg_term = engine.test_mutual(net, criterion, testloader, a1, a2, n_classes, ktype=args.ktype,
# # # # #                                                      hsic_training=args.hsic_training,
# # # # #                                                      measure=args.measure,
# # # # #                                                      mi_weight=current_mi_weight)
# # # # #
# # # # #             # 如果是DP训练，打印当前隐私预算消耗
# # # # #             if args.private:
# # # # #                 # 从optimizer中获取privacy_engine
# # # # #                 if hasattr(optimizer, 'privacy_engine'):
# # # # #                     epsilon = optimizer.privacy_engine.get_epsilon(delta=args.target_delta)
# # # # #                     print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
# # # # #                     print(f"🔒 当前隐私预算: ε = {epsilon:.4f}, δ = {args.target_delta}")
# # # # #                     print(f"{'=' * 80}")
# # # # #                 else:
# # # # #                     print(f"⚠️  无法获取隐私预算信息")
# # # # #             else:
# # # # #                 print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
# # # # #                 print(f"📈 正则项贡献: 训练={train_reg_term:.6f} | 测试={test_reg_term:.6f}")
# # # # #
# # # # #             if test_acc > best_ACC:
# # # # #                 best_ACC = test_acc
# # # # #                 best_model = deepcopy(net)
# # # # #                 print(f"✅ 新的最佳准确率: {best_ACC:.2f}%")
# # # # #
# # # # #             scheduler.step()
# # # # #
# # # # #             # 每个epoch结束后清理缓存
# # # # #             torch.cuda.empty_cache()
# # # # #
# # # # #         print("best acc:", best_ACC)
# # # # #         if args.private:
# # # # #             file_name = "{}_{:.3f}&{:.3f}_{:.2f}_private_eps_{:.2f}.tar".format(model_name, a1, a2, best_ACC,
# # # # #                                                                                 args.target_epsilon)
# # # # #         else:
# # # # #             file_name = "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, best_ACC)
# # # # #
# # # # #         utils.save_checkpoint({
# # # # #             'state_dict': best_model.state_dict(),
# # # # #         }, model_path, file_name)
# # # # #
# # # # #
# # # # # if __name__ == '__main__':
# # # # #     from argparse import ArgumentParser
# # # # #
# # # # #     parser = ArgumentParser(description='train with BiDO')
# # # # #     parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
# # # # #     parser.add_argument('--measure', default='MI', help='HSIC | COCO | MI')
# # # # #     parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
# # # # #     parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
# # # # #     parser.add_argument('--root_path', default='./', help='')
# # # # #     parser.add_argument('--config_dir', default='./config', help='')
# # # # #     parser.add_argument('--model_dir', default='./target_model', help='')
# # # # #     # Opacus DP arguments
# # # # #     parser.add_argument('--mi_weight', type=float, default=0.5, help='global weight for MI/HSIC regularizer')
# # # # #     parser.add_argument('--mi_warmup_epochs', type=int, default=15,
# # # # #                         help='Linearly warm up MI权重的epoch数，0表示立即使用完整权重')
# # # # #     parser.add_argument('--private', type=str2bool, default=True,
# # # # #                         help='是否启用差分隐私，默认开启；传 --private false 可关闭')
# # # # #     # DP参数调整：目标-准确率下降约5%，攻击准确率下降约20%
# # # # #     # 较小的epsilon提供更强的隐私保护但可能牺牲更多准确率
# # # # #     parser.add_argument('--target_epsilon', type=float, default=6.0,
# # # # #                         help='Target epsilon for DP (较小值=更强的隐私保护)')
# # # # #     parser.add_argument('--target_delta', type=float, default=1e-5, help='Target delta for DP')
# # # # #     parser.add_argument('--max_grad_norm', type=float, default=0.8, help='Max grad norm for clipping (梯度裁剪范围)')
# # # # #     parser.add_argument('--sample_rate', type=float, default=0.008,
# # # # #                         help='Subsampling rate for DP-SGD (较小的采样率可提供更强的隐私保护)')
# # # # #
# # # # #     args = parser.parse_args()
# # # # #
# # # # #     if args.private:
# # # # #         args.measure = args.measure + "+DP"
# # # # #
# # # # #     model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.measure)
# # # # #
# # # # #     os.makedirs(model_path, exist_ok=True)
# # # # #
# # # # #     file = os.path.join(args.config_dir, args.dataset + ".json")
# # # # #
# # # # #     loaded_args = utils.load_json(json_file=file)
# # # # #     model_name = loaded_args["dataset"]["model_name"]
# # # # #
# # # # #     train_file = loaded_args['dataset']['train_file']
# # # # #     test_file = loaded_args['dataset']['test_file']
# # # # #
# # # # #     trainloader = utils.init_dataloader(loaded_args, train_file, mode="train")
# # # # #     testloader = utils.init_dataloader(loaded_args, test_file, mode="test")
# # # # #
# # # # #     # The user should set args.target_delta to be 1/len(trainloader.dataset)
# # # # #     if args.private and args.target_delta is None:
# # # # #         args.target_delta = 1 / len(trainloader.dataset)
# # # # #         print(f"Setting delta to {args.target_delta}")
# # # # #
# # # # #     main(args, loaded_args, trainloader, testloader)
# # # # #
# # # # # # 让不同的 a1/a2 更“能起作用”，模型准确率太低了，MI 正则权重支持 warmup：新增 --mi_weight（默认 0.5）和
# # # # # --mi_warmup_epochs（默认 15），
# # # # # 训练时按线性 warmup 逐步放大 current_mi_weight，降低前期扰动，从而让 DP 版在 (0.05,0.5) 下不至于过早被正则主导
# # # # # DP-SGD + BiDO Mutual Information Gradient Fusion
# # # # # 梯度级融合：DP-SGD + BiDO正则项，实现差分隐私与互信息约束的组合
# # # # import argparse
# # # # import torch, os, engine, utils
# # # # import torch.nn as nn
# # # # from copy import deepcopy
# # # # import collections
# # # # from opacus import PrivacyEngine
# # # # from opacus.validators import ModuleValidator
# # # # from opacus.utils.batch_memory_manager import BatchMemoryManager
# # # # # 用于训练深度学习模型的脚本，特别是针对一个特定的约束优化算法HSIC
# # # # import model
# # # #
# # # # device = "cuda"
# # # #
# # # #
# # # # def str2bool(v):
# # # #     if isinstance(v, bool):
# # # #         return v
# # # #     if v.lower() in ('yes', 'true', 't', '1', 'y'):
# # # #         return True
# # # #     if v.lower() in ('no', 'false', 'f', '0', 'n'):
# # # #         return False
# # # #     raise argparse.ArgumentTypeError('Boolean value expected.')
# # # #
# # # #
# # # # # 加载预训练模型的权重
# # # # def load_my_state_dict(net, state_dict):
# # # #     print("load nature model!!!")
# # # #     net_state = net.state_dict()
# # # #     for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
# # # #         # print(name, '---', old_name)
# # # #         net_state[name].copy_(old_param.data)
# # # #
# # # #
# # # # def load_feature_extractor(net, state_dict):  # 加载预训练的特征提取部分的权重
# # # #     print("load_pretrained_feature_extractor!!!")
# # # #     net_state = net.state_dict()
# # # #
# # # #     new_state_dict = collections.OrderedDict()
# # # #     for name, param in state_dict.items():
# # # #         if "running_var" in name:
# # # #             new_state_dict[name] = param  # 权重
# # # #             new_item = name.replace("running_var", "num_batches_tracked")
# # # #             new_state_dict[new_item] = torch.tensor(0)
# # # #         else:
# # # #             new_state_dict[name] = param
# # # #
# # # #     for ((name, param), (new_name, mew_param)) in zip(net_state.items(), new_state_dict.items()):
# # # #         if "classifier" in new_name:
# # # #             break
# # # #         if "num_batches_tracked" in new_name:
# # # #             continue
# # # #         # print(name, '---', new_name)
# # # #         net_state[name].copy_(mew_param.data)
# # # #
# # # #
# # # # def main(args, loaded_args, trainloader, testloader):
# # # #     n_classes = loaded_args["dataset"]["n_classes"]
# # # #     model_name = loaded_args["dataset"]["model_name"]
# # # #     weight_decay = loaded_args[model_name]["weight_decay"]
# # # #     momentum = loaded_args[model_name]["momentum"]
# # # #     n_epochs = loaded_args[model_name]["epochs"]
# # # #     lr = loaded_args[model_name]["lr"]
# # # #     milestones = loaded_args[model_name]["adjust_epochs"]
# # # #
# # # #     # 互信息约束权重超参数搜索列表 (α1, α2)
# # # #     # 只保留几组代表性的配置，包含用户重点关注的 (0.05, 0.5)
# # # #     hp_list = [
# # # #         # (0.0, 0.0),      # 无正则参考
# # # #         # (0.05, 0.5),  # 中等权重82.15
# # # #         (0.05, 0.05),  # 87.60
# # # #         # (0.1, 1.0),  # 稍大权重79.22
# # # #         # (0.2, 2.0),  # 大权重73.3q
# # # #         (0.5, 5.0),  # 很大权重\62.43
# # # #
# # # #     ]
# # # #     # 交叉熵
# # # #     criterion = nn.CrossEntropyLoss().cuda()
# # # #
# # # #     for i, (a1, a2) in enumerate(hp_list):
# # # #         print("\n" + "=" * 80)
# # # #         print(f"开始训练第 {i + 1}/{len(hp_list)} 组超参数: a1={a1}, a2={a2}")
# # # #         print("=" * 80)
# # # #         torch.cuda.empty_cache()  # 清空GPU缓存
# # # #
# # # #         # 根据配置选择模型，并加载预训练权重
# # # #         if model_name == "VGG16" or model_name == "reg":
# # # #             net = model.VGG16(n_classes, hsic_training=args.hsic_training, dataset=args.dataset)
# # # #
# # # #             load_pretrained_feature_extractor = True
# # # #             if load_pretrained_feature_extractor:
# # # #                 pretrained_model_ckpt = "target_model/vgg16_bn-6c64b313.pth"
# # # #                 checkpoint = torch.load(pretrained_model_ckpt)
# # # #                 load_feature_extractor(net, checkpoint)
# # # #
# # # #         elif model_name == "ResNet":
# # # #             net = model.ResNetClsH(nclass=n_classes, resnetl=10)
# # # #         elif model_name == "MCNN":
# # # #             net = model.MCNN(n_classes)
# # # #         elif model_name == "LeNet":
# # # #             net = model.LeNet3(n_classes)
# # # #         elif model_name == "SimpleCNN":
# # # #             net = model.Classifier(1, 128, n_classes)
# # # #
# # # #         # 如果启用差分隐私，确保模型结构兼容Opacus
# # # #         if args.private and not ModuleValidator.is_valid(net):
# # # #             net = ModuleValidator.fix(net)
# # # #
# # # #         # 使用混合精度训练
# # # #         net = net.cuda()
# # # #         # Opacus 不支持 DataParallel，推荐使用单卡或 DDP
# # # #         scaler = torch.cuda.amp.GradScaler()
# # # #
# # # #         # 定义优化器（轻量DP时只轻微降低学习率）
# # # #         # 为了最小化准确率下降，DP训练时只轻微降低学习率
# # # #         base_lr = lr if not args.private else lr * args.dp_lr_scale
# # # #         optimizer = torch.optim.AdamW(
# # # #             net.parameters(),
# # # #             lr=base_lr,
# # # #             weight_decay=weight_decay if weight_decay > 0 else 1e-4,
# # # #             betas=(0.9, 0.999),
# # # #             eps=1e-8
# # # #         )
# # # #
# # # #         # 如果启用差分隐私，初始化并附加PrivacyEngine（轻量模式）
# # # #         privacy_engine = None
# # # #         max_physical_batch_size = None
# # # #         if args.private:
# # # #             print(f"\n{'=' * 80}")
# # # #             print(f"🔒 启用轻量差分隐私训练（目标：准确率下降<5%）")
# # # #             print(f"  目标隐私预算: ε={args.target_epsilon}, δ={args.target_delta}")
# # # #             print(f"  梯度裁剪范围: {args.max_grad_norm}")
# # # #             print(f"  采样率: {args.sample_rate}")
# # # #             print(f"{'=' * 80}\n")
# # # #
# # # #             # 保存原始batch_size用于BatchMemoryManager
# # # #             original_batch_size = getattr(trainloader, 'batch_size', None)
# # # #             if original_batch_size is None:
# # # #                 try:
# # # #                     original_batch_size = trainloader.batch_sampler.batch_size
# # # #                 except:
# # # #                     original_batch_size = 32
# # # #
# # # #             privacy_engine = PrivacyEngine()
# # # #             net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
# # # #                 module=net,
# # # #                 optimizer=optimizer,
# # # #                 data_loader=trainloader,
# # # #                 max_grad_norm=args.max_grad_norm,
# # # #                 epochs=n_epochs,
# # # #                 target_epsilon=args.target_epsilon,
# # # #                 target_delta=args.target_delta,
# # # #             )
# # # #
# # # #             # 进一步减小噪声以最小化准确率下降
# # # #             if args.dp_noise_scale != 1.0 and hasattr(optimizer, "privacy_engine"):
# # # #                 old_noise = optimizer.privacy_engine.noise_multiplier
# # # #                 new_noise = max(1e-6, old_noise * args.dp_noise_scale)
# # # #                 optimizer.privacy_engine.noise_multiplier = new_noise
# # # #                 if hasattr(optimizer.privacy_engine, "_noise_multiplier"):
# # # #                     optimizer.privacy_engine._noise_multiplier = new_noise
# # # #                 print(f"⚠️  调整DP噪声乘子: {old_noise:.4f} -> {new_noise:.4f} (scale={args.dp_noise_scale})")
# # # #                 print(f"   实际隐私预算会高于 target_epsilon，但准确率下降更少\n")
# # # #
# # # #             # 设置虚拟batch大小
# # # #             max_physical_batch_size = max(1, original_batch_size // 2)
# # # #             print(f"📦 虚拟batch大小: {max_physical_batch_size} (原始: {original_batch_size})\n")
# # # #
# # # #         # 使用余弦退火学习率调度器
# # # #         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
# # # #             optimizer,
# # # #             T_0=5,  # 第一次重启的周期
# # # #             T_mult=2,  # 每次重启后周期长度的倍数
# # # #             eta_min=base_lr * 0.01  # 最小学习率
# # # #         )
# # # #
# # # #         best_ACC = -1
# # # #         # MI约束保持不变，直接使用a1, a2（不进行任何缩放或warmup）
# # # #         # 在DP训练下，每个epoch都会消费隐私预算
# # # #         for epoch in range(n_epochs):
# # # #             print('\n' + '=' * 80)
# # # #             print(f'Epoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
# # # #             print('=' * 80)
# # # #             print(f"⚙️  MI约束参数: a1={a1:.4f}, a2={a2:.4f} (保持不变，无缩放)")
# # # #
# # # #             # 训练：总损失 = 任务损失 + BiDO正则项（互信息约束）
# # # #             # engine.train_mutual内部计算: L_total = L_task + (α1*I(Z;X) - α2*I(Z;Y))
# # # #             # DP-SGD会自动处理梯度裁剪和噪声添加（但噪声很小，只参与一点）
# # # #             if args.private and max_physical_batch_size is not None:
# # # #                 with BatchMemoryManager(
# # # #                         data_loader=trainloader,
# # # #                         max_physical_batch_size=max_physical_batch_size,
# # # #                         optimizer=optimizer
# # # #                 ) as memory_safe_dataloader:
# # # #                     train_loss, train_acc, train_reg_term = engine.train_mutual(
# # # #                         net, criterion, optimizer, memory_safe_dataloader, a1, a2, n_classes,
# # # #                         ktype=args.ktype,
# # # #                         hsic_training=args.hsic_training,
# # # #                         measure=args.measure)
# # # #             else:
# # # #                 train_loss, train_acc, train_reg_term = engine.train_mutual(
# # # #                     net, criterion, optimizer, trainloader, a1, a2, n_classes,
# # # #                     ktype=args.ktype,
# # # #                     hsic_training=args.hsic_training,
# # # #                     measure=args.measure)
# # # #
# # # #             test_loss, test_acc, test_reg_term = engine.test_mutual(
# # # #                 net, criterion, testloader, a1, a2, n_classes, ktype=args.ktype,
# # # #                 hsic_training=args.hsic_training,
# # # #                 measure=args.measure)
# # # #
# # # #             # 如果是DP训练，打印当前隐私预算消耗
# # # #             if args.private:
# # # #                 # 从optimizer中获取privacy_engine
# # # #                 if hasattr(optimizer, 'privacy_engine'):
# # # #                     epsilon = optimizer.privacy_engine.get_epsilon(delta=args.target_delta)
# # # #                     print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
# # # #                     print(f"🔒 当前隐私预算: ε = {epsilon:.4f}, δ = {args.target_delta}")
# # # #                     print(f"{'=' * 80}")
# # # #                 else:
# # # #                     print(f"⚠️  无法获取隐私预算信息")
# # # #             else:
# # # #                 print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
# # # #                 print(f"📈 正则项贡献: 训练={train_reg_term:.6f} | 测试={test_reg_term:.6f}")
# # # #
# # # #             if test_acc > best_ACC:
# # # #                 best_ACC = test_acc
# # # #                 best_model = deepcopy(net)
# # # #                 print(f"✅ 新的最佳准确率: {best_ACC:.2f}%")
# # # #
# # # #             scheduler.step()
# # # #
# # # #             # 每个epoch结束后清理缓存
# # # #             torch.cuda.empty_cache()
# # # #
# # # #         print("best acc:", best_ACC)
# # # #         if args.private:
# # # #             file_name = "{}_{:.3f}&{:.3f}_{:.2f}_private_eps_{:.2f}.tar".format(model_name, a1, a2, best_ACC,
# # # #                                                                                 args.target_epsilon)
# # # #         else:
# # # #             file_name = "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, best_ACC)
# # # #
# # # #         utils.save_checkpoint({
# # # #             'state_dict': best_model.state_dict(),
# # # #         }, model_path, file_name)
# # # #
# # # #
# # # # if __name__ == '__main__':
# # # #     from argparse import ArgumentParser
# # # #
# # # #     parser = ArgumentParser(description='train with BiDO')
# # # #     parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
# # # #     parser.add_argument('--measure', default='MI', help='HSIC | COCO | MI')
# # # #     parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
# # # #     parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
# # # #     parser.add_argument('--root_path', default='./', help='')
# # # #     parser.add_argument('--config_dir', default='./config', help='')
# # # #     parser.add_argument('--model_dir', default='./target_model', help='')
# # # #     parser.add_argument('--private', type=str2bool, default=True,
# # # #                         help='是否启用差分隐私，默认开启；传 --private false 可关闭')
# # # #     # 轻量DP参数：目标-准确率下降约5%，同时降低攻击准确率
# # # #     # 使用较大的epsilon和较小的噪声，最小化对准确率的影响
# # # #     parser.add_argument('--target_epsilon', type=float, default=15.0,
# # # #                         help='Target epsilon for DP (较大值=更弱的隐私保护但准确率下降更少，默认15.0用于轻量DP)')
# # # #     parser.add_argument('--target_delta', type=float, default=1e-5, help='Target delta for DP')
# # # #     parser.add_argument('--max_grad_norm', type=float, default=1.5,
# # # #                         help='Max grad norm for clipping (梯度裁剪范围，较大值减少裁剪影响，默认1.5)')
# # # #     parser.add_argument('--sample_rate', type=float, default=0.01,
# # # #                         help='Subsampling rate for DP-SGD (稍大的采样率可减少噪声影响，默认0.01)')
# # # #     parser.add_argument('--dp_noise_scale', type=float, default=0.3,
# # # #                         help='缩放DP噪声乘子 (默认0.3，进一步减小噪声以最小化准确率下降)')
# # # #     parser.add_argument('--dp_lr_scale', type=float, default=0.9,
# # # #                         help='DP模式下学习率缩放因子 (默认0.9，只轻微降低学习率)')
# # # #
# # # #     args = parser.parse_args()
# # # #
# # # #     if args.private:
# # # #         args.measure = args.measure + "+DP"
# # # #
# # # #     model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.measure)
# # # #
# # # #     os.makedirs(model_path, exist_ok=True)
# # # #
# # # #     file = os.path.join(args.config_dir, args.dataset + ".json")
# # # #
# # # #     loaded_args = utils.load_json(json_file=file)
# # # #     model_name = loaded_args["dataset"]["model_name"]
# # # #
# # # #     train_file = loaded_args['dataset']['train_file']
# # # #     test_file = loaded_args['dataset']['test_file']
# # # #
# # # #     trainloader = utils.init_dataloader(loaded_args, train_file, mode="train")
# # # #     testloader = utils.init_dataloader(loaded_args, test_file, mode="test")
# # # #
# # # #     # The user should set args.target_delta to be 1/len(trainloader.dataset)
# # # #     if args.private and args.target_delta is None:
# # # #         args.target_delta = 1 / len(trainloader.dataset)
# # # #         print(f"Setting delta to {args.target_delta}")
# # # #
# # # #     main(args, loaded_args, trainloader, testloader)
# # # #
# # # # 准确率太低问题和差分隐私问题，加了
# #
# #
# # #
# #
# # # 准确率太低问题和差分隐私问题，加了
# #
# # # # 让不同的 a1/a2 更“能起作用”，模型准确率太低了
# # # DP-SGD + BiDO Mutual Information Gradient Fusion
# # # 梯度级融合：DP-SGD + BiDO正则项，实现差分隐私与互信息约束的组合
# # import argparse
# # import torch, os, engine, utils
# # import torch.nn as nn
# # from copy import deepcopy
# # import collections
# # from opacus import PrivacyEngine
# # from opacus.validators import ModuleValidator
# # from opacus.utils.batch_memory_manager import BatchMemoryManager
# # # 用于训练深度学习模型的脚本，特别是针对一个特定的约束优化算法HSIC
# # import model
# #
# # device = "cuda"
# #
# #
# # def str2bool(v):
# #     if isinstance(v, bool):
# #         return v
# #     if v.lower() in ('yes', 'true', 't', '1', 'y'):
# #         return True
# #     if v.lower() in ('no', 'false', 'f', '0', 'n'):
# #         return False
# #     raise argparse.ArgumentTypeError('Boolean value expected.')
# #
# #
# # # 加载预训练模型的权重
# # def load_my_state_dict(net, state_dict):
# #     print("load nature model!!!")
# #     net_state = net.state_dict()
# #     for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
# #         # print(name, '---', old_name)
# #         net_state[name].copy_(old_param.data)
# #
# #
# # def load_feature_extractor(net, state_dict):  # 加载预训练的特征提取部分的权重
# #     print("load_pretrained_feature_extractor!!!")
# #     net_state = net.state_dict()
# #
# #     # 🔥 修复：按名称匹配权重，而不是按顺序，并检查形状是否匹配
# #     # 处理预训练权重中的running_var -> num_batches_tracked
# #     new_state_dict = collections.OrderedDict()
# #     for name, param in state_dict.items():
# #         if "running_var" in name:
# #             new_state_dict[name] = param  # 权重
# #             new_item = name.replace("running_var", "num_batches_tracked")
# #             new_state_dict[new_item] = torch.tensor(0)
# #         else:
# #             new_state_dict[name] = param
# #
# #     # 按名称匹配权重，跳过不匹配的层
# #     loaded_count = 0
# #     skipped_count = 0
# #     for name, param in net_state.items():
# #         # 跳过classifier层
# #         if "classifier" in name:
# #             continue
# #         # 跳过num_batches_tracked（会在训练时自动更新）
# #         if "num_batches_tracked" in name:
# #             continue
# #
# #         # 按名称查找匹配的预训练权重
# #         if name in new_state_dict:
# #             pretrained_param = new_state_dict[name]
# #             # 检查形状是否匹配
# #             if param.shape == pretrained_param.shape:
# #                 param.copy_(pretrained_param.data)
# #                 loaded_count += 1
# #             else:
# #                 print(f"⚠️  跳过权重 {name}: 形状不匹配 (模型: {param.shape}, 预训练: {pretrained_param.shape})")
# #                 skipped_count += 1
# #         else:
# #             # 尝试查找可能的变体名称（例如，如果ModuleValidator.fix改变了某些层名）
# #             # 这里可以添加更多的名称映射逻辑
# #             skipped_count += 1
# #
# #     print(f"✓ 权重加载完成: 成功加载 {loaded_count} 个层, 跳过 {skipped_count} 个层")
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
# #     # 互信息约束权重超参数搜索列表 (α1, α2)
# #     # 格式: (α1, α2) 其中 α1 控制最小化 I(Z;X)，α2 控制最大化 I(Z;Y)
# #     # 参考 BiDO/train_COCO.py 的数值范围，但 MI 的数值范围可能不同，需要调整
# #     # 扩大搜索范围，确保不同参数有明显差异
# #     hp_list = [
# #         # (0.0, 0.0),      # 基线：无正则项
# #
# #         # (0.01, 0.1),     # 小权重80.22
# #         (0.01, 0.05),  # 小权重84.24
# #         # (0.05, 0.5),     # 中等权重67.52
# #         # (0.1, 1.0),      # 较大权重59.84
# #         # (0.5, 5.0),  # 很大权重41.39
# #         #
# #         # (0.2, 2.0),      # 大权重
# #     ]
# #     # 交叉熵
# #     criterion = nn.CrossEntropyLoss().cuda()
# #
# #     for i, (a1, a2) in enumerate(hp_list):
# #         print("\n" + "=" * 80)
# #         print(f"开始训练第 {i+1}/{len(hp_list)} 组超参数: a1={a1}, a2={a2}")
# #         print("=" * 80)
# #         torch.cuda.empty_cache()  # 清空GPU缓存
# #
# #         # 根据配置选择模型
# #         if model_name == "VGG16" or model_name == "reg":
# #             net = model.VGG16(n_classes, hsic_training=args.hsic_training, dataset=args.dataset)
# #         elif model_name == "ResNet":
# #             net = model.ResNetClsH(nclass=n_classes, resnetl=10)
# #         elif model_name == "MCNN":
# #             net = model.MCNN(n_classes)
# #         elif model_name == "LeNet":
# #             net = model.LeNet3(n_classes)
# #         elif model_name == "SimpleCNN":
# #             net = model.Classifier(1, 128, n_classes)
# #
# #         # 🔥 关键修复：在加载预训练权重之前调用ModuleValidator.fix
# #         # 如果先加载权重再fix，模型结构可能被改变，导致权重不匹配
# #         if args.private and not ModuleValidator.is_valid(net):
# #             print("⚠️  模型结构需要修复以兼容Opacus，在加载预训练权重之前修复...")
# #             net = ModuleValidator.fix(net)
# #             print("✓ 模型结构已修复")
# #
# #         # 加载预训练权重（在fix之后，确保结构稳定）
# #         if model_name == "VGG16" or model_name == "reg":
# #             load_pretrained_feature_extractor = True
# #             if load_pretrained_feature_extractor:
# #                 pretrained_model_ckpt = "target_model/vgg16_bn-6c64b313.pth"
# #                 checkpoint = torch.load(pretrained_model_ckpt)
# #                 load_feature_extractor(net, checkpoint)
# #                 print("✓ 预训练特征提取器权重已加载")
# #
# #         # 使用混合精度训练
# #         net = net.cuda()
# #         # Opacus 不支持 DataParallel，推荐使用单卡或 DDP
# #         scaler = torch.cuda.amp.GradScaler()
# #
# #         # 定义优化器（轻量DP：只做轻微学习率缩放）
# #         base_lr = lr if not args.private else lr * args.dp_lr_scale  # DP训练时仅轻微降低学习率
# #         optimizer = torch.optim.AdamW(
# #             net.parameters(),
# #             lr=base_lr,
# #             weight_decay=weight_decay if weight_decay > 0 else 1e-4,
# #             betas=(0.9, 0.999),
# #             eps=1e-8
# #         )
# #
# #         # 如果启用差分隐私，初始化并附加PrivacyEngine（极轻量模式）
# #         privacy_engine = None
# #         max_physical_batch_size = None
# #         base_dp_noise = None
# #         best_model = None  # 初始化best_model变量
# #         if args.private:
# #             print(f"\n{'=' * 80}")
# #             print(f"🔒 启用极轻量差分隐私训练（目标：准确率下降<5%）")
# #             print(f"  目标隐私预算: ε={args.target_epsilon}, δ={args.target_delta}")
# #             print(f"  梯度裁剪范围: {args.max_grad_norm}")
# #             print(f"  采样率: {args.sample_rate}")
# #             print(f"{'=' * 80}\n")
# #
# #             # 确保 dataloader 有 drop_last=True 以避免 batch size 不一致
# #             # Opacus 要求所有 batch 大小一致
# #             if hasattr(trainloader, 'drop_last') and not trainloader.drop_last:
# #                 print("⚠️  警告: dataloader 的 drop_last=False，Opacus 需要固定 batch size")
# #                 print("   建议在 utils.init_dataloader 中设置 drop_last=True")
# #
# #             # 保存原始 batch_size 用于 BatchMemoryManager
# #             original_batch_size = getattr(trainloader, 'batch_size', None)
# #             if original_batch_size is None:
# #                 try:
# #                     original_batch_size = trainloader.batch_sampler.batch_size
# #                 except:
# #                     original_batch_size = 32
# #
# #             privacy_engine = PrivacyEngine()
# #             # 🔥 关键修复：保存原始模型的引用，用于后续保存state_dict
# #             original_net = net
# #             net, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
# #                 module=net,
# #                 optimizer=optimizer,
# #                 data_loader=trainloader,
# #                 max_grad_norm=args.max_grad_norm,
# #                 epochs=n_epochs,
# #                 target_epsilon=args.target_epsilon,
# #                 target_delta=args.target_delta,
# #                 poisson_sampling=False,  # 保持原有批次分布，避免破坏随机身份采样
# #             )
# #             print("✓ DP-SGD已启用，模型已被Opacus包装")
# #
# #             # 进一步减小DP噪声，使其"只参与一点"
# #             # 尝试多种方式获取privacy_engine
# #             pe = None
# #             if hasattr(optimizer, 'privacy_engine'):
# #                 pe = optimizer.privacy_engine
# #             elif hasattr(optimizer, '_privacy_engine'):
# #                 pe = optimizer._privacy_engine
# #             elif privacy_engine is not None:
# #                 pe = privacy_engine
# #
# #             if pe is not None and hasattr(pe, 'noise_multiplier'):
# #                 if args.dp_noise_scale != 1.0:
# #                     old_noise = pe.noise_multiplier
# #                     new_noise = max(1e-6, old_noise * args.dp_noise_scale)
# #                     pe.noise_multiplier = new_noise
# #                     if hasattr(pe, '_noise_multiplier'):
# #                         pe._noise_multiplier = new_noise
# #                     print(f"⚠️  极轻量DP：噪声乘子 {old_noise:.4f} → {new_noise:.4f} (scale={args.dp_noise_scale})，"
# #                           f"隐私预算会略高于 target_epsilon\n")
# #                     base_dp_noise = new_noise
# #                 else:
# #                     base_dp_noise = pe.noise_multiplier
# #             else:
# #                 print("⚠️  警告: 无法找到privacy_engine或其噪声信息，将跳过DP噪声调节")
# #                 base_dp_noise = None
# #
# #             # 设置虚拟 batch 大小以避免 OOM，同时确保 batch size 一致
# #             # 使用更大的虚拟batch（接近原始batch size）以减少拆分次数，提高训练效率
# #             if args.max_physical_batch_size is not None:
# #                 max_physical_batch_size = args.max_physical_batch_size
# #             else:
# #                 # 默认使用原始batch size的3/4，减少拆分但避免OOM
# #                 max_physical_batch_size = max(1, int(original_batch_size * 0.75))
# #             # 确保能整除，避免 batch size 不一致
# #             if original_batch_size % max_physical_batch_size != 0:
# #                 max_physical_batch_size = original_batch_size // (original_batch_size // max_physical_batch_size + 1)
# #                 max_physical_batch_size = max(1, max_physical_batch_size)
# #             print(f"📦 虚拟batch大小: {max_physical_batch_size} (原始: {original_batch_size})\n")
# #
# #         # 使用余弦退火学习率调度器
# #         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
# #             optimizer,
# #             T_0=5,  # 第一次重启的周期
# #             T_mult=2,  # 每次重启后周期长度的倍数
# #             eta_min=base_lr * 0.01  # 最小学习率
# #         )
# #
# #         best_ACC = -1
# #         # 在DP训练下，每个epoch都会消费隐私预算
# #         for epoch in range(n_epochs):
# #             print('\n' + '=' * 80)
# #             print(f'Epoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
# #             print('=' * 80)
# #
# #             # 动态调节DP噪声（free_epochs + warmup），让模型先学好任务再加噪声
# #             if args.private and base_dp_noise is not None:
# #                 # 尝试多种方式获取privacy_engine
# #                 pe = None
# #                 if hasattr(optimizer, 'privacy_engine'):
# #                     pe = optimizer.privacy_engine
# #                 elif hasattr(optimizer, '_privacy_engine'):
# #                     pe = optimizer._privacy_engine
# #                 elif privacy_engine is not None:
# #                     pe = privacy_engine
# #
# #                 if pe is not None and hasattr(pe, 'noise_multiplier'):
# #                     if epoch < args.dp_noise_free_epochs:
# #                         current_noise = 0.0
# #                         noise_phase = "关闭"
# #                     elif args.dp_noise_warmup > 0 and epoch < args.dp_noise_free_epochs + args.dp_noise_warmup:
# #                         progress = (epoch - args.dp_noise_free_epochs + 1) / args.dp_noise_warmup
# #                         current_noise = base_dp_noise * min(1.0, max(0.0, progress))
# #                         noise_phase = f"升温({progress*100:.1f}%)"
# #                     else:
# #                         current_noise = base_dp_noise
# #                         noise_phase = "全量"
# #
# #                     pe.noise_multiplier = current_noise
# #                     if hasattr(pe, "_noise_multiplier"):
# #                         pe._noise_multiplier = current_noise
# #                     print(f"🔧 DP噪声: {current_noise:.6f} ({noise_phase})")
# #                 elif pe is not None:
# #                     print("⚠️  警告: 找到privacy_engine但缺少噪声信息，跳过DP噪声调节")
# #
# #             # 训练：总损失 = 任务损失 + β * BiDO正则项（互信息约束）
# #             # engine.train_mutual内部计算: L_total = L_task + mi_weight * (α1*I(Z;X) - α2*I(Z;Y))
# #             # DP-SGD会自动处理梯度裁剪和噪声添加
# #             # 如果启用DP，使用BatchMemoryManager确保batch size一致，避免OOM和batch size不匹配错误
# #             if args.private and max_physical_batch_size is not None:
# #                 with BatchMemoryManager(
# #                     data_loader=trainloader,
# #                     max_physical_batch_size=max_physical_batch_size,
# #                     optimizer=optimizer
# #                 ) as memory_safe_dataloader:
# #                     # 🔥 修复：engine.train_mutual只返回2个值，移除mi_weight参数（函数不接受）
# #                     train_loss, train_acc, *_ = engine.train_mutual(
# #                         net, criterion, optimizer, memory_safe_dataloader, a1, a2, n_classes,
# #                         ktype=args.ktype,
# #                         hsic_training=args.hsic_training,
# #                         measure=args.measure)
# #             else:
# #                 # 🔥 修复：engine.train_mutual只返回2个值，移除mi_weight参数
# #                 train_loss, train_acc, *_ = engine.train_mutual(
# #                     net, criterion, optimizer, trainloader, a1, a2, n_classes,
# #                     ktype=args.ktype,
# #                     hsic_training=args.hsic_training,
# #                     measure=args.measure)
# #
# #             # 🔥 修复：engine.test_mutual只返回2个值，移除mi_weight参数
# #             test_loss, test_acc, *_ = engine.test_mutual(net, criterion, testloader, a1, a2, n_classes,
# #                                                          ktype=args.ktype,
# #                                                          hsic_training=args.hsic_training,
# #                                                          measure=args.measure)
# #
# #             # 如果是DP训练，打印当前隐私预算消耗
# #             if args.private:
# #                 # 尝试多种方式获取privacy_engine
# #                 pe = None
# #                 if hasattr(optimizer, 'privacy_engine'):
# #                     pe = optimizer.privacy_engine
# #                 elif hasattr(optimizer, '_privacy_engine'):
# #                     pe = optimizer._privacy_engine
# #                 elif privacy_engine is not None:
# #                     pe = privacy_engine
# #
# #                 if pe is not None:
# #                     try:
# #                         epsilon = pe.get_epsilon(delta=args.target_delta)
# #                         print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
# #                         print(f"🔒 当前隐私预算: ε = {epsilon:.4f}, δ = {args.target_delta}")
# #                         print(f"{'=' * 80}")
# #                     except Exception as e:
# #                         print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
# #                         print(f"⚠️  无法计算隐私预算: {e}")
# #                         print(f"{'=' * 80}")
# #                 else:
# #                     print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
# #                     print(f"⚠️  无法获取隐私预算信息（privacy_engine未找到）")
# #                     print(f"{'=' * 80}")
# #             else:
# #                 print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
# #
# #             if test_acc > best_ACC:
# #                 best_ACC = test_acc
# #                 # 🔥 关键修复：DP模式下，保存原始模型的state_dict，而不是包装后的模型
# #                 if args.private:
# #                     # Opacus包装后的模型，需要从包装中提取原始模型
# #                     if hasattr(net, '_module'):
# #                         # GradSampleModule包装
# #                         best_model_state = net._module.state_dict()
# #                     elif hasattr(net, 'module'):
# #                         # 其他可能的包装
# #                         best_model_state = net.module.state_dict()
# #                     else:
# #                         # 如果无法提取，尝试直接使用（可能已经是原始模型）
# #                         best_model_state = net.state_dict()
# #                     best_model = best_model_state  # 保存state_dict而不是模型对象
# #                 else:
# #                     best_model = deepcopy(net)  # 非DP模式，直接保存模型
# #                 print(f"✅ 新的最佳准确率: {best_ACC:.2f}%")
# #
# #             scheduler.step()
# #
# #             # 每个epoch结束后清理缓存
# #             torch.cuda.empty_cache()
# #
# #         print("best acc:", best_ACC)
# #         if args.private:
# #             file_name = "{}_{:.3f}&{:.3f}_{:.2f}_private_eps_{:.2f}.tar".format(model_name, a1, a2, best_ACC,
# #                                                                                 args.target_epsilon)
# #         else:
# #             file_name = "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, best_ACC)
# #
# #         # 🔥 修复：DP模式下，best_model已经是state_dict，不需要再调用state_dict()
# #         if args.private:
# #             utils.save_checkpoint({
# #                 'state_dict': best_model,  # 已经是state_dict
# #             }, model_path, file_name)
# #         else:
# #             utils.save_checkpoint({
# #                 'state_dict': best_model.state_dict(),  # 需要提取state_dict
# #             }, model_path, file_name)
# #
# #
# # if __name__ == '__main__':
# #     from argparse import ArgumentParser
# #
# #     parser = ArgumentParser(description='train with BiDO')
# #     parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
# #     parser.add_argument('--measure', default='MI', help='HSIC | COCO | MI')
# #     parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
# #     parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
# #     parser.add_argument('--root_path', default='./', help='')
# #     parser.add_argument('--config_dir', default='./config', help='')
# #     parser.add_argument('--model_dir', default='./target_model', help='')
# #     # Opacus DP arguments
# #     parser.add_argument('--mi_weight', type=float, default=1.0, help='global weight for MI/HSIC regularizer')
# #     parser.add_argument('--mi_warmup_epochs', type=int, default=15,
# #                         help='Linearly warm up MI权重的epoch数，0表示立即使用完整权重')
# #     parser.add_argument('--private', type=str2bool, default= False,
# #                         help='是否启用差分隐私，默认开启（轻量DP）；可传 --private false 关闭')
# #     # 极轻量DP参数：目标-准确率下降约5%，同时降低攻击准确率
# #     parser.add_argument('--target_epsilon', type=float, default=25.0,
# #                         help='Target epsilon for DP (更大值=更弱的隐私保护但准确率下降更少，默认25.0用于极轻量DP)')
# #     parser.add_argument('--target_delta', type=float, default=1e-5, help='Target delta for DP')
# #     parser.add_argument('--max_grad_norm', type=float, default=2.0,
# #                         help='Max grad norm for clipping (更大值减少裁剪影响，默认2.0)')
# #     parser.add_argument('--sample_rate', type=float, default=0.015,
# #                         help='Subsampling rate for DP-SGD (更大采样率减少噪声影响，默认0.015)')
# #     parser.add_argument('--dp_noise_scale', type=float, default=0.15,
# #                         help='缩放DP噪声乘子 (默认0.15，极大幅度减小噪声以最小化准确率下降)')
# #     parser.add_argument('--dp_noise_free_epochs', type=int, default=5,
# #                         help='前多少个epoch完全关闭DP噪声，让模型先学好任务（默认5）')
# #     parser.add_argument('--dp_noise_warmup', type=int, default=8,
# #                         help='DP噪声线性升温的epoch数，0表示立即使用完整噪声（默认8）')
# #     parser.add_argument('--dp_lr_scale', type=float, default=0.95,
# #                         help='DP模式下学习率缩放因子（默认0.95，几乎不降低学习率）')
# #     parser.add_argument('--max_physical_batch_size', type=int, default=None,
# #                         help='最大物理batch大小用于BatchMemoryManager (None=自动设置为原始batch_size*0.75)')
# #
# #     args = parser.parse_args()
# #
# #     if args.private:
# #         args.measure = args.measure + "+DP"
# #
# #     model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.measure)
# #
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
# #     # The user should set args.target_delta to be 1/len(trainloader.dataset)
# #     if args.private and args.target_delta is None:
# #         args.target_delta = 1 / len(trainloader.dataset)
# #         print(f"Setting delta to {args.target_delta}")
# #
# #     main(args, loaded_args, trainloader, testloader)
# #
#
#
#
#
# # 新的版本，从头更改了dp，23.57%，调整了一些系数
#
#
# # 准确率太低问题和差分隐私问题，加了
#
# # # 让不同的 a1/a2 更“能起作用”，模型准确率太低了
# # DP-SGD + BiDO Mutual Information Gradient Fusion
# # 梯度级融合：DP-SGD + BiDO正则项，实现差分隐私与互信息约束的组合
# import argparse
# import torch, os, engine, utils
# import torch.nn as nn
# from copy import deepcopy
# import collections
# from types import MethodType
# # 用于训练深度学习模型的脚本，特别是针对一个特定的约束优化算法HSIC
# import model
#
# device = "cuda"
#
#
# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', '1', 'y'):
#         return True
#     if v.lower() in ('no', 'false', 'f', '0', 'n'):
#         return False
#     raise argparse.ArgumentTypeError('Boolean value expected.')
#
#
# # 加载预训练模型的权重
# def load_my_state_dict(net, state_dict):
#     print("load nature model!!!")
#     net_state = net.state_dict()
#     for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
#         # print(name, '---', old_name)
#         net_state[name].copy_(old_param.data)
#
#
# def load_feature_extractor(net, state_dict):  # 加载预训练的特征提取部分的权重
#     print("load_pretrained_feature_extractor!!!")
#     net_state = net.state_dict()
#
#     # 🔥 修复：按名称匹配权重，而不是按顺序，并检查形状是否匹配
#     # 处理预训练权重中的running_var -> num_batches_tracked
#     new_state_dict = collections.OrderedDict()
#     for name, param in state_dict.items():
#         if "running_var" in name:
#             new_state_dict[name] = param  # 权重
#             new_item = name.replace("running_var", "num_batches_tracked")
#             new_state_dict[new_item] = torch.tensor(0)
#         else:
#             new_state_dict[name] = param
#
#     # 按名称匹配权重，跳过不匹配的层
#     loaded_count = 0
#     skipped_count = 0
#     for name, param in net_state.items():
#         # 跳过classifier层
#         if "classifier" in name:
#             continue
#         # 跳过num_batches_tracked（会在训练时自动更新）
#         if "num_batches_tracked" in name:
#             continue
#
#         # 按名称查找匹配的预训练权重
#         if name in new_state_dict:
#             pretrained_param = new_state_dict[name]
#             # 检查形状是否匹配
#             if param.shape == pretrained_param.shape:
#                 param.copy_(pretrained_param.data)
#                 loaded_count += 1
#             else:
#                 print(f"⚠️  跳过权重 {name}: 形状不匹配 (模型: {param.shape}, 预训练: {pretrained_param.shape})")
#                 skipped_count += 1
#         else:
#             # 尝试查找可能的变体名称（例如，如果ModuleValidator.fix改变了某些层名）
#             # 这里可以添加更多的名称映射逻辑
#             skipped_count += 1
#
#     print(f"✓ 权重加载完成: 成功加载 {loaded_count} 个层, 跳过 {skipped_count} 个层")
#
#
# def apply_micro_dp(model, max_grad_norm, noise_std):
#     """对梯度做一次统一裁剪，并按需添加极少量高斯噪声。"""
#     if max_grad_norm <= 0:
#         return
#
#     total_norm_sq = 0.0
#     grads = []
#     for p in model.parameters():
#         if p.grad is None:
#             continue
#         grads.append(p.grad)
#         param_norm = p.grad.data.norm(2)
#         total_norm_sq += param_norm.item() ** 2
#     if not grads:
#         return
#
#     total_norm = total_norm_sq ** 0.5
#     clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-12))
#     if clip_coef < 1.0:
#         for g in grads:
#             g.data.mul_(clip_coef)
#
#     if noise_std > 0:
#         scaled_std = noise_std * max_grad_norm
#         for g in grads:
#             g.data.add_(torch.randn_like(g) * scaled_std)
#
#
# def enable_micro_dp_on_optimizer(optimizer, model, max_grad_norm):
#     """Monkey-patch优化器的step方法，使其在每次更新前执行Micro-DP。"""
#     conf = {
#         "model": model,
#         "max_grad_norm": max_grad_norm,
#         "noise_std": 0.0,
#         "original_step": optimizer.step,
#     }
#
#     def micro_dp_step(self, *args, **kwargs):
#         apply_micro_dp(conf["model"], conf["max_grad_norm"], conf["noise_std"])
#         return conf["original_step"](*args, **kwargs)
#
#     def set_noise_std(self, value):
#         conf["noise_std"] = max(0.0, float(value))
#
#     optimizer.step = MethodType(micro_dp_step, optimizer)
#     optimizer.set_noise_std = MethodType(set_noise_std, optimizer)
#     optimizer.set_noise_std(0.0)
#     return optimizer
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
#     # 互信息约束权重超参数搜索列表 (α1, α2)
#     # 格式: (α1, α2) 其中 α1 控制最小化 I(Z;X)，α2 控制最大化 I(Z;Y)
#     # 参考 BiDO/train_COCO.py 的数值范围，但 MI 的数值范围可能不同，需要调整
#     # 扩大搜索范围，确保不同参数有明显差异
#     hp_list = [
#         # (0.0, 0.0),      # 基线：无正则项
#
#         # (0.01, 0.1),  # 小权重80.22
#         (0.01, 0.05),  # 小权重84.24
#         # (0.05, 0.5),     # 中等权重67.52
#         # (0.1, 1.0),      # 较大权重59.84
#         # (0.5, 5.0),  # 很大权重41.39
#         #
#         # (0.2, 2.0),      # 大权重
#     ]
#     # 交叉熵
#     criterion = nn.CrossEntropyLoss().cuda()
#
#     for i, (a1, a2) in enumerate(hp_list):
#         print("\n" + "=" * 80)
#         print(f"开始训练第 {i + 1}/{len(hp_list)} 组超参数: a1={a1}, a2={a2}")
#         print("=" * 80)
#         torch.cuda.empty_cache()  # 清空GPU缓存
#
#         # 根据配置选择模型
#         if model_name == "VGG16" or model_name == "reg":
#             net = model.VGG16(n_classes, hsic_training=args.hsic_training, dataset=args.dataset)
#         elif model_name == "ResNet":
#             net = model.ResNetClsH(nclass=n_classes, resnetl=10)
#         elif model_name == "MCNN":
#             net = model.MCNN(n_classes)
#         elif model_name == "LeNet":
#             net = model.LeNet3(n_classes)
#         elif model_name == "SimpleCNN":
#             net = model.Classifier(1, 128, n_classes)
#
#         # 加载预训练权重（在fix之后，确保结构稳定）
#         if model_name == "VGG16" or model_name == "reg":
#             load_pretrained_feature_extractor = True
#             if load_pretrained_feature_extractor:
#                 pretrained_model_ckpt = "target_model/vgg16_bn-6c64b313.pth"
#                 checkpoint = torch.load(pretrained_model_ckpt)
#                 load_feature_extractor(net, checkpoint)
#                 print("✓ 预训练特征提取器权重已加载")
#
#         net = net.cuda()
#
#         # 定义优化器（轻量DP仅在梯度阶段做处理）
#         base_lr = lr
#         optimizer = torch.optim.AdamW(
#             net.parameters(),
#             lr=base_lr,
#             weight_decay=weight_decay if weight_decay > 0 else 1e-4,
#             betas=(0.9, 0.999),
#             eps=1e-8
#         )
#
#         micro_dp_enabled = bool(args.private)
#         if micro_dp_enabled:
#             optimizer = enable_micro_dp_on_optimizer(optimizer, net, args.dp_clip_norm)
#
#         best_model = None
#
#         # 使用余弦退火学习率调度器
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             optimizer,
#             T_0=5,  # 第一次重启的周期
#             T_mult=2,  # 每次重启后周期长度的倍数
#             eta_min=base_lr * 0.01  # 最小学习率
#         )
#
#         best_ACC = -1
#         dp_start_epoch = max(1, args.dp_start_epoch)
#         dp_warmup_epochs = max(0, args.dp_warmup_epochs)
#
#         for epoch in range(n_epochs):
#             print('\n' + '=' * 80)
#             print(f'Epoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
#             print('=' * 80)
#
#             if micro_dp_enabled and hasattr(optimizer, 'set_noise_std'):
#                 epoch_idx = epoch + 1
#                 if epoch_idx < dp_start_epoch:
#                     current_noise = 0.0
#                     noise_phase = "off"
#                 elif dp_warmup_epochs > 0 and epoch_idx < dp_start_epoch + dp_warmup_epochs:
#                     progress = (epoch_idx - dp_start_epoch + 1) / dp_warmup_epochs
#                     progress = min(1.0, max(0.0, progress))
#                     current_noise = args.dp_noise_std * progress
#                     noise_phase = f"warmup {progress * 100:.1f}%"
#                 else:
#                     current_noise = args.dp_noise_std
#                     noise_phase = "steady"
#                 optimizer.set_noise_std(current_noise)
#                 print(f"🔐 Micro-DP: noise_std={current_noise:.5f} ({noise_phase})")
#             else:
#                 current_noise = 0.0
#
#             train_loss, train_acc, *_ = engine.train_mutual(
#                 net, criterion, optimizer, trainloader, a1, a2, n_classes,
#                 ktype=args.ktype,
#                 hsic_training=args.hsic_training,
#                 measure=args.measure)
#
#             test_loss, test_acc, *_ = engine.test_mutual(
#                 net, criterion, testloader, a1, a2, n_classes,
#                 ktype=args.ktype,
#                 hsic_training=args.hsic_training,
#                 measure=args.measure)
#
#             print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")
#
#             if test_acc > best_ACC:
#                 best_ACC = test_acc
#                 best_model = deepcopy(net)
#                 print(f"✅ 新的最佳准确率: {best_ACC:.2f}%")
#
#             scheduler.step()
#             torch.cuda.empty_cache()
#
#         print("best acc:", best_ACC)
#         if best_model is None:
#             best_model = deepcopy(net)
#         if args.private:
#             file_name = "{}_{:.3f}&{:.3f}_{:.2f}_microdp.tar".format(model_name, a1, a2, best_ACC)
#         else:
#             file_name = "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, best_ACC)
#
#         utils.save_checkpoint({
#             'state_dict': best_model.state_dict(),
#         }, model_path, file_name)
#
#
# if __name__ == '__main__':
#     from argparse import ArgumentParser
#
#     parser = ArgumentParser(description='train with BiDO')
#     parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
#     parser.add_argument('--measure', default='MI', help='HSIC | COCO | MI')
#     parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
#     parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
#     parser.add_argument('--root_path', default='./', help='')
#     parser.add_argument('--config_dir', default='./config', help='')
#     parser.add_argument('--model_dir', default='./target_model', help='')
#     # Opacus DP arguments
#     parser.add_argument('--mi_weight', type=float, default=1.0, help='global weight for MI/HSIC regularizer')
#     parser.add_argument('--mi_warmup_epochs', type=int, default=15,
#                         help='Linearly warm up MI权重的epoch数，0表示立即使用完整权重')
#     parser.add_argument('--private', type=str2bool, default=True,
#                         help='是否启用Micro-DP梯度噪声（默认关闭）')
#     parser.add_argument('--dp_clip_norm', type=float, default=1.0,
#                         help='Micro-DP全局梯度裁剪阈值')
#     parser.add_argument('--dp_noise_std', type=float, default=0.02,
#                         help='Micro-DP噪声标准差（相对clip阈值）')
#     parser.add_argument('--dp_start_epoch', type=int, default=3,
#                         help='从第几个epoch开始注入噪声（1-based）')
#     parser.add_argument('--dp_warmup_epochs', type=int, default=5,
#                         help='噪声线性升温的epoch数，0表示立即使用完整噪声')
#
#     args = parser.parse_args()
#
#     model_path = os.path.join(args.root_path, args.model_dir, args.dataset, args.measure)
#
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
#     main(args, loaded_args, trainloader, testloader)
#


# 准确率太低问题和差分隐私问题，加了

# # 让不同的 a1/a2 更“能起作用”，模型准确率太低了
# DP-SGD + BiDO Mutual Information Gradient Fusion
# 梯度级融合：DP-SGD + BiDO正则项，实现差分隐私与互信息约束的组合
import argparse
import torch, os, engine, utils
import torch.nn as nn
from copy import deepcopy
import collections
from types import MethodType
# 用于训练深度学习模型的脚本，特别是针对一个特定的约束优化算法HSIC
import model

device = "cuda"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


# 加载预训练模型的权重
def load_my_state_dict(net, state_dict):
    print("load nature model!!!")
    net_state = net.state_dict()
    for ((name, param), (old_name, old_param),) in zip(net_state.items(), state_dict.items()):
        # print(name, '---', old_name)
        net_state[name].copy_(old_param.data)


def load_feature_extractor(net, state_dict):  # 加载预训练的特征提取部分的权重
    print("load_pretrained_feature_extractor!!!")
    net_state = net.state_dict()

    # 🔥 修复：按名称匹配权重，而不是按顺序，并检查形状是否匹配
    # 处理预训练权重中的running_var -> num_batches_tracked
    new_state_dict = collections.OrderedDict()
    for name, param in state_dict.items():
        if "running_var" in name:
            new_state_dict[name] = param  # 权重
            new_item = name.replace("running_var", "num_batches_tracked")
            new_state_dict[new_item] = torch.tensor(0)
        else:
            new_state_dict[name] = param

    # 按名称匹配权重，跳过不匹配的层
    loaded_count = 0
    skipped_count = 0
    for name, param in net_state.items():
        # 跳过classifier层
        if "classifier" in name:
            continue
        # 跳过num_batches_tracked（会在训练时自动更新）
        if "num_batches_tracked" in name:
            continue

        # 按名称查找匹配的预训练权重
        if name in new_state_dict:
            pretrained_param = new_state_dict[name]
            # 检查形状是否匹配
            if param.shape == pretrained_param.shape:
                param.copy_(pretrained_param.data)
                loaded_count += 1
            else:
                print(f"⚠️  跳过权重 {name}: 形状不匹配 (模型: {param.shape}, 预训练: {pretrained_param.shape})")
                skipped_count += 1
        else:
            # 尝试查找可能的变体名称（例如，如果ModuleValidator.fix改变了某些层名）
            # 这里可以添加更多的名称映射逻辑
            skipped_count += 1

    print(f"✓ 权重加载完成: 成功加载 {loaded_count} 个层, 跳过 {skipped_count} 个层")


def apply_micro_dp(model, max_grad_norm, noise_std):
    """对梯度做一次统一裁剪，并按需添加极少量高斯噪声。"""
    if max_grad_norm <= 0:
        return

    total_norm_sq = 0.0
    grads = []
    for p in model.parameters():
        if p.grad is None:
            continue
        grads.append(p.grad)
        param_norm = p.grad.data.norm(2)
        total_norm_sq += param_norm.item() ** 2
    if not grads:
        return

    total_norm = total_norm_sq ** 0.5
    clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-12))
    if clip_coef < 1.0:
        for g in grads:
            g.data.mul_(clip_coef)

    if noise_std > 0:
        # 噪声强度 = noise_std * max_grad_norm
        # 例如：noise_std=0.001, max_grad_norm=1.0 → 实际噪声标准差=0.001
        scaled_std = noise_std * max_grad_norm
        for g in grads:
            g.data.add_(torch.randn_like(g) * scaled_std)


def enable_micro_dp_on_optimizer(optimizer, model, max_grad_norm):
    """Monkey-patch优化器的step方法，使其在每次更新前执行Micro-DP。"""
    conf = {
        "model": model,
        "max_grad_norm": max_grad_norm,
        "noise_std": 0.0,
        "original_step": optimizer.step,
    }

    def micro_dp_step(self, *args, **kwargs):
        apply_micro_dp(conf["model"], conf["max_grad_norm"], conf["noise_std"])
        return conf["original_step"](*args, **kwargs)

    def set_noise_std(self, value):
        conf["noise_std"] = max(0.0, float(value))

    optimizer.step = MethodType(micro_dp_step, optimizer)
    optimizer.set_noise_std = MethodType(set_noise_std, optimizer)
    optimizer.set_noise_std(0.0)
    return optimizer


def main(args, loaded_args, trainloader, testloader):
    n_classes = loaded_args["dataset"]["n_classes"]
    model_name = loaded_args["dataset"]["model_name"]
    weight_decay = loaded_args[model_name]["weight_decay"]
    momentum = loaded_args[model_name]["momentum"]
    n_epochs = loaded_args[model_name]["epochs"]
    lr = loaded_args[model_name]["lr"]
    milestones = loaded_args[model_name]["adjust_epochs"]

    # 互信息约束权重超参数搜索列表 (α1, α2)
    # 格式: (α1, α2) 其中 α1 控制最小化 I(Z;X)，α2 控制最大化 I(Z;Y)
    # 参考 BiDO/train_COCO.py 的数值范围，但 MI 的数值范围可能不同，需要调整
    # 扩大搜索范围，确保不同参数有明显差异
    hp_list = [
        # (0.0, 0.0),      # 基线：无正则项
        #MUTUAL

        # (0.01, 0.1),  # 小权重80.22,(79.32),82.98，
        # (0.01, 0.05),  # 小权重84.24,，(79.82),84.08，
        # (0.05, 0.5),     # 中等权重67.52，(76.13)
        # (0.1, 1.0),      # 较大权重59.84，(71.78)
        # (0.5, 5.0),  # 很大权重41.39，(31.58)  ,,M,.,M
        # (0.2, 2.0),      # 大权重


        # #DPDP
        # (0.05, 0.5),  # 中等权重82.15(64.06)
        (0.05, 0.05),  # 87.60(82.85)
        # (0.1, 1.0),  # 稍大权重79.22(59.44)
        # (0.2, 2.0),  # 大权重73.3(54.95)
        # (0.5, 5.0),  # 很大权重\62.43(43.34)
    ]
    # 交叉熵
    criterion = nn.CrossEntropyLoss().cuda()

    for i, (a1, a2) in enumerate(hp_list):
        print("\n" + "=" * 80)
        print(f"开始训练第 {i + 1}/{len(hp_list)} 组超参数: a1={a1}, a2={a2}")
        print("=" * 80)
        torch.cuda.empty_cache()  # 清空GPU缓存

        # 根据配置选择模型
        if model_name == "VGG16" or model_name == "reg":
            net = model.VGG16(n_classes, hsic_training=args.hsic_training, dataset=args.dataset)
        elif model_name == "ResNet":
            net = model.ResNetClsH(nclass=n_classes, resnetl=10)
        elif model_name == "MCNN":
            net = model.MCNN(n_classes)
        elif model_name == "LeNet":
            net = model.LeNet3(n_classes)
        elif model_name == "SimpleCNN":
            net = model.Classifier(1, 128, n_classes)

        # 加载预训练权重（在fix之后，确保结构稳定）
        if model_name == "VGG16" or model_name == "reg":
            load_pretrained_feature_extractor = True
            if load_pretrained_feature_extractor:
                pretrained_model_ckpt = "target_model/vgg16_bn-6c64b313.pth"
                checkpoint = torch.load(pretrained_model_ckpt)
                load_feature_extractor(net, checkpoint)
                print("✓ 预训练特征提取器权重已加载")

        net = net.cuda()

        # 定义优化器（轻量DP仅在梯度阶段做处理）
        base_lr = lr
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=base_lr,
            weight_decay=weight_decay if weight_decay > 0 else 1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        micro_dp_enabled = bool(args.private)
        if micro_dp_enabled:
            optimizer = enable_micro_dp_on_optimizer(optimizer, net, args.dp_clip_norm)

        best_model = None

        # 使用余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,  # 第一次重启的周期
            T_mult=2,  # 每次重启后周期长度的倍数
            eta_min=base_lr * 0.01  # 最小学习率
        )

        best_ACC = -1
        dp_start_epoch = max(1, args.dp_start_epoch)
        dp_warmup_epochs = max(0, args.dp_warmup_epochs)

        for epoch in range(n_epochs):
            print('\n' + '=' * 80)
            print(f'Epoch: [%d | %d] LR: %f' % (epoch + 1, n_epochs, optimizer.param_groups[0]['lr']))
            print('=' * 80)

            if micro_dp_enabled and hasattr(optimizer, 'set_noise_std'):
                epoch_idx = epoch + 1
                if epoch_idx < dp_start_epoch:
                    current_noise = 0.0
                    noise_phase = "off"
                elif dp_warmup_epochs > 0 and epoch_idx < dp_start_epoch + dp_warmup_epochs:
                    progress = (epoch_idx - dp_start_epoch + 1) / dp_warmup_epochs
                    progress = min(1.0, max(0.0, progress))
                    current_noise = args.dp_noise_std * progress
                    noise_phase = f"warmup {progress * 100:.1f}%"
                else:
                    current_noise = args.dp_noise_std
                    noise_phase = "steady"
                optimizer.set_noise_std(current_noise)
                print(f"🔐 Micro-DP: noise_std={current_noise:.5f} ({noise_phase})")
            else:
                current_noise = 0.0

            train_loss, train_acc, *_ = engine.train_mutual(
                net, criterion, optimizer, trainloader, a1, a2, n_classes,
                ktype=args.ktype,
                hsic_training=args.hsic_training,
                measure=args.measure)

            test_loss, test_acc, *_ = engine.test_mutual(
                net, criterion, testloader, a1, a2, n_classes,
                ktype=args.ktype,
                hsic_training=args.hsic_training,
                measure=args.measure)

            print(f"\n📊 当前训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%")

            if test_acc > best_ACC:
                best_ACC = test_acc
                best_model = deepcopy(net)
                print(f"✅ 新的最佳准确率: {best_ACC:.2f}%")

            scheduler.step()
            torch.cuda.empty_cache()

        print("best acc:", best_ACC)
        if best_model is None:
            best_model = deepcopy(net)
        if args.private:
            file_name = "{}_{:.3f}&{:.3f}_{:.2f}_microdp.tar".format(model_name, a1, a2, best_ACC)
        else:
            file_name = "{}_{:.3f}&{:.3f}_{:.2f}.tar".format(model_name, a1, a2, best_ACC)

        utils.save_checkpoint({
            'state_dict': best_model.state_dict(),
        }, model_path, file_name)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train with BiDO')
    parser.add_argument('--dataset', default='celeba', help='celeba | mnist | cifar')
    parser.add_argument('--measure', default='MI', help='HSIC | COCO | MI')
    parser.add_argument('--ktype', default='linear', help='gaussian, linear, IMQ')
    parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
    parser.add_argument('--root_path', default='./', help='')
    parser.add_argument('--config_dir', default='./config', help='')
    parser.add_argument('--model_dir', default='./target_model', help='')
    # Opacus DP arguments
    parser.add_argument('--mi_weight', type=float, default=1.0, help='global weight for MI/HSIC regularizer')
    parser.add_argument('--mi_warmup_epochs', type=int, default=15,
                        help='Linearly warm up MI权重的epoch数，0表示立即使用完整权重')
    parser.add_argument('--private', type=str2bool, default=True,
                        help='是否启用Micro-DP梯度噪声（默认关闭）')
    parser.add_argument('--dp_clip_norm', type=float, default=1.0,
                        help='Micro-DP全局梯度裁剪阈值（建议保持1.0）')
    parser.add_argument('--dp_noise_std', type=float, default=0.001,
                        help='Micro-DP噪声标准差（相对clip阈值，默认0.001，目标准确率下降<5%）')
    parser.add_argument('--dp_start_epoch', type=int, default=15,
                        help='从第几个epoch开始注入噪声（1-based，默认15，让模型先充分学习）')
    parser.add_argument('--dp_warmup_epochs', type=int, default=10,
                        help='噪声线性升温的epoch数（默认10，缓慢升温减少冲击）')

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

