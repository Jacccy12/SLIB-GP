# #加入DP版本，跑mi可用
# import torch, time, utils, hsic
# from util import AverageMeter, accuracy
# from tqdm import tqdm
#
# device = "cuda"
#
#
# # 计算主要公式
#
# def test(model, criterion, dataloader):
#     tf = time.time()
#     model.eval()
#     loss, cnt, ACC = 0.0, 0, 0
#
#     for img, iden in dataloader:
#         img, iden = img.to(device), iden.to(device)
#         bs = img.size(0)  # 当前批次大小
#         iden = iden.view(-1)
#
#         out_digit = model(img)[-1]
#         out_iden = torch.argmax(out_digit, dim=1).view(-1)
#         ACC += torch.sum(iden == out_iden).item()
#         cnt += bs
#
#     return ACC * 100.0 / cnt
#
#
# def multilayer_hsic(model, criterion, inputs, target, a1, a2, n_classes, ktype, hsic_training, measure):
#     """使用HSIC / COCO / MI度量优化隐藏层特征与目标 / 输入的独立性
#     Args:
#         model: 模型
#         criterion: 损失函数
#         inputs: 输入数据
#         target: 目标标签
#         a1, a2: 权重参数
#         n_classes: 类别数
#         ktype: 核类型
#         hsic_training: 是否使用HSIC训练
#         measure: 度量方式
#     """
#     hx_l_list = []
#     hy_l_list = []
#     bs = inputs.size(0)
#     total_loss = 0
#
#     if hsic_training:
#         # 获取模型输出
#         hiddens, out_digit = model(inputs)
#
#         # 检查模型输出
#         if torch.isnan(out_digit).any():
#             print("Warning: NaN in model output")
#             out_digit = torch.nan_to_num(out_digit, nan=0.0)
#
#         # 计算交叉熵损失
#         cross_loss = criterion(out_digit, target)
#
#         # 检查损失值
#         if torch.isnan(cross_loss):
#             print("Warning: NaN in cross entropy loss")
#             cross_loss = torch.tensor(0.0, device=cross_loss.device)
#
#             # # === SSD后处理 ===
#             # D_prime = torch.eye(out_digit.size(1)).to(out_digit.device)
#             # out_digit = ssd.Stealthy_Shield_Defense(out_digit, D_prime)
#             # # =================
#
#         total_loss += cross_loss
#
#         # 转换目标为one-hot编码
#         h_target = utils.to_categorical(target, num_classes=n_classes).float()
#         h_data = inputs.view(bs, -1)
#
#         # 对每个隐藏层计算互信息
#         for i, hidden in enumerate(hiddens):
#             hidden = hidden.view(bs, -1)
#
#             # 检查隐藏层特征
#             if torch.isnan(hidden).any():
#                 print(f"Warning: NaN in hidden layer {i}")
#                 hidden = torch.nan_to_num(hidden, nan=0.0)
#
#             try:
#                 # 根据度量方式计算互信息
#                 if measure == 'HSIC':
#                     hxz_l, hyz_l = hsic.hsic_objective(
#                         hidden,
#                         h_target=h_target.float(),
#                         h_data=h_data,
#                         sigma=5.,
#                         ktype=ktype
#                     )
#                 elif measure == 'COCO':
#                     hxz_l, hyz_l = hsic.coco_objective(
#                         hidden,
#                         h_target=h_target.float(),
#                         h_data=h_data,
#                         sigma=5.,
#                         ktype=ktype
#                     )
#                 elif measure == 'MI':
#                     # 对MI计算添加额外的数值稳定性检查
#                     if hidden.size(1) > 1024:  # 如果特征维度太大，进行降维
#                         with torch.no_grad():
#                             proj_matrix = torch.randn(hidden.size(1), 512, device=hidden.device) / (
#                                         hidden.size(1) ** 0.5)
#                             hidden = hidden @ proj_matrix
#
#                     # 确保数据在合理范围内
#                     hidden = torch.clamp(hidden, min=-10.0, max=10.0)
#                     h_data = torch.clamp(h_data, min=-10.0, max=10.0)
#                     h_target = torch.clamp(h_target, min=-10.0, max=10.0)
#
#                     hxz_l, hyz_l = hsic.mi_objective(
#                         hidden,
#                         h_target=h_target.float(),
#                         h_data=h_data
#                     )
#
#                 # 检查互信息值
#                 if torch.isnan(hxz_l) or torch.isnan(hyz_l):
#                     print(f"Warning: NaN in mutual information calculation for layer {i}")
#                     hxz_l = torch.tensor(0.0, device=hxz_l.device)
#                     hyz_l = torch.tensor(0.0, device=hyz_l.device)
#
#                 # 计算加权损失，添加梯度裁剪
#                 temp_hsic = a1 * hxz_l - a2 * hyz_l
#                 temp_hsic = torch.clamp(temp_hsic, min=-10.0, max=10.0)
#
#
#                 # # 计算加权损失；HSIC/MI 为稳定性保留 clamp，COCO 不做 clamp 以保留大系数的效果
#                 # temp_hsic = a1 * hxz_l - a2 * hyz_l
#                 # if measure in ('HSIC', 'MI'):
#                 #     temp_hsic = torch.clamp(temp_hsic, min=-10.0, max=10.0)
#
#
#                 # 检查加权损失
#                 if torch.isnan(temp_hsic):
#                     print(f"Warning: NaN in weighted loss for layer {i}")
#                     temp_hsic = torch.tensor(0.0, device=temp_hsic.device)
#
#                 total_loss += temp_hsic
#
#                 # 记录互信息值
#                 hx_l_list.append(round(hxz_l.item(), 5))
#                 hy_l_list.append(round(hyz_l.item(), 5))
#             except Exception as e:
#                 print(f"Error in HSIC/MI calculation for layer {i}: {e}")
#                 hx_l_list.append(0)
#                 hy_l_list.append(0)
#
#     else:
#         # 不使用HSIC训练的情况
#         feats, out_digit = model(inputs)
#         cross_loss = criterion(out_digit, target)
#         # # === SSD后处理 ===
#         # D_prime = torch.eye(out_digit.size(1)).to(out_digit.device)
#         # out_digit = ssd.Stealthy_Shield_Defense(out_digit, D_prime)
#         # # =================
#         total_loss += cross_loss
#
#         h_target = utils.to_categorical(target, num_classes=n_classes).float()
#         h_data = inputs.view(bs, -1)
#
#         hxy_l = hsic.coco_normalized_cca(h_data, out_digit, sigma=5., ktype=ktype)
#         temp_hsic = a1 * hxy_l
#         total_loss += temp_hsic
#
#         hxz_l = hxy_l
#         hyz_l = hxy_l
#         hx_l_list.append(round(hxz_l.item(), 5))
#         hy_l_list.append(round(hyz_l.item(), 5))
#
#     # 最终检查总损失
#     if torch.isnan(total_loss):
#         print("Warning: NaN in total loss")
#         total_loss = torch.tensor(0.0, device=total_loss.device)
#
#     return total_loss, cross_loss, out_digit, hx_l_list, hy_l_list
#
# def train_HSIC(model, criterion, optimizer, trainloader, a1, a2, n_classes,
#                ktype='gaussian', hsic_training=True, measure='HSIC'):
#     # 集成HSIC约束的训练流程，平衡分类精度和隐私保护
#     # optimizer优化器
#     # n_classes分类任务的类别数
#     # trainloader训练数据的dataloader
#     # hsic_training=True决定是否启用HSIC约束训练
#     model.train()
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     loss_cls = AverageMeter()  # 分类损失
#     lxz, lyz = AverageMeter(), AverageMeter()
#     end = time.time()
#     # AverageMeter用于计算一个数值序列的平均值
#
#     pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)
#
#     for batch_idx, (inputs, iden) in pbar:
#         data_time.update(time.time() - end)
#         bs = inputs.size(0)
#         inputs, iden = inputs.to(device), iden.to(device)  # 输入和标签
#         iden = iden.view(-1)
#
#         loss, cross_loss, out_digit, hx_l_list, hy_l_list = multilayer_hsic(model, criterion, inputs, iden, a1, a2,
#                                                                             n_classes, ktype, hsic_training, measure)
#         optimizer.zero_grad()  # 清空优化器的梯度
#         loss.backward()  # 计算梯度
#         optimizer.step()  # 执行梯度更新
#
#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
#         losses.update(loss.item())  # 更新当前批次的总损失
#         loss_cls.update(float(cross_loss.detach().cpu().numpy()))  # 更新分类损失
#         lxz.update(sum(hx_l_list) / len(hx_l_list))
#         lyz.update(sum(hy_l_list) / len(hy_l_list))
#
#         top1.update(prec1.item())
#         top5.update(prec5.item())
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)  # 时间更新
#         end = time.time()
#
#         # plot progress
#         msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
#               'top1:{top1: .4f} | top5:{top5: .4f}'.format(
#             cls=loss_cls.avg,
#             lxz=lxz.avg,
#             lyz=lyz.avg,
#             loss=losses.avg,
#             top1=top1.avg,
#             top5=top5.avg,
#         )  # 用于显示进度条的描述信息
#         pbar.set_description(msg)
#     print("hx_l_list:", hx_l_list)
#     print("hy_l_list:", hy_l_list)
#     return losses.avg, top1.avg
#
#
# def test_HSIC(model, criterion, testloader, a1, a2, n_classes, ktype='gaussian', hsic_training=True, measure='HSIC'):
#     model.eval()
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     loss_cls = AverageMeter()
#     lxz, lyz = AverageMeter(), AverageMeter()
#     end = time.time()
#
#     pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)
#     with torch.no_grad():
#         for batch_idx, (inputs, iden) in pbar:
#             data_time.update(time.time() - end)
#
#             inputs, iden = inputs.to(device), iden.to(device)
#             bs = inputs.size(0)
#             iden = iden.view(-1)
#
#             loss, cross_loss, out_digit, hx_l_list, hy_l_list = multilayer_hsic(model, criterion, inputs, iden, a1, a2,
#                                                                                 n_classes, ktype, hsic_training,
#                                                                                 measure)
#
#             # measure accuracy and record loss
#
#             prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
#             losses.update(loss.item(), bs)
#             loss_cls.update(cross_loss.item(), bs)
#             lxz.update(sum(hx_l_list) / len(hx_l_list), bs)
#             lyz.update(sum(hy_l_list) / len(hy_l_list), bs)
#
#             top1.update(prec1.item(), bs)
#             top5.update(prec5.item(), bs)
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             # plot progress
#             msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
#                   'top1:{top1: .4f} | top5:{top5: .4f}'.format(
#                 cls=loss_cls.avg,
#                 lxz=lxz.avg,
#                 lyz=lyz.avg,
#                 loss=losses.avg,
#                 top1=top1.avg,
#                 top5=top5.avg,
#             )
#             pbar.set_description(msg)
#
#     print("hx_l_list:", hx_l_list)
#     print("hy_l_list:", hy_l_list)
#     print('-' * 80)
#     return losses.avg, top1.avg
#
#
# def train_reg(model, criterion, optimizer, trainloader):
#     # 常规分类训练（交叉熵损失）
#     model.train()
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     loss_cls = AverageMeter()
#     end = time.time()  # 记录当前时间，用于计算每个批次的时间
#
#     pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)
#
#     for batch_idx, (inputs, iden) in pbar:
#         inputs, iden = inputs.to(device), iden.to(device)
#         iden = iden.view(-1)
#
#         feats, out_digit = model(inputs)
#         cross_loss = criterion(out_digit, iden)  # iden是真实标签
#         loss = cross_loss
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure accuracy and record loss
#         bs = inputs.size(0)
#         prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
#         losses.update(loss.item(), bs)
#         loss_cls.update(cross_loss.item(), bs)
#
#         top1.update(prec1.item(), bs)
#         top5.update(prec5.item(), bs)
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         # plot progress
#         msg = '({batch}/{size}) | ' \
#               'Loss:{loss:.4f} | ' \
#               'top1:{top1: .4f} | top5:{top5: .4f}'.format(
#             batch=batch_idx + 1,
#             size=len(trainloader),
#             cls=loss_cls.avg,
#             loss=losses.avg,
#             top1=top1.avg,
#             top5=top5.avg,
#         )
#         pbar.set_description(msg)
#
#     return losses.avg, top1.avg
#
#
# def test_reg(model, criterion, testloader):
#     model.eval()
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     end = time.time()
#
#     pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)
#
#     with torch.no_grad():
#         for batch_idx, (inputs, iden) in pbar:
#             data_time.update(time.time() - end)
#
#             inputs, iden = inputs.to(device), iden.to(device)
#             bs = inputs.size(0)
#             iden = iden.view(-1)
#             feats, out_digit = model(inputs)
#             cross_loss = criterion(out_digit, iden)
#
#             loss = cross_loss
#
#             # measure accuracy and record loss
#             prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
#             losses.update(loss.item(), bs)
#             top1.update(prec1.item(), bs)
#             top5.update(prec5.item(), bs)
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             # plot progress
#             # plot progress
#             msg = '({batch}/{size}) | ' \
#                   'Loss:{loss:.4f} | ' \
#                   'top1:{top1: .4f} | top5:{top5: .4f}'.format(
#                 batch=batch_idx + 1,
#                 size=len(testloader),
#                 loss=losses.avg,
#                 top1=top1.avg,
#                 top5=top5.avg,
#             )
#             pbar.set_description(msg)
#
#     return losses.avg, top1.avg
#
#
# def train_vib(model, criterion, optimizer, trainloader, beta=1e-2):  # 控制信息瓶颈的强度的超参数
#     # 变分信息瓶颈（VIB）方法，通过信息约束提升模型鲁棒性
#     model.train()
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     end = time.time()
#
#     pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)
#
#     for batch_idx, (inputs, targets) in pbar:
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         inputs, targets = inputs.cuda(), targets.cuda()
#         bs = inputs.size(0)
#
#         # compute output
#         _, mu, std, out_digit = model(inputs)  # mu均值，out-digit分类输出，std标准差
#         cross_loss = criterion(out_digit, targets)
#         info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
#         # 计算信息损失（变分约束）计算潜在变量分布的负对数似然或KL散度损失
#         loss = cross_loss + beta * info_loss
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure accuracy and record loss
#         prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
#         losses.update(loss.item(), bs)
#         top1.update(prec1.item(), bs)
#         top5.update(prec5.item(), bs)
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         # plot progress
#         # plot progress
#         msg = '({batch}/{size}) | ' \
#               'Loss:{loss:.4f} | ' \
#               'top1:{top1: .4f} | top5:{top5: .4f}'.format(
#             batch=batch_idx + 1,
#             size=len(trainloader),
#             cls=losses.avg,
#             loss=losses.avg,
#             top1=top1.avg,
#             top5=top5.avg,
#         )
#         pbar.set_description(msg)
#     return losses.avg, top1.avg
#
#
# def test_vib(model, criterion, testloader, beta=1e-2):
#     global best_acc
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     end = time.time()
#     pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in pbar:
#             # measure data loading time
#             data_time.update(time.time() - end)
#
#             inputs, targets = inputs.cuda(), targets.cuda()
#             bs = inputs.size(0)
#
#             # compute output
#             _, mu, std, out_digit = model(inputs)
#             cross_loss = criterion(out_digit, targets)
#             info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
#             loss = cross_loss + beta * info_loss
#
#             # measure accuracy and record loss
#             prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
#             losses.update(loss.item(), bs)
#             top1.update(prec1.item(), bs)
#             top5.update(prec5.item(), bs)
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             # plot progress
#             msg = '({batch}/{size}) | ' \
#                   'Loss:{loss:.4f} | ' \
#                   'top1:{top1: .4f} | top5:{top5: .4f}'.format(
#                 batch=batch_idx + 1,
#                 size=len(testloader),
#                 cls=losses.avg,
#                 loss=losses.avg,
#                 top1=top1.avg,
#                 top5=top5.avg,
#             )
#             pbar.set_description(msg)
#     return losses.avg, top1.avg
#
#
# def train_mutual(model, criterion, optimizer, trainloader, a1, a2, n_classes,
#                  ktype='gaussian', hsic_training=True, measure='MI'):
#     model.train()
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     loss_cls = AverageMeter()
#     lxz, lyz = AverageMeter(), AverageMeter()
#     end = time.time()
#
#     # 使用tqdm，并确保完成后保留进度条
#     pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150, leave=True)
#
#     for batch_idx, (inputs, targets) in pbar:
#         try:
#             data_time.update(time.time() - end)
#             bs = inputs.size(0)
#
#             # 确保数据在正确的设备上，使用异步传输
#             device = next(model.parameters()).device
#             inputs = inputs.to(device, non_blocking=True)
#             targets = targets.to(device, non_blocking=True)
#             targets = targets.view(-1)
#
#             # 清空之前的梯度
#             optimizer.zero_grad(set_to_none=True)
#
#             # 计算loss和更新参数
#             loss, cross_loss, out_digit, hx_l_list, hy_l_list = multilayer_hsic(
#                 model, criterion, inputs, targets, a1, a2, n_classes, ktype, hsic_training, measure
#             )
#
#             # 检查loss是否为NaN
#             if torch.isnan(loss):
#                 print(f"Warning: Loss is NaN at batch {batch_idx}")
#                 print(f"cross_loss: {cross_loss}")
#                 print(f"hx_l_list: {hx_l_list}")
#                 print(f"hy_l_list: {hy_l_list}")
#                 continue
#
#             # 反向传播
#             loss.backward()
#
#             # 检查梯度是否为NaN
#             if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
#                 print(f"Warning: Gradients are NaN at batch {batch_idx}")
#                 optimizer.zero_grad(set_to_none=True)
#                 continue
#
#             # 更新参数
#             optimizer.step()
#
#             # 计算准确率并更新统计信息
#             with torch.no_grad():  # 使用no_grad减少内存使用
#                 prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
#                 losses.update(loss.item())
#                 loss_cls.update(float(cross_loss.detach().cpu().numpy()))
#                 lxz.update(sum(hx_l_list) / len(hx_l_list))
#                 lyz.update(sum(hy_l_list) / len(hy_l_list))
#                 top1.update(prec1.item())
#                 top5.update(prec5.item())
#
#             # 主动释放不需要的张量
#             del loss, cross_loss, out_digit, hx_l_list, hy_l_list
#             torch.cuda.empty_cache()
#
#             # 更新时间统计
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             # 更新进度条信息
#             msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
#                   'top1:{top1: .4f} | top5:{top5: .4f}'.format(
#                 cls=loss_cls.avg,
#                 lxz=lxz.avg,
#                 lyz=lyz.avg,
#                 loss=losses.avg,
#                 top1=top1.avg,
#                 top5=top5.avg,
#             )
#             pbar.set_description(msg)
#
#         except RuntimeError as e:
#             if "out of memory" in str(e):
#                 print(f"OOM error in batch {batch_idx}, trying to recover...")
#                 # 清理所有中间变量
#                 if 'loss' in locals():
#                     del loss, cross_loss, out_digit, hx_l_list, hy_l_list
#                 torch.cuda.empty_cache()
#                 continue
#             elif "unscale_() has already been called" in str(e):
#                 print(f"Warning: Unscale error in batch {batch_idx}, skipping...")
#                 optimizer.zero_grad(set_to_none=True)
#                 continue
#             raise e
#
#     return losses.avg, top1.avg
#
#
# def test_mutual(model, criterion, testloader, a1, a2, n_classes, ktype='gaussian', hsic_training=True,
#                 measure='MI'):
#     model.eval()
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     loss_cls = AverageMeter()
#     lxz, lyz = AverageMeter(), AverageMeter()
#     end = time.time()
#
#     pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in pbar:
#             data_time.update(time.time() - end)
#
#             inputs, targets = inputs.to(device), targets.to(device)
#             bs = inputs.size(0)
#             targets = targets.view(-1)
#
#             loss, cross_loss, out_digit, hx_l_list, hy_l_list = multilayer_hsic(model, criterion, inputs, targets,
#                                                                                 a1, a2, n_classes, ktype,
#                                                                                 hsic_training, measure)
#             prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
#             losses.update(loss.item(), bs)
#             loss_cls.update(cross_loss.item(), bs)
#             lxz.update(sum(hx_l_list) / len(hx_l_list), bs)
#             lyz.update(sum(hy_l_list) / len(hy_l_list), bs)
#             top1.update(prec1.item(), bs)
#             top5.update(prec5.item(), bs)
#             batch_time.update(time.time() - end)
#             end = time.time()
#             msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
#                   'top1:{top1: .4f} | top5:{top5: .4f}'.format(
#                 cls=loss_cls.avg,
#                 lxz=lxz.avg,
#                 lyz=lyz.avg,
#                 loss=losses.avg,
#                 top1=top1.avg,
#                 top5=top5.avg,
#             )
#             pbar.set_description(msg)
#     return losses.avg, top1.avg

#修改了不同 a1/a2 参数效果相同
# 加入DP版本
import torch, time, utils, hsic
from util import AverageMeter, accuracy
from tqdm import tqdm

device = "cuda"


# 计算主要公式

def test(model, criterion, dataloader):
    tf = time.time()
    model.eval()
    loss, cnt, ACC = 0.0, 0, 0

    for img, iden in dataloader:
        img, iden = img.to(device), iden.to(device)
        bs = img.size(0)  # 当前批次大小
        iden = iden.view(-1)

        out_digit = model(img)[-1]
        out_iden = torch.argmax(out_digit, dim=1).view(-1)
        ACC += torch.sum(iden == out_iden).item()
        cnt += bs

    return ACC * 100.0 / cnt


def multilayer_hsic(model, criterion, inputs, target, a1, a2, n_classes, ktype, hsic_training, measure, mi_weight=1.0):
    """使用HSIC / COCO / MI度量优化隐藏层特征与目标 / 输入的独立性
    Args:
        model: 模型
        criterion: 损失函数
        inputs: 输入数据
        target: 目标标签
        a1, a2: 权重参数
        n_classes: 类别数
        ktype: 核类型
        hsic_training: 是否使用HSIC训练
        measure: 度量方式
    """
    hx_l_list = []
    hy_l_list = []
    bs = inputs.size(0)
    total_loss = 0
    reg_sum = 0.0

    if hsic_training:
        # 获取模型输出
        hiddens, out_digit = model(inputs)

        # 检查模型输出
        if torch.isnan(out_digit).any():
            print("Warning: NaN in model output")
            out_digit = torch.nan_to_num(out_digit, nan=0.0)

        # 计算交叉熵损失
        cross_loss = criterion(out_digit, target)

        # 检查损失值
        if torch.isnan(cross_loss):
            print("Warning: NaN in cross entropy loss")
            cross_loss = torch.tensor(0.0, device=cross_loss.device)

            # # === SSD后处理 ===
            # D_prime = torch.eye(out_digit.size(1)).to(out_digit.device)
            # out_digit = ssd.Stealthy_Shield_Defense(out_digit, D_prime)
            # # =================

        total_loss += cross_loss

        # 转换目标为one-hot编码
        h_target = utils.to_categorical(target, num_classes=n_classes).float()
        h_data = inputs.view(bs, -1)

        # 对每个隐藏层计算互信息
        for i, hidden in enumerate(hiddens):
            hidden = hidden.view(bs, -1)

            # 检查隐藏层特征
            if torch.isnan(hidden).any():
                print(f"Warning: NaN in hidden layer {i}")
                hidden = torch.nan_to_num(hidden, nan=0.0)

            try:
                # 参考 BiDO 版本：直接使用完整 batch，不做子采样
                # 只有在真正 OOM 时才考虑子采样（注释掉以保持与 BiDO 一致）
                # max_mi_batch = 128
                # if bs > max_mi_batch:
                #     idx = torch.randperm(bs, device=hidden.device)[:max_mi_batch]
                #     hidden_sub = hidden.index_select(0, idx)
                #     h_data_sub = h_data.index_select(0, idx)
                #     h_target_sub = h_target.index_select(0, idx)
                # else:
                #     hidden_sub = hidden
                #     h_data_sub = h_data
                #     h_target_sub = h_target

                # 直接使用完整数据，与 BiDO 版本保持一致
                hidden_sub = hidden
                h_data_sub = h_data
                h_target_sub = h_target

                # 根据度量方式计算互信息（参考 BiDO 版本，不做额外的数据 clamp）
                if measure == 'HSIC' or measure.startswith('HSIC'):
                    hxz_l, hyz_l = hsic.hsic_objective(
                        hidden_sub,
                        h_target=h_target_sub.float(),
                        h_data=h_data_sub,
                        sigma=5.,
                        ktype=ktype
                    )
                elif measure == 'COCO' or measure.startswith('COCO'):
                    hxz_l, hyz_l = hsic.coco_objective(
                        hidden_sub,
                        h_target=h_target_sub.float(),
                        h_data=h_data_sub,
                        sigma=5.,
                        ktype=ktype
                    )
                elif 'MI' in measure:
                    # 对于 MI，如果特征维度太大，使用固定投影矩阵（避免每次随机）
                    # 但参考 BiDO 版本，先尝试不做降维，直接计算
                    # 如果 OOM，再考虑降维
                    try:
                        hxz_l, hyz_l = hsic.mi_objective(
                            hidden_sub,
                            h_target=h_target_sub.float(),
                            h_data=h_data_sub
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e) and hidden_sub.size(1) > 1024:
                            # 只有在 OOM 且特征维度 > 1024 时才降维
                            # 使用固定的投影矩阵（基于特征维度生成，但固定随机种子）
                            with torch.no_grad():
                                torch.manual_seed(42)  # 固定随机种子，确保投影矩阵一致
                                proj_matrix = torch.randn(hidden_sub.size(1), 512, device=hidden_sub.device) / (
                                        hidden_sub.size(1) ** 0.5)
                                hidden_sub = hidden_sub @ proj_matrix
                            hxz_l, hyz_l = hsic.mi_objective(
                                hidden_sub,
                                h_target=h_target_sub.float(),
                                h_data=h_data_sub
                            )
                        else:
                            raise e

                # 检查互信息值
                if torch.isnan(hxz_l) or torch.isnan(hyz_l):
                    print(f"Warning: NaN in mutual information calculation for layer {i}")
                    hxz_l = torch.tensor(0.0, device=hxz_l.device)
                    hyz_l = torch.tensor(0.0, device=hyz_l.device)

                # 记录原始 MI 数值（便于日志观察）
                orig_hxz = hxz_l
                orig_hyz = hyz_l

                # 当 MI 指标接近 1 时，梯度容易饱和；通过 logit 拉伸增强梯度分辨率
                if 'MI' in measure:
                    eps = 1e-4
                    hxz_l = torch.clamp(hxz_l, eps, 1 - eps)
                    hyz_l = torch.clamp(hyz_l, eps, 1 - eps)
                    hxz_l = torch.log(hxz_l / (1 - hxz_l))
                    hyz_l = torch.log(hyz_l / (1 - hyz_l))

                # 计算加权损失；移除 clamp，让 a1/a2 的效果充分体现
                temp_hsic = a1 * hxz_l - a2 * hyz_l

                # 检查加权损失
                if torch.isnan(temp_hsic):
                    print(f"Warning: NaN in weighted loss for layer {i}")
                    temp_hsic = torch.tensor(0.0, device=temp_hsic.device)

                total_loss += mi_weight * temp_hsic
                reg_sum = reg_sum + temp_hsic

                # 记录互信息值
                hx_l_list.append(round(float(orig_hxz.detach().cpu().item()), 5))
                hy_l_list.append(round(float(orig_hyz.detach().cpu().item()), 5))
            except Exception as e:
                print(f"Error in HSIC/MI calculation for layer {i}: {e}")
                hx_l_list.append(0)
                hy_l_list.append(0)

    else:
        # 不使用HSIC训练的情况
        feats, out_digit = model(inputs)
        cross_loss = criterion(out_digit, target)
        # # === SSD后处理 ===
        # D_prime = torch.eye(out_digit.size(1)).to(out_digit.device)
        # out_digit = ssd.Stealthy_Shield_Defense(out_digit, D_prime)
        # # =================
        total_loss += cross_loss

        h_target = utils.to_categorical(target, num_classes=n_classes).float()
        h_data = inputs.view(bs, -1)

        hxy_l = hsic.coco_normalized_cca(h_data, out_digit, sigma=5., ktype=ktype)
        orig_hxy = hxy_l
        if 'MI' in measure:
            eps = 1e-4
            hxy_l = torch.clamp(hxy_l, eps, 1 - eps)
            hxy_l = torch.log(hxy_l / (1 - hxy_l))
        temp_hsic = a1 * hxy_l
        total_loss += mi_weight * temp_hsic
        reg_sum = reg_sum + temp_hsic

        hxz_l = orig_hxy
        hyz_l = orig_hxy
        hx_l_list.append(round(float(hxz_l.detach().cpu().item()), 5))
        hy_l_list.append(round(float(hyz_l.detach().cpu().item()), 5))

    # 最终检查总损失
    if torch.isnan(total_loss):
        print("Warning: NaN in total loss")
        total_loss = torch.tensor(0.0, device=total_loss.device)

    return total_loss, cross_loss, out_digit, hx_l_list, hy_l_list, reg_sum


def train_HSIC(model, criterion, optimizer, trainloader, a1, a2, n_classes,
               ktype='gaussian', hsic_training=True, measure='HSIC'):
    # 集成HSIC约束的训练流程，平衡分类精度和隐私保护
    # optimizer优化器
    # n_classes分类任务的类别数
    # trainloader训练数据的dataloader
    # hsic_training=True决定是否启用HSIC约束训练
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()  # 分类损失
    lxz, lyz = AverageMeter(), AverageMeter()
    end = time.time()
    # AverageMeter用于计算一个数值序列的平均值

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)

    for batch_idx, (inputs, iden) in pbar:
        data_time.update(time.time() - end)
        bs = inputs.size(0)
        inputs, iden = inputs.to(device), iden.to(device)  # 输入和标签
        iden = iden.view(-1)

        loss, cross_loss, out_digit, hx_l_list, hy_l_list, _ = multilayer_hsic(model, criterion, inputs, iden, a1, a2,
                                                                               n_classes, ktype, hsic_training, measure)
        optimizer.zero_grad()  # 清空优化器的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 执行梯度更新

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
        losses.update(loss.item())  # 更新当前批次的总损失
        loss_cls.update(float(cross_loss.detach().cpu().numpy()))  # 更新分类损失
        lxz.update(sum(hx_l_list) / len(hx_l_list))
        lyz.update(sum(hy_l_list) / len(hy_l_list))

        top1.update(prec1.item())
        top5.update(prec5.item())

        # measure elapsed time
        batch_time.update(time.time() - end)  # 时间更新
        end = time.time()

        # plot progress
        msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            cls=loss_cls.avg,
            lxz=lxz.avg,
            lyz=lyz.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )  # 用于显示进度条的描述信息
        pbar.set_description(msg)
    print("hx_l_list:", hx_l_list)
    print("hy_l_list:", hy_l_list)
    return losses.avg, top1.avg


def test_HSIC(model, criterion, testloader, a1, a2, n_classes, ktype='gaussian', hsic_training=True, measure='HSIC'):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)
    with torch.no_grad():
        for batch_idx, (inputs, iden) in pbar:
            data_time.update(time.time() - end)

            inputs, iden = inputs.to(device), iden.to(device)
            bs = inputs.size(0)
            iden = iden.view(-1)

            loss, cross_loss, out_digit, hx_l_list, hy_l_list, _ = multilayer_hsic(model, criterion, inputs, iden, a1,
                                                                                   a2,
                                                                                   n_classes, ktype, hsic_training,
                                                                                   measure)

            # measure accuracy and record loss

            prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            loss_cls.update(cross_loss.item(), bs)
            lxz.update(sum(hx_l_list) / len(hx_l_list), bs)
            lyz.update(sum(hy_l_list) / len(hy_l_list), bs)

            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                cls=loss_cls.avg,
                lxz=lxz.avg,
                lyz=lyz.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)

    print("hx_l_list:", hx_l_list)
    print("hy_l_list:", hy_l_list)
    print('-' * 80)
    return losses.avg, top1.avg


def train_reg(model, criterion, optimizer, trainloader):
    # 常规分类训练（交叉熵损失）
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    end = time.time()  # 记录当前时间，用于计算每个批次的时间

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)

    for batch_idx, (inputs, iden) in pbar:
        inputs, iden = inputs.to(device), iden.to(device)
        iden = iden.view(-1)

        feats, out_digit = model(inputs)
        cross_loss = criterion(out_digit, iden)  # iden是真实标签
        loss = cross_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        bs = inputs.size(0)
        prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
        losses.update(loss.item(), bs)
        loss_cls.update(cross_loss.item(), bs)

        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        msg = '({batch}/{size}) | ' \
              'Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            cls=loss_cls.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)

    return losses.avg, top1.avg


def test_reg(model, criterion, testloader):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)

    with torch.no_grad():
        for batch_idx, (inputs, iden) in pbar:
            data_time.update(time.time() - end)

            inputs, iden = inputs.to(device), iden.to(device)
            bs = inputs.size(0)
            iden = iden.view(-1)
            feats, out_digit = model(inputs)
            cross_loss = criterion(out_digit, iden)

            loss = cross_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out_digit.data, iden.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            # plot progress
            msg = '({batch}/{size}) | ' \
                  'Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)

    return losses.avg, top1.avg


def train_vib(model, criterion, optimizer, trainloader, beta=1e-2):  # 控制信息瓶颈的强度的超参数
    # 变分信息瓶颈（VIB）方法，通过信息约束提升模型鲁棒性
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150)

    for batch_idx, (inputs, targets) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.cuda(), targets.cuda()
        bs = inputs.size(0)

        # compute output
        _, mu, std, out_digit = model(inputs)  # mu均值，out-digit分类输出，std标准差
        cross_loss = criterion(out_digit, targets)
        info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
        # 计算信息损失（变分约束）计算潜在变量分布的负对数似然或KL散度损失
        loss = cross_loss + beta * info_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        # plot progress
        msg = '({batch}/{size}) | ' \
              'Loss:{loss:.4f} | ' \
              'top1:{top1: .4f} | top5:{top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            cls=losses.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        pbar.set_description(msg)
    return losses.avg, top1.avg


def test_vib(model, criterion, testloader, beta=1e-2):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar:
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda()
            bs = inputs.size(0)

            # compute output
            _, mu, std, out_digit = model(inputs)
            cross_loss = criterion(out_digit, targets)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
            loss = cross_loss + beta * info_loss

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            msg = '({batch}/{size}) | ' \
                  'Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                cls=losses.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)
    return losses.avg, top1.avg


def train_mutual(model, criterion, optimizer, trainloader, a1, a2, n_classes,
                 ktype='gaussian', hsic_training=True, measure='MI', mi_weight=1.0):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    reg_term_avg = AverageMeter()  # 记录正则项的平均值
    end = time.time()

    # 使用tqdm，并确保完成后保留进度条
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=150, leave=True)

    for batch_idx, (inputs, targets) in pbar:
        try:
            data_time.update(time.time() - end)
            bs = inputs.size(0)

            # 确保数据在正确的设备上，使用异步传输
            device = next(model.parameters()).device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            targets = targets.view(-1)

            # 清空之前的梯度
            optimizer.zero_grad(set_to_none=True)

            # 计算loss和更新参数
            loss, cross_loss, out_digit, hx_l_list, hy_l_list, reg_sum = multilayer_hsic(
                model, criterion, inputs, targets, a1, a2, n_classes, ktype, hsic_training, measure, mi_weight=mi_weight
            )

            # 检查loss是否为NaN
            if torch.isnan(loss):
                print(f"Warning: Loss is NaN at batch {batch_idx}")
                print(f"cross_loss: {cross_loss}")
                print(f"hx_l_list: {hx_l_list}")
                print(f"hy_l_list: {hy_l_list}")
                continue

            # 反向传播
            loss.backward()

            # 检查梯度是否为NaN
            if any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None):
                print(f"Warning: Gradients are NaN at batch {batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                continue

            # 更新参数
            optimizer.step()

            # 计算准确率并更新统计信息
            with torch.no_grad():  # 使用no_grad减少内存使用
                prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
                losses.update(loss.item())
                loss_cls.update(float(cross_loss.detach().cpu().numpy()))
                lxz.update(sum(hx_l_list) / len(hx_l_list))
                lyz.update(sum(hy_l_list) / len(hy_l_list))
                reg_term_avg.update(float(reg_sum.detach().cpu().numpy()))  # 记录正则项
                top1.update(prec1.item())
                top5.update(prec5.item())

            # 主动释放不需要的张量
            del loss, cross_loss, out_digit, hx_l_list, hy_l_list, reg_sum
            torch.cuda.empty_cache()

            # 更新时间统计
            batch_time.update(time.time() - end)
            end = time.time()

            # 更新进度条信息
            msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                cls=loss_cls.avg,
                lxz=lxz.avg,
                lyz=lyz.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error in batch {batch_idx}, trying to recover...")
                # 清理所有中间变量
                try:
                    del loss, cross_loss, out_digit, hx_l_list, hy_l_list, reg_sum
                except NameError:
                    # 某些变量可能未定义，忽略
                    pass
                torch.cuda.empty_cache()
                continue
            elif "unscale_() has already been called" in str(e):
                print(f"Warning: Unscale error in batch {batch_idx}, skipping...")
                optimizer.zero_grad(set_to_none=True)
                continue
            raise e

    return losses.avg, top1.avg, reg_term_avg.avg


def test_mutual(model, criterion, testloader, a1, a2, n_classes, ktype='gaussian', hsic_training=True,
                measure='MI', mi_weight=1.0):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_cls = AverageMeter()
    lxz, lyz = AverageMeter(), AverageMeter()
    reg_term_avg = AverageMeter()  # 记录正则项的平均值
    end = time.time()

    pbar = tqdm(enumerate(testloader), total=len(testloader), ncols=150)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar:
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device)
            bs = inputs.size(0)
            targets = targets.view(-1)

            loss, cross_loss, out_digit, hx_l_list, hy_l_list, reg_sum = multilayer_hsic(model, criterion, inputs,
                                                                                         targets,
                                                                                         a1, a2, n_classes, ktype,
                                                                                         hsic_training, measure,
                                                                                         mi_weight=mi_weight)
            prec1, prec5 = accuracy(out_digit.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), bs)
            loss_cls.update(cross_loss.item(), bs)
            lxz.update(sum(hx_l_list) / len(hx_l_list), bs)
            lyz.update(sum(hy_l_list) / len(hy_l_list), bs)
            reg_term_avg.update(float(reg_sum.detach().cpu().numpy()), bs)  # 记录正则项
            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)
            batch_time.update(time.time() - end)
            end = time.time()
            msg = 'CE:{cls:.4f} | Lxz(down):{lxz:.5f} | Lyz(up):{lyz:.5f} | Loss:{loss:.4f} | ' \
                  'top1:{top1: .4f} | top5:{top5: .4f}'.format(
                cls=loss_cls.avg,
                lxz=lxz.avg,
                lyz=lyz.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            pbar.set_description(msg)
    return losses.avg, top1.avg, reg_term_avg.avg