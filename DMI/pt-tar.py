import torch

# 假设你已经有一个名为FaceNet_95.88.pt的文件
# 加载.pt文件
model_state_dict = torch.load('/home/jys/.virtualenvs/Defend_MI-master/DMI/eval_ckp/FaceNet_95.88.pt')

# 创建一个字典，用于保存模型状态字典（这里假设只保存状态字典，你可以根据需要添加其他信息）
checkpoint = {
   'state_dict': model_state_dict
}

# 保存为.tar文件
torch.save(checkpoint, '/home/jys/.virtualenvs/Defend_MI-master/DMI/eval_ckp/FaceNet_95.88.tar')