
import subprocess
import os

# 各脚本的绝对路径（你可以根据实际位置调整）coco跑完
scripts = [

    # "/home/jys/.virtualenvs/Defend_MI-master/BiDO/train_mutual_dp_test.py",#0.05，0.1，试试更低的会不会效果更好，6.0，0.8，0.008

    # "/home/jys/.virtualenvs/Defend_MI-master/DMI/recovery.py",#更新一下vib（）

    # "/home/jys/.virtualenvs/Defend_MI-master/DMI/recovery.py",#hsic的一系列数据跑一下结果（没跑完）


    # "/home/jys/.virtualenvs/Defend_MI-master/DMI/k+1_gan_COCO.py",#coco
    # coco图像生成
    # "/home/jys/.virtualenvs/Defend_MI-master/DMI/recovery_COCO.py",#结果不对，继续修改


#6.	试一下用新的recover文件跑之前的模型看看效果，效果很好
    # "/home/jys/.virtualenvs/Defend_MI-master/DMI/recover_MI_net.py",

    # COCO 未落地到引擎、预训练恒开、约束被 clamp 削弱
    # "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_COCO.py",


# 跑出来的coco准确率还是不行，继续修改代码，把engine换成之前的
# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_COCO.py",

    # "/data2/jys/.virtualenvs/Defend_MI-master/DMI/k+1_gan_COCO.py",#先跑出来了一个，先k+1，修改了，暂时不用这一项



# 跑出来的coco准确率还是不行，继续修改代码，把engine换成之前的，改了ab的参数
# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_COCO.py",

# hsic模型准确率会达到60%，攻击准确率达到50%，修改了（前面一版的准确率还是有78，我需要三个一个80，一个70，一个60做对比）
# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_HSIC.py",
#     没加断点保存的版本放在了same里面跑
# "/home/jys/.virtualenvs/Defend_MI-master/BiDO/train_hsic_same.py",


# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/k+1_gan_HSIC.py",现在整不了，还没有准确率出来
# hsic也重新跑，用之前的engine,没有修改的hsic


#MI找一个准确率低的例子
# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_mutual_jianjinDP.py",#改了预训练模型

# # vib的一组数据，73.几，不准学术造假，所以要选择一个数据让他位于上下之间
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/k+1_gan_vib.py",



# 1."/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_COCO.py",
# 2."/home/jys/.virtualenvs/Defend_MI-master/BiDO/train_hsic_same.py",
#     # 记得跑之前改一下engine
# 3."/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_mutual_jianjinDP.py",



# #0.05，0.1，试试更低的会不会效果更好，6.0，0.8，0.008,后续
#     "/home/jys/.virtualenvs/Defend_MI-master/DMI/k+1_gan_MI.py",
#     "/home/jys/.virtualenvs/Defend_MI-master/DMI/recover_MI_DP.py"
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/k+1_gan_COCO.py",


# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/k+1_gan_MI_DP.py",

# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_mutual.py",
# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_mutual.py",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/recovery_COCO.py",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/recovery.py",
# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_HSIC.py",



    # MI
# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO使用MI/train_mutual_jianjinDP.py",

# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_mutual.py",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/recovery.py",
# LS（训练和测试）
# "/data2/jys/.virtualenvs/Defend_MI-master/LS/train_model.py",

# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_COCO.py",
# "/data2/jys/.virtualenvs/Defend_MI-master/BiDO/train_HSIC.py",

]

for script in scripts:
    script_dir = os.path.dirname(script)
    script_name = os.path.basename(script)

    print(f"\n🚀 开始运行：{script_name}")

    # 在脚本所在目录执行，防止找不到依赖文件
    result = subprocess.run(
        ["python", script_name],
        cwd=script_dir,  # 切换工作目录
        check=True
    )

    print(f"✅ 已完成：{script_name}\n")
