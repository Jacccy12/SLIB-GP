#

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# -----------------------------------
# 1️⃣ 手动定义每个方法的图片路径
# 第一行每个样本有两张图片（原图 + 目标图）
# 后面每行每个样本一张图
# -----------------------------------
image_paths = {
    "Target (GT)": [  # 第一行两张图（原图 + 目标图）
["/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_001_0th.png", "/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_001_3th.png"],
["/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_002_1th.png", "/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_002_4th.png"],
["/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_003_1th.png", "/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_003_4th.png"],
["/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_004_0th.png", "/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_004_3th.png"],
["/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_005_2th.png", "/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_005_4th.png"],
["/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_006_2th.png", "/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_006_3th.png"],
["/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_007_3th.png", "/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_007_4th.png"],
["/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_008_1th.png", "/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/trainset/iden_008_2th.png"],
    ],
    "No Defense": [
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/reg/all/attack_iden_000|2.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/reg/all/attack_iden_001|5.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/reg/all/attack_iden_002|1.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/reg/all/attack_iden_003|4.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/reg/all/attack_iden_004|4.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/reg/all/attack_iden_005|2.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/reg/all/attack_iden_006|1.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/reg/all/attack_iden_007|3.png",
    ],
    "MID": [#79(vib)
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/vib/all/attack_iden_000|3.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/vib/all/attack_iden_001|3.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/vib/all/attack_iden_002|3.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/vib/all/attack_iden_003|4.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/vib/all/attack_iden_004|5.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/vib/all/attack_iden_005|5.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/vib/all/attack_iden_006|4.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/vib/all/attack_iden_007|5.png",
    ],
    "BiDO-COCO": [
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/COCO/all/attack_iden_000|1.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/COCO/all/attack_iden_001|5.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/COCO/all/attack_iden_002|1.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/COCO/all/attack_iden_003|3.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/COCO/all/attack_iden_004|4.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/COCO/all/attack_iden_005|5.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/COCO/all/attack_iden_006|1.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/COCO/all/attack_iden_007|4.png",
    ],
    "BiDO-HSIC": [#67用的还是mi版本的hsic
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/HSIC/all/attack_iden_000|5.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/HSIC/all/attack_iden_001|5.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/HSIC/all/attack_iden_002|2.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/HSIC/all/attack_iden_003|4.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/HSIC/all/attack_iden_004|4.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/HSIC/all/attack_iden_005|5.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/HSIC/all/attack_iden_006|1.png",
"/home/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/HSIC/all/attack_iden_007|4.png",
    ],
"LIB": [#用的MI，无噪声
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI/all/attack_iden_000|1.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI/all/attack_iden_001|5.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI/all/attack_iden_002|1.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI/all/attack_iden_003|3.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI/all/attack_iden_004|4.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI/all/attack_iden_005|5.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI/all/attack_iden_006|1.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI/all/attack_iden_007|4.png",
    ],
"SLIB-GP":[#用的MI+DP，有噪声
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_000|5.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_001|5.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_002|2.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_003|2.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_004|4.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_005|5.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_006|1.png",
"/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_007|3.png",

# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_000|5.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_001|5.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_002|2.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_003|2.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_004|4.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_005|5.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_006|1.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI+DP/all/attack_iden_007|3.png",
#
#
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI++/all/attack_iden_000|1.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI++/all/attack_iden_001|5.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI++/all/attack_iden_002|1.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI++/all/attack_iden_003|3.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI++/all/attack_iden_004|4.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI++/all/attack_iden_005|5.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI++/all/attack_iden_006|1.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MI++/all/attack_iden_007|4.png",
# #
#
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MIOLD/MI/all/attack_iden_000|1.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MIOLD/MI/all/attack_iden_001|5.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MIOLD/MI/all/attack_iden_002|1.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MIOLD/MI/all/attack_iden_003|3.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MIOLD/MI/all/attack_iden_004|4.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MIOLD/MI/all/attack_iden_005|5.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MIOLD/MI/all/attack_iden_006|1.png",
# "/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/celeba/MIOLD/MI/all/attack_iden_007|4.png",
]
}

# -----------------------------------
# 1.1️⃣ 辅助函数：找不到路径时输出空白
# -----------------------------------
def make_blank(height=128, width=128, channels=3):
    # 生成白色空白图
    if channels == 1:
        return np.ones((height, width), dtype=np.uint8) * 255
    return np.ones((height, width, channels), dtype=np.uint8) * 255


def safe_read_image(path, fallback_shape=None):
    # 若路径无效，返回空白图（形状尽可能与fallback一致）
    try:
        if isinstance(path, str) and os.path.exists(path):
            return mpimg.imread(path)
    except Exception:
        pass

    if fallback_shape is not None:
        if len(fallback_shape) == 2:
            h, w = fallback_shape
            c = 3
        elif len(fallback_shape) == 3:
            h, w, c = fallback_shape
        else:
            h, w, c = 128, 128, 3
        return make_blank(h, w, c)

    return make_blank()


def pad_to_same_height(img, target_height):
    # 将图像在高度方向用白色进行上下填充到目标高度
    h = img.shape[0]
    if h == target_height:
        return img
    pad_total = target_height - h
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    if img.ndim == 3:
        return np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=255)
    else:
        return np.pad(img, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=255)


def hstack_images(img1, img2):
    # 将两张图在高度上对齐后水平拼接
    h = max(img1.shape[0], img2.shape[0])
    img1p = pad_to_same_height(img1, h)
    img2p = pad_to_same_height(img2, h)
    # 若通道数不同（极端情况），统一到3通道
    def to3(img):
        if img.ndim == 2:
            return np.stack([img, img, img], axis=-1)
        if img.shape[-1] == 4:  # RGBA -> RGB
            return (img[..., :3] * 255 if img.dtype == np.float32 else img[..., :3]).astype(np.uint8)
        if img.shape[-1] == 3:
            return img
        return make_blank(img.shape[0], img.shape[1], 3)

    img1p = to3(img1p)
    img2p = to3(img2p)
    return np.concatenate((img1p, img2p), axis=1)

# -----------------------------------
# 2️⃣ 基本参数
# -----------------------------------
methods = [
    "Target (GT) #1",   # 第一行：Target 第1张
    "Target (GT) #2",   # 第二行：Target 第2张
    "No Defense",
    "MID",
    "BiDO-COCO",
    "BiDO-HSIC",
    "LIB",
   "SLIB-GP",

]

# 计算样本数：基于 Target 对（两张一列）与其他行（单张一列）的最大列数
def row_length(method):
    try:
        if method.startswith("Target (GT)"):
            return len(image_paths.get("Target (GT)", []))
        return len(image_paths.get(method.replace(" #1", "").replace(" #2", ""), []))
    except Exception:
        return 0

num_samples = max([row_length(m) for m in methods] + [0])
fig, axes = plt.subplots(len(methods), num_samples, figsize=(12, 8))

# 如果 axes 是一维的（例如只有一行），统一转成二维列表
if len(methods) == 1:
    axes = [axes]

# -----------------------------------
# 3️⃣ 绘制每行图片
# -----------------------------------
for i, method in enumerate(methods):
    for j in range(num_samples):
        ax = axes[i, j]

        # Target 两行分别取一张图：#1 -> 第一个，#2 -> 第二个
        if method.startswith("Target (GT)"):
            pair_list = image_paths.get("Target (GT)", [])
            if j < len(pair_list) and isinstance(pair_list[j], (list, tuple)) and len(pair_list[j]) >= 2:
                p = pair_list[j][0] if "#1" in method else pair_list[j][1]
            else:
                p = None
            img = safe_read_image(p)
            ax.imshow(img)
        else:
            base_key = method
            row_list = image_paths.get(base_key, [])
            path = row_list[j] if j < len(row_list) else None
            img = safe_read_image(path)
            ax.imshow(img)

        ax.axis("off")  # 去掉坐标轴

        # 顶部列编号 (1)…(N)
        if i == 0:
            ax.set_title(f"({j+1})", fontsize=11, weight='bold')

    # 行名与分隔线在布局调整后统一绘制

# 在布局最终确定后绘制：左侧行名与（除第一行外）每行下方虚线



# -----------------------------------
# 4️⃣ 调整布局并保存
# -----------------------------------

plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.06,
                    wspace=0, hspace=0.15)

# 再手动绘制左侧标签和虚线
try:
    import matplotlib.lines as mlines
    fig.canvas.draw()  # 更新坐标以获取准确位置

    num_rows = len(methods)
    # -------- 缩小第1行与第2行之间的缝隙：将第二行整体上移 --------
    try:
        if num_rows >= 2:
            # 以第一列的轴为参考，计算行间距
            b0 = axes[0, 0].get_position()  # 第1行bbox（顶部行）
            b1 = axes[1, 0].get_position()  # 第2行bbox
            gap = b0.y0 - b1.y1  # 行间实际空隙（>0 为间隔）
            if gap > 0:
                reduce_ratio = 0.5  # 将缝隙减少 50%
                shift = gap * reduce_ratio
                for j in range(num_samples):
                    b = axes[1, j].get_position()
                    axes[1, j].set_position([b.x0, b.y0 + shift, b.width, b.height])
            fig.canvas.draw()
    except Exception:
        pass
    first_left = axes[0, 0].get_position().x0
    text_x = max(0.1, first_left * 0.4)  # 左侧文字中心位置

    # -------- Target 两行共用标签 --------
    bbox0 = axes[0, 0].get_position()
    bbox1 = axes[1, 0].get_position() if num_rows > 1 else bbox0
    target_center_y = ((bbox0.y0 + bbox0.y1) + (bbox1.y0 + bbox1.y1)) / 4.0
    fig.text(text_x, target_center_y, "Target", ha='right', va='center',
             fontsize=13, weight='bold')

    # -------- 其他行标签 --------
    row_labels = [None, None, "No Def.", "MID", "BiDO-COCO", "BiDO-HSIC","LIB","SLIB-GP",]
    for row_idx in range(2, num_rows):
        label = row_labels[row_idx] if row_idx < len(row_labels) else None
        if not label:
            continue
        bbox = axes[row_idx, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2.0
        fig.text(text_x, y_center, label, ha='right', va='center',
                 fontsize=13, weight='bold')

    # -------- 行间虚线分隔 --------
    for row_idx in range(1, num_rows - 1):
        left = axes[row_idx, 0].get_position().x0
        right = axes[row_idx, -1].get_position().x1
        bbox_curr = axes[row_idx, 0].get_position()
        bbox_next = axes[row_idx + 1, 0].get_position()
        y = (bbox_curr.y0 + bbox_next.y1) / 2.0
        line = mlines.Line2D([left, right], [y, y], transform=fig.transFigure,
                             linestyle=(0, (5, 5)), color='black', linewidth=2, zorder=10)
        line.set_clip_on(False)
        fig.add_artist(line)

except Exception as e:
    print("Label drawing failed:", e)

# -----------------------------------
# 5️⃣ 保存结果
# -----------------------------------
plt.savefig("comparison_grid_shifted.png", dpi=300, bbox_inches='tight')
plt.show()

