import os
import random
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="从 MNIST Img 目录中采样一部分图片到 FID 真实图像目录")
    parser.add_argument(
        "--src_dir",
        default="/data2/jys/.virtualenvs/Defend_MI-master/attack_dataset/MNIST/Img",
        help="MNIST 原始图片目录"
    )
    parser.add_argument(
        "--dst_dir",
        default="/data2/jys/.virtualenvs/Defend_MI-master/DMI/attack_res/mnist/Trainset",
        help="采样后保存目录（FID 真实图像使用的目录）"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,   # 建议值：10000 张，FID 比较稳定
        help="要采样的图片数量（如果多于总数则自动截断为总数）"
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="要匹配的图片扩展名"
    )
    args = parser.parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    exts = tuple([e.lower() for e in args.exts])

    if not os.path.isdir(src_dir):
        raise RuntimeError(f"源目录不存在: {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    # 收集所有图片文件
    all_files = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(exts):
                all_files.append(os.path.join(root, f))

    if not all_files:
        raise RuntimeError(f"在 {src_dir} 下没有找到任何扩展名为 {exts} 的图片")

    num_samples = min(args.num_samples, len(all_files))
    print(f"在 {src_dir} 中共找到 {len(all_files)} 张图片，准备随机采样 {num_samples} 张。")

    sampled_files = random.sample(all_files, num_samples)

    for i, src_path in enumerate(sampled_files, 1):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, filename)
        # 如有重名文件，简单加编号后缀
        base, ext = os.path.splitext(filename)
        cnt = 1
        while os.path.exists(dst_path):
            dst_path = os.path.join(dst_dir, f"{base}_{cnt}{ext}")
            cnt += 1
        shutil.copy2(src_path, dst_path)
        if i % 1000 == 0 or i == num_samples:
            print(f"已复制 {i}/{num_samples} 张")

    print(f"完成！采样图片已保存到: {dst_dir}")

if __name__ == "__main__":
    main()