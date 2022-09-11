import os

src = "/home/bzhuang/dl65/liujing/dataset/imagenet"

dataset_src = "/home/bzhuang/dl65/m3_imagenet"
dataset_dst = "/home/bzhuang/dl65_scratch/bzhuang/jing/dataset/imagenet"

for folder in ["train100", "val100"]:
    liujing_file_src = os.path.join(src, folder)
    ls_dir = os.listdir(liujing_file_src)
    dataset_src_dir = os.path.join(dataset_src, "train" if "train" in folder else "val")
    dataset_dst_dir = os.path.join(dataset_dst, folder)
    for dir in ls_dir:
        dataset_src_path = os.path.join(dataset_src_dir, dir)
        dataset_dst_path = os.path.join(dataset_dst_dir, dir)
        os.symlink(dataset_src_path, dataset_dst_path)