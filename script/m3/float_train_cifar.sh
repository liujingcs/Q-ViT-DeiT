python -m torch.distributed.launch \
--nproc_per_node=2 --use_env main.py \
--model deit_tiny_patch16_224_float \
--batch-size 64 \
--dist-eval \
--epochs 300 \
--output_dir results/deit_tiny_float_cifar \
--data-set CIFAR \
--data-path /home/jliu/dl65/liujing/dataset/cifar \
--resume /scratch/dl65/jing/Codes/Q-ViT-DeiT/results/deit_tiny_float_cifar/ckpt/checkpoint_280.pth