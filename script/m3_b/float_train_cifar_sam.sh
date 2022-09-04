python -m torch.distributed.launch \
--nproc_per_node=2 --use_env main_sam.py \
--model deit_tiny_patch16_224_float \
--batch-size 64 \
--dist-eval \
--epochs 300 \
--output_dir results/deit_tiny_float_cifar_sam_rho0.1 \
--data-set CIFAR \
--data-path /home/bzhuang/dl65/bzhuang/cifar100 \
--rho 0.1 \
--sam_type "SAM"