python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_tiny_patch16_224_float \
--batch-size 128 \
--dist-eval \
--epochs 300 \
--output_dir results/deit_tiny_float_bs128 \
--data-path /home/jliu/dl65/m3_imagenet