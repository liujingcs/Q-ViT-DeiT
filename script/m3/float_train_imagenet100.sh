python -m torch.distributed.launch \
--nproc_per_node=2 --use_env main.py \
--model deit_tiny_patch16_224_float \
--batch-size 64 \
--dist-eval \
--epochs 300 \
--output_dir results/deit_tiny_float_imagenet100 \
--data-set IMNET100 \
--data-path /home/jliu/dl65/liujing/dataset/imagenet