wbits=4
abits=4
lr=2e-4
epochs=100
id=4bit_uniform

python -m torch.distributed.launch \
--nproc_per_node=2 --use_env main.py \
--model deit_tiny_patch16_224_mix \
--batch-size 64 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir results/deit_tiny_${id}_cifar/${wbits}w${abits}a_bs512_baselr${lr}_ft${epochs} \
--finetune /scratch/dl65/jing/Codes/Q-ViT-DeiT/results/deit_tiny_float_cifar/ckpt/best_checkpoint.pth \
--data-set CIFAR \
--data-path /home/jliu/dl65/liujing/dataset/cifar