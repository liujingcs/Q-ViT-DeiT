wbits=2
abits=2
lr=2e-4
epochs=100
id=2bit_uniform

for j in 0
do
python -m torch.distributed.launch \
--nproc_per_node=2 --use_env main.py \
--model deit_wn_tiny_patch16_224_mix \
--batch-size 128 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir results/deit_wn_tiny_${id}_cifar_bs128/${wbits}w${abits}a_bs512_baselr${lr}_ft${epochs}_${j} \
--finetune /scratch/dl65/pzz/jing/Codes/iclr2022/Q-ViT-DeiT/results/deit_tiny_float_cifar_bs128/ckpt/best_checkpoint.pth \
--data-set CIFAR \
--data-path /home/zpan/dl65/zpan/cifar100 \
--seed ${j}
done