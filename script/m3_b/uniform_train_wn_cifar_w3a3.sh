wbits=3
abits=3
lr=2e-4
epochs=100
id=3bit_uniform

for j in 1 2 3 4
do
python -m torch.distributed.launch \
--nproc_per_node=2 --use_env main.py \
--model deit_wn_tiny_patch16_224_mix \
--batch-size 64 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir results/deit_wn_tiny_${id}_cifar/${wbits}w${abits}a_bs512_baselr${lr}_ft${epochs}_seed${j} \
--finetune /scratch/dl65/bzhuang/jing/Q-ViT-DeiT/results/deit_tiny_float_cifar/ckpt/best_checkpoint.pth \
--data-set CIFAR \
--data-path /home/bzhuang/dl65/bzhuang/cifar100 \
--seed ${j}
done