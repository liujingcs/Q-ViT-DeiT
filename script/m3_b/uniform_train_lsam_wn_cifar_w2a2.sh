wbits=2
abits=2
lr=2e-4
epochs=100
id=2bit_uniform

for rho in 0.01 0.02 0.05 0.1
do
for j in 0
do
python -m torch.distributed.launch \
--nproc_per_node=2 --use_env main_sam.py \
--model deit_wn_sam_tiny_patch16_224_mix \
--batch-size 64 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir results/deit_wn_tiny_${id}_cifar/${wbits}w${abits}a_bs512_baselr${lr}_ft${epochs}_LSAM_rho${rho}_seed${j} \
--finetune /scratch/dl65/bzhuang/jing/Q-ViT-DeiT/results/deit_tiny_float_cifar/ckpt/best_checkpoint.pth \
--data-set CIFAR \
--data-path /home/bzhuang/dl65/bzhuang/cifar100 \
--seed ${j} \
--rho ${rho} \
--sam_type "LSAM_wn"
done
done