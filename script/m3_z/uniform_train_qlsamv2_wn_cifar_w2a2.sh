wbits=2
abits=2
lr=2e-4
epochs=100
id=2bit_uniform

for rho in 0.01 0.007 0.005 0.003 0.001
do
for j in 0
do
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--master_port 62255 \
--nproc_per_node=2 --use_env main_sam.py \
--model deit_wn_tiny_patch16_224_mix \
--batch-size 128 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir results/deit_wn_tiny_${id}_cifar_bs128/${wbits}w${abits}a_bs512_baselr${lr}_ft${epochs}_QLSAMv2_rho${rho}_${j} \
--finetune /scratch/dl65/pzz/jing/Codes/iclr2022/Q-ViT-DeiT/results/deit_tiny_float_cifar_bs128/ckpt/best_checkpoint.pth \
--data-set CIFAR \
--data-path /home/zpan/dl65/zpan/cifar100 \
--seed ${j} \
--rho ${rho} \
--sam_type "QLSAMv2"
done
done