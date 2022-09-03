wbits=2
abits=2
lr=2e-4
epochs=100
id=2bit_uniform

for rho in 0.05
do
for j in 0
do
python -m torch.distributed.launch \
--nproc_per_node=2 --use_env main_sam.py \
--model deit_wn_tiny_patch16_224_mix \
--batch-size 64 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir results/deit_wn_tiny_${id}_cifar/${wbits}w${abits}a_bs512_baselr${lr}_ft${epochs}_QLSAM_rho${rho}_${j} \
--finetune /scratch/dl65/jing/Codes/Q-ViT-DeiT/results/deit_tiny_float_cifar/ckpt/best_checkpoint.pth \
--data-set CIFAR \
--data-path /home/jliu/dl65/liujing/dataset/cifar \
--seed ${j} \
--rho ${rho} \
--sam_type "QLSAM" \
--resume /scratch/dl65/jing/Codes/Q-ViT-DeiT/results/deit_wn_tiny_2bit_uniform_cifar/2w2a_bs512_baselr2e-4_ft100_QLSAM_rho0.05_0/ckpt/current_checkpoint.pth
done
done