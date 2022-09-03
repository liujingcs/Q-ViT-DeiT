python main_sam.py \
--model deit_wn_qsamv2_tiny_patch16_224_mix \
--batch-size 64 \
--lr 2e-4 \
--min-lr 0 \
--epochs 100 \
--warmup-epochs 0 \
--wbits 2 \
--abits 2 \
--dist-eval \
--output_dir results/deit_wn_tiny_2bit_uniform_cifar/2w2a_bs512_baselr2e-4_ft100_QLSAMv2_rho0.01_0 \
--finetune /home/liujing/Codes/transformer/Q-ViT-DeiT/results/deit_tiny_float_cifar/ckpt/best_checkpoint.pth \
--data-set CIFAR \
--data-path /home/liujing/Datasets/cifar100 \
--seed 0 \
--rho 0.01 \
--sam_type "QLSAMv2" \
--num-workers 0