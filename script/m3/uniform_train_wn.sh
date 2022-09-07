wbits=4
abits=4
lr=5e-4
epochs=300
id=4bit_uniform

python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_wn_tiny_patch16_224_mix \
--batch-size 64 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir results/deit_wn_tiny_${id}/${wbits}w${abits}a_bs512_baselr${lr}_ft${epochs} \
--finetune /scratch/dl65/jing/Codes/Q-ViT-DeiT/results/deit_tiny_float/ckpt/best_checkpoint.pth \
--data-path /home/jliu/dl65/m3_imagenet \
--resume /fs03/dl65/jing/Codes/Q-ViT-DeiT/results/deit_wn_tiny_4bit_uniform/4w4a_bs512_baselr5e-4_ft300/ckpt/current_checkpoint.pth