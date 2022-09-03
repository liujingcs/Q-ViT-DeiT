# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
from numpy.lib.shape_base import hsplit
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, model
import torch.nn as nn

from datasets import build_dataset
from engine import initialize_quantization, train_one_epoch_tb_sam, evaluate, train_one_epoch_tb, initialize_muitihead_quantization, head_analysis
from losses import DistillationLoss
from samplers import RASampler
import models
from quantization import quantvit, quant_vit_mixpre, quant_wn_vit_mixpre, quant_wn_qsamv2_vit_mixpre, quant_wn_sam_vit_mixpre
import utils
from params import args
# from torch.utils.tensorboard import SummaryWriter
from logger import logger
from sam.optim import get_minimizer

def main(args):
    utils.init_distributed_mode(args)
    if utils.get_rank() != 0:
        logger.disabled = True

    logger.info(args)

    args.act_layer = nn.ReLU if args.act_layer == 'relu' else nn.GELU

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            logger.info("Using dist eval.")
            if len(dataset_val) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    data_loader_sampler = torch.utils.data.DataLoader(
        dataset_val, sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=64,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size= 2 * args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        wbits=args.wbits,
        abits=args.abits,
        act_layer=args.act_layer,
        offset=args.use_offset,
        learned=not args.fixed_scale,
        mixpre=args.mixpre,
        headwise=args.head_wise
    )
    logger.info(model)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    

    
    output_dir = Path(args.output_dir)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # Initialize scales for quantization

    # if args.head_wise:
    #     initialize_muitihead_quantization(model, device)

    if args.resume == '' and (args.abits > 0 or args.wbits > 0) and utils.is_main_process():
        initialize_quantization(data_loader_sampler, model, device, output_dir, sample_iters=1)

    

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    # get minimizer
    minimizer = get_minimizer(model, optimizer, args)
    logger.info(minimizer)
    logger.info("Include norm: {}".format(minimizer.include_norm))

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        logger.info(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )


    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint: 
            # if not args.head_wise:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.save_ptq and args.output_dir:
        checkpoint_paths = [output_dir / 'ckpt' / f'{args.wbits}w{args.abits}a_ptq_checkpoint.pth']
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                # 'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)
    
    # if args.head_wise and args.head_analysis:
    #     test_stats = evaluate(data_loader_val, model, device)
    #     logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #     log_stats = {'head_index': 'None (uniform)',
    #     **{f'test_{k}': v for k, v in test_stats.items()}}
    #     if args.output_dir and utils.is_main_process():
    #             with (output_dir / "head_analysis.txt").open("a") as f:
    #                 f.write(json.dumps(log_stats) + "\n")
    #     for head_index in range(6):
    #         head_analysis(model_without_ddp, head_index)
    #         test_stats = evaluate(data_loader_val, model, device)
    #         logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #         log_stats = {'head_index': head_index,
    #         **{f'test_{k}': v for k, v in test_stats.items()}}

    #         if args.output_dir and utils.is_main_process():
    #             with (output_dir / "head_analysis.txt").open("a") as f:
    #                 f.write(json.dumps(log_stats) + "\n")
    #     return


    if args.show_bit_state:
        for name, m in model_without_ddp.named_modules():
            if hasattr(m, 'nbits'):
                nbits_float = m.nbits.item()
                nbits = m.nbits.round().clamp(2,8).item()
                logger.info(f"{name} is {nbits}({nbits_float})-bit\n")
        

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy5 = 0.0

    writer = None
    # if utils.is_main_process():
        
    #     writer = SummaryWriter(comment=args.comment)

    test_stats = evaluate(data_loader_val, model, device)
    logger.info(f"Accuracy of the network on the {50000} test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats["acc1"])
    max_accuracy5 = max(max_accuracy5, test_stats["acc5"])
    logger.info(f'Max accuracy: {max_accuracy:.2f}%')
    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # freeze bitwidth updating for the post 50% training stage
        if epoch == int(args.stage_ratio * args.epochs):
            for m in model.modules():
                if hasattr(m, 'nbits'):
                    m.nbits.requires_grad = False
                    
        train_stats = train_one_epoch_tb_sam(
            model, criterion, data_loader_train,
            optimizer, minimizer, device, epoch,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '',
            writer=writer,
            total_epochs = args.epochs,
            output_dir = output_dir,
            sam_type=args.sam_type
        )

        lr_scheduler.step(epoch)
        if args.output_dir and epoch % 20 == 0:
            checkpoint_paths = [output_dir / 'ckpt' / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    # 'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'ckpt' / f'current_checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    # 'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)
        is_best = max_accuracy < test_stats["acc1"]
        if is_best:
            checkpoint_paths = [output_dir / 'ckpt' / f'best_checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    # 'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        max_accuracy5 = test_stats["acc5"] if is_best else max_accuracy5
        logger.info(f'Max accuracy: {max_accuracy:.2f}%, Max accuracy5: {max_accuracy5:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            with (output_dir / "bit.txt").open("w") as f:
                for name, m in model_without_ddp.named_modules():
                    if hasattr(m, 'nbits'):
                        nbits_float = m.nbits.data
                        nbits = m.nbits.round().clamp(2,8).data
                        f.write(f"{name} is {nbits}({nbits_float})-bit\n")

    # if utils.is_main_process():
    #     writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    if args.output_dir:
        Path(args.output_dir +'/ckpt').mkdir(parents=True, exist_ok=True)
    main(args)

def hook(module, grad_input, grad_output):
    if hasattr(module, 'nbits'):
        module.nbits.grad = None