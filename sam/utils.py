import contextlib

import torch
from torch.distributed import ReduceOp


@torch.no_grad()
def sync_grad(optimizer):
    if torch.distributed.is_initialized():  # synchronize final gardients
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                torch.distributed.all_reduce(p.grad, op=ReduceOp.SUM)
                world_size = torch.distributed.get_world_size()
                p.grad.div_(float(world_size))


def maybe_no_sync(model):
    if torch.distributed.is_initialized():
        return model.no_sync()
    else:
        return contextlib.ExitStack()
