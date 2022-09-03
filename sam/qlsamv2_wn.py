from collections import defaultdict

import torch
import torch.nn as nn

from quantization.lsq_wn_qsamv2_layer import QuantWnConv2d, QuantWnLinear
from sam.utils import sync_grad


class QSAM:
    def __init__(
            self,
            optimizer,
            model,
            rho=0.5,
            include_norm=True,
    ):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.include_norm = include_norm
        self.state = defaultdict(dict)

    @torch.no_grad()
    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.optimizer.param_groups[0]["params"][0].device
        wgrads = []
        for n, m in self.model.named_modules():
            if isinstance(
                    m, (QuantWnConv2d, QuantWnLinear),
            ):
                if m.x.grad is not None:
                    wgrads.append(torch.norm(m.x.grad, p=2).to(shared_device))

                if (
                        hasattr(m, "bias")
                        and m.bias is not None
                        and m.bias.grad is not None
                ):
                    wgrads.append(torch.norm(m.bias.grad, p=2).to(shared_device))
            if self.include_norm:
                if isinstance(m, nn.LayerNorm):
                    if m.weight.grad is None:
                        continue
                    wgrads.append(torch.norm(m.weight.grad, p=2).to(shared_device))
                    wgrads.append(torch.norm(m.bias.grad, p=2).to(shared_device))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2)
        return wgrad_norm

    @torch.no_grad()
    def ascent_step(self):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for n, m in self.model.named_modules():
            if isinstance(
                    m, (QuantWnConv2d, QuantWnLinear),
            ):
                if m.x.grad is not None:
                    p = m.x
                    self.state[m]["old_p"] = p.data.clone()
                    e_w = p.grad * scale.to(p)
                    m.epsilon = e_w
                else:
                    m.epsilon = m.x.new_zeros(1)

                if (
                        hasattr(m, "bias")
                        and m.bias is not None
                        and m.bias.grad is not None
                ):
                    p = m.bias
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)
            if self.include_norm:
                if isinstance(m, nn.LayerNorm):
                    if m.weight.grad is None:
                        continue
                    p = m.weight
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)

                    p = m.bias
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)

    @torch.no_grad()
    def descent_step(self):
        for n, m in self.model.named_modules():
            if isinstance(
                    m, (QuantWnConv2d, QuantWnLinear),
            ):
                if (
                        hasattr(m, "bias")
                        and m.bias is not None
                        and m.bias.grad is not None
                ):
                    p = m.bias
                    p.data = self.state[p]["old_p"]
            if self.include_norm:
                if isinstance(m, nn.LayerNorm):
                    if m.weight.grad is None:
                        continue
                    p = m.weight
                    p.data = self.state[p]["old_p"]

                    p = m.bias
                    p.data = self.state[p]["old_p"]
        sync_grad(self.optimizer)
        self.optimizer.step()
        self.optimizer.zero_grad()
