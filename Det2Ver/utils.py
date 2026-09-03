"""Utility functions used by Det2Ver.

This module contains

* the LoRA / (IA)^3 style linear adapter (``LoRALinear``) that replaces every
  attention / MLP linear layer selected by ``configs.lora_modules`` and
  ``configs.lora_layers``;
* ``modify_with_lora``, which walks the T0-3B transformer and swaps in the
  adapter modules in-place;
* ``set_seeds`` and ``build_exp_dir`` helpers.

The LoRA design follows the T-Few / ProToCo convention:

* ``rank == 0`` and ``scaling_rank == 1`` gives (IA)^3, i.e. two learnable
  1-d rescaling vectors (``multi_lora_a``, ``multi_lora_b``) applied to the
  input / output of the frozen linear layer.
* ``rank > 0`` and ``scaling_rank == 0`` gives standard LoRA with a low-rank
  additive update ``B @ A / rank`` on the weight.
"""

import os
import re
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import (
    lora_modules, lora_layers, lora_rank, lora_scaling_rank, lora_init_scale,
)


class LoRALinear(nn.Module):
    """LoRA / (IA)^3 wrapper for ``nn.Linear``."""

    def __init__(self, linear_layer: nn.Linear, rank: int, scaling_rank: int, init_scale: float):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)
            if init_scale < 0:
                self.lora_b = nn.Parameter(torch.randn(linear_layer.out_features, rank) * init_scale)
            else:
                self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale
            )
            if init_scale < 0:
                self.multi_lora_b = nn.Parameter(
                    torch.ones(linear_layer.out_features, self.scaling_rank)
                    + torch.randn(linear_layer.out_features, self.scaling_rank) * init_scale
                )
            else:
                self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features, self.scaling_rank))

    def forward(self, input_):
        if self.scaling_rank == 1 and self.rank == 0:
            # parsimonious implementation for (IA)^3 style scaling
            if self.multi_lora_a.requires_grad:
                hidden = F.linear(input_ * self.multi_lora_a.flatten().contiguous(), self.weight, self.bias)
            else:
                hidden = F.linear(input_, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                hidden = hidden * self.multi_lora_b.flatten().contiguous()
            return hidden
        # general LoRA path (scaling and/or additive)
        weight = self.weight
        if self.scaling_rank:
            weight = weight * torch.matmul(self.multi_lora_b, self.multi_lora_a) / self.scaling_rank
        if self.rank:
            weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
        return F.linear(input_, weight, self.bias)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, rank={self.rank}, scaling_rank={self.scaling_rank}"
        )


def modify_with_lora(transformer, rank=None, scaling_rank=None, init_scale=None,
                     modules_re=None, layers_re=None):
    """Replace every ``nn.Linear`` that lives under a module matching
    ``modules_re`` and whose child name matches ``layers_re`` with a
    ``LoRALinear`` wrapper. The original weight is preserved and only the
    adapter parameters are trainable.
    """
    if rank is None:
        rank = lora_rank
    if scaling_rank is None:
        scaling_rank = lora_scaling_rank
    if init_scale is None:
        init_scale = lora_init_scale
    if modules_re is None:
        modules_re = lora_modules
    if layers_re is None:
        layers_re = lora_layers

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(modules_re, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(layers_re, c_name):
                    assert isinstance(layer, nn.Linear), (
                        f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    )
                    setattr(module, c_name, LoRALinear(layer, rank, scaling_rank, init_scale))
    return transformer


def set_seeds(seed: int):
    """Reproducibility helper."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_exp_dir(exp_root: str, exp_name: str) -> str:
    exp_dir = os.path.join(exp_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
