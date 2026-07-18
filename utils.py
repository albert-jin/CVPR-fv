"""Utility functions for CVPR-FV.

Contents
--------

* ``LoRALinear`` — a low-rank additive linear adapter that supports
  either pure LoRA (rank > 0, scaling_rank = 0) or T-Few (IA)^3
  (rank = 0, scaling_rank = 1). The same class is used regardless of
  the backbone (T0-3B seq2seq or Qwen / Llama causal).
* ``modify_with_lora`` — walks the transformer and swaps every
  ``nn.Linear`` that matches ``modules_re`` + ``layers_re`` with a
  ``LoRALinear`` wrapper.
* ``get_lora_regex_for_backbone`` — returns the pair of regexes
  appropriate for the requested backbone kind.
* Small helpers (``set_seeds``, ``count_trainable_params``,
  ``build_exp_dir``).
"""

import os
import re
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import configs


class LoRALinear(nn.Module):
    """LoRA / (IA)^3 wrapper for ``nn.Linear``."""

    def __init__(self, linear_layer: nn.Linear, rank: int, scaling_rank: int,
                 init_scale: float, dropout: float = 0.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.scaling_rank = scaling_rank
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
            # (IA)^3 path
            if self.multi_lora_a.requires_grad:
                hidden = F.linear(input_ * self.multi_lora_a.flatten().contiguous(),
                                  self.weight, self.bias)
            else:
                hidden = F.linear(input_, self.weight, self.bias)
            if self.multi_lora_b.requires_grad:
                hidden = hidden * self.multi_lora_b.flatten().contiguous()
            return hidden
        # Pure LoRA path with additive rank-r update.
        base = F.linear(input_, self.weight, self.bias)
        if self.rank:
            update = F.linear(F.linear(self.dropout(input_), self.lora_a), self.lora_b) / self.rank
            base = base + update
        if self.scaling_rank:
            base = base * (self.multi_lora_b.flatten().contiguous())
        return base

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, rank={self.rank}, scaling_rank={self.scaling_rank}"
        )


def get_lora_regex_for_backbone(kind: str):
    """Return (modules_re, layers_re) suited to the backbone kind."""
    if kind == 'seq2seq':
        return configs.lora_modules_seq2seq, configs.lora_layers_seq2seq
    return configs.lora_modules_causal, configs.lora_layers_causal


def modify_with_lora(transformer, kind: str = 'seq2seq',
                     rank: int = None, scaling_rank: int = None,
                     init_scale: float = None, dropout: float = None,
                     modules_re: str = None, layers_re: str = None):
    """Replace matching ``nn.Linear`` modules with ``LoRALinear`` wrappers.

    The default kwargs fall back to values in ``configs``.
    """
    rank = configs.lora_rank if rank is None else rank
    scaling_rank = configs.lora_scaling_rank if scaling_rank is None else scaling_rank
    init_scale = configs.lora_init_scale if init_scale is None else init_scale
    dropout = configs.lora_dropout if dropout is None else dropout
    if modules_re is None or layers_re is None:
        m_default, l_default = get_lora_regex_for_backbone(kind)
        modules_re = modules_re or m_default
        layers_re = layers_re or l_default

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(modules_re, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(layers_re, c_name):
                    if not isinstance(layer, nn.Linear):
                        continue
                    setattr(module, c_name,
                            LoRALinear(layer, rank, scaling_rank, init_scale, dropout))
    return transformer


def set_seeds(seed: int):
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
