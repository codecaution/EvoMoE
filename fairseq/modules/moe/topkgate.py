# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

from typing import Callable, Dict, Tuple, Optional, Any

import math
import torch
from torch import Tensor
import torch.nn.functional as F
from fairseq import parameter
from .top1gate import top1gating
from .top2gate import entropy, one_hot


# maximum capacity of 1 expert as a fraction of number of tokens in the batch
# Note: setting this to 1.0 causes inference to significantly slow down
EVAL_CAPACITY_TOKEN_FRACTION = 0.25

# logging
SAMPLE_FRACTION = 0.2
GUMBEL_THRESHOLD = 0.001

def topkgating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    topk=1,
    use_fp32=False,
    capacity_factor=1.0,
    eval_mode=False,
    moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements TopkGating on logits."""
    
    if topk != logits.shape[1]:
        assert False, "Only support top-1, top-2 and top-N (all) now!"

    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()
    # gates has shape of SE
    num_tokens = logits.shape[0]
    num_experts = logits.shape[1]
    # Compute l_aux
    l_aux = None
    if parameter.gumbel_temperature > 0:
        if parameter.soft_gumbel_training == True:
            gumbel_gates = F.gumbel_softmax(logits, tau=parameter.gumbel_temperature, hard=False)
            experts_choosed = torch.gt(gumbel_gates, GUMBEL_THRESHOLD)
        else:
            gumbel_gates = F.gumbel_softmax(logits, tau=parameter.gumbel_temperature, hard=False)
            indices1_s = torch.argmax(gumbel_gates, dim=1)
            experts_choosed = one_hot(indices1_s, num_classes=num_experts, unsqueeze_indices=True)
        #############Switch between different gates weights#######
        if parameter.use_gumbel_gates:
            gates = experts_choosed * gumbel_gates
        else:
            # use softmax gates
            gates = experts_choosed * F.softmax(logits, dim=1)
        ##########################################################
        # Compute l_aux
        me = torch.mean(gates, dim=0)
        ce = torch.mean(experts_choosed.to(gates.dtype), dim=0)
        l_aux = torch.mean(me * ce)
        l_aux = l_aux * num_experts * num_experts
        ##########################################################
        metadata["gumbel_choosed_experts"] = experts_choosed.sum()
    else:
        gates = F.softmax(logits, dim=1) #(num_tokens, num_experts)
        l_aux = torch.zeros(size=(1,)).to(gates.device)
    metadata["gate_weights"] = gates
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()
    capacity = int(num_tokens)
    # S, E, C
    # combine_sec = torch.zeros((num_tokens, num_experts, capacity), dtype=torch.float32, device=gates.device)
    # for i in range(num_tokens):
    #     for j in range(num_experts):
    #         combine_sec[i][j][i] = gates[i][j]
    
    # ------------------------------------------- 
    # !!!! Need to be optimized !!!
    combine_sec = torch.diag_embed(gates.permute(1,0)).permute(1, 0, 2).contiguous()
    dispatch_mask = combine_sec.bool()
    # ------------------------------------------- 

    if use_fp32:
        return l_aux, combine_sec.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine_sec, dispatch_mask, metadata


class TopKGate(torch.nn.Module):
    """Gate module which implements TopNGating (same as weighted sum).
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        topk=-1,
        use_fp32=False,
        input_noise_type=None,
        capacity_factor=1.0,
        moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
    ) -> None:
        # TODO: merge this to top2gate.py
        #
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.use_fp32 = use_fp32
        if topk == -1:
            self.topk = num_experts
        else:
            self.topk = topk

        self.input_noise_type = input_noise_type
        self.capacity_factor = capacity_factor
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None,) -> Tuple[Tensor, Tensor, Tensor, Dict]:  # type: ignore
        logits = self.wg(input)
        if self.training or parameter.dense_validate == True:
            return topkgating(
                logits,
                mask,
                self.topk,
                use_fp32=self.use_fp32,
                capacity_factor=self.capacity_factor,
                eval_mode=not self.training,
                moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            )
        else:
            # logits.shape[1]: num experts
            return top1gating(
                logits,
                mask,
                use_fp32=self.use_fp32,
                eval_mode=not self.training,
                capacity_factor=logits.shape[1],
                moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            )
