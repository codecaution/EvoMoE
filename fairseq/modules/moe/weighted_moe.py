# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast
import os
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

class WeightedMOELayer(Base):
    """MOELayer module which distributed the tokens to all-experts(top-N). 
       Then do weighted sum to combine the output of each experts.
    ::
        gate = TopNGate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, experts: Union[Module, ModuleList], args: Optional[Any] = None, group: Optional[Any] = None) -> None:
        super().__init__()
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])

        self.expert_number = len(self.experts)
        self.model_dim = args.decoder_embed_dim
        self.args = args
        self.in_generation = False
        self.wg = torch.nn.Linear(self.model_dim, self.expert_number, bias=False)

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        # shape of input: [Batch_size, Seq_len, d_model]
        input = input[0]
        d_model = input.shape[2]
        
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape

        # reshapeed_input: [tokens_number, d_model]
        logits = self.wg(reshaped_input)
        
        # gates_weight: [tokens_number, expert_number]
        if self.args.use_gumbel_softmax:
            gates_weight = F.softmax(logits/self.args.gumbel_temperature, dim=1)
        else:
            gates_weight = F.softmax(logits, dim=1)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs += [expert(reshaped_input)]     
        
        expert_outputs = torch.cat(expert_outputs, 1).reshape(reshaped_input_shape[0], self.expert_number, reshaped_input_shape[1])
        gates_weight = gates_weight.reshape(reshaped_input_shape[0], 1, self.expert_number)
        combined_output = torch.bmm(gates_weight, expert_outputs)

        combined_output = combined_output.reshape(input.shape)
        
        return combined_output, None