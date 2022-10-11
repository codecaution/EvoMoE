from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast
import torch
import torch.distributed as dist
from torch import Tensor

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


class _DecoupledAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))

class _FlexibleAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, input_splits: Tensor, output_splits: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        ctx.rank = dist.get_rank()
        input = input.contiguous()
        ctx.input_dim0 = input.size(0)
        ctx.output_dim0 = output_splits.sum(dim=0).item()
        ctx.input_splits = input_splits.tolist()
        ctx.output_splits = output_splits.tolist()
        ctx.model_dim = input.size(1)
        ctx.device = input.device
        output = torch.empty(size = (ctx.output_dim0, ctx.model_dim), device = ctx.device)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, output_split_sizes = ctx.output_splits,\
                                            input_split_sizes = ctx.input_splits, group=group)
        else:
            assert group is None
            output = input
        # print("Rank {}, {}, {}".format(dist.get_rank(), output.size(), output))
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        output = torch.empty(size = (ctx.input_dim0, ctx.model_dim), device = ctx.device)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, *grad_output, output_split_sizes = ctx.input_splits,\
                                            input_split_sizes = ctx.output_splits, group=ctx.group)
        else:
            assert ctx.group is None
            output = input
        return None, output, None, None, 
