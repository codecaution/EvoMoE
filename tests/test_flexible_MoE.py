import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
from fairseq import distributed_utils as dist_utils, utils
from fairseq.distributed import utils as fsdp_wrap
from fairseq.modules.fmoe import FlexibleMOELayer
from fairseq.modules.fmoe import Top2Gate
from fairseq.modules.fmoe import Top1Gate
from fairseq.modules import (
    TransformerDecoderLayer,
)
from fairseq.modules.transformer_layer import FeedForwardNetwork
from fairseq.models.transformer import fsdp_wrap_expert
import functools, math

# step 1. imbalanced all2all
# step 2. expert allreduce  
# step 3. model migaration(expert transferring)

def build_decoder_layer(self, args, no_encoder_attn=False, is_moe_layer=False):
    layer = TransformerDecoderLayer(
        args, no_encoder_attn=no_encoder_attn, is_moe_layer=is_moe_layer
    )
    layer = fsdp_wrap_expert(args, layer, min_num_params=0)
    return layer

def make_experts(args, embed_dim, expert_ffn_dim, dropout_module) -> nn.ModuleList:
    world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    expert_list = []
    ddp_rank = dist_utils.get_data_parallel_rank()
    start_seed = torch.randint(1000000, (1,)).item()
    # at least as many experts than gpus
    if args.moe_expert_count >= world_size:
        assert args.moe_expert_count % world_size == 0, f'{args.moe_expert_count}, {world_size}'
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with utils.set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_list.append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
    # less experts than gpus
    else:
        assert world_size % args.moe_expert_count == 0, f'{world_size}, {args.moe_expert_count}'
        # initialize each FFN with the same seed on different GPUs
        with utils.set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
            expert_list.append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
    experts = nn.ModuleList(expert_list)
    return experts

# Read here
def fsdp_wrap_expert(args, layer, min_num_params=0):
    # Wrap MoE layer with FSDP using a process group with all replicated ranks
    process_group = layer.moe_layer.expert_group
    world_size = dist_utils.get_data_parallel_group().size()
    pg_size = process_group.size()
    num_experts = world_size/pg_size

    for i, expert in enumerate(layer.moe_layer.experts):
        layer.moe_layer.experts[i] = fsdp_wrap(
            expert, process_group=process_group, min_num_params=0
        )
    divide_choice = getattr(args, "moe_normalize_expert_grad", "none")
    if divide_choice == "sqrt_world_size":
        expert_normalization_term = math.sqrt(num_experts)
    elif divide_choice == "world_size":
        expert_normalization_term = num_experts
    else:
        expert_normalization_term = 1.0

    for p in layer.moe_layer.experts.parameters():
        p.expert = True
        # Scale grads by world_size/pg_size so that grads match the equivalent replicated
        # world size expected within Trainer
        if divide_choice != "none":
            p.register_hook(functools.partial(div_by_world_size, expert_normalization_term))

    # Everything else gets wrapped as normal.
    layer = fsdp_wrap(layer, min_num_params=min_num_params)
    return layer

def div_by_world_size(world_size, tensor):
    return tensor / world_size


def testfmoe_for_accuracy(args):
    torch.cuda.set_device(args.local_rank)
    input_shape = [args.batch_size, args.sequence_length, args.decoder_embed_dim]
    input = torch.randn(input_shape)
    input = input.cuda()
    print("Rank: {}, input size:{}".format(args.local_rank, input.size()))
    # *****************************************************************************
    # gate = Top2Gate(
    #     args.decoder_embed_dim,
    #     args.moe_expert_count,
    #     args.moe_gating_use_fp32,
    #     args.moe_second_expert_policy,
    #     args.moe_normalize_gate_prob_before_dropping,
    #     getattr(args, "moe_eval_capacity_token_fraction", 0.25),
    #     getattr(args, "moe_batch_prioritized_routing", False),
    # )
    gate = Top1Gate(
        args.decoder_embed_dim,
        args.moe_expert_count,
        args.moe_gating_use_fp32,
        getattr(args, "moe_eval_capacity_token_fraction", 0.25),
        getattr(args, "moe_batch_prioritized_routing", False),
    )
    # gate.to(args.local_rank)
    experts = make_experts(args, args.decoder_embed_dim, args.decoder_ffn_dim, None)
    flexibleMoeLayer = FlexibleMOELayer(gate, experts, args)
    # *****************************************************************************
    flexibleMoeLayer.cuda()

    output, l_aux = flexibleMoeLayer(input)
    # print(output)
    print("Rank: {}, Output size:{}".format(args.local_rank, output.size()))
    a = output.mean() + l_aux.mean()
    a.backward()
    print("Rank: {}, Success!".format(args.local_rank))


def testfmoe(args):
    torch.cuda.set_device(args.local_rank)
    input_shape = [args.batch_size, args.sequence_length, args.decoder_embed_dim]
    input = torch.randn(input_shape)
    input = input.cuda()
    print("Rank: {}, input size:{}".format(args.local_rank, input.size()))
    # *****************************************************************************
    gate = Top2Gate(
        args.decoder_embed_dim,
        args.moe_expert_count,
        args.moe_gating_use_fp32,
        args.moe_second_expert_policy,
        args.moe_normalize_gate_prob_before_dropping,
        getattr(args, "moe_eval_capacity_token_fraction", 0.25),
        getattr(args, "moe_batch_prioritized_routing", False),
    )
    # gate = Top1Gate(
    #     args.decoder_embed_dim,
    #     args.moe_expert_count,
    #     args.moe_gating_use_fp32,
    #     getattr(args, "moe_eval_capacity_token_fraction", 0.25),
    #     getattr(args, "moe_batch_prioritized_routing", False),
    # )
    # gate.to(args.local_rank)
    experts = make_experts(args, args.decoder_embed_dim, args.decoder_ffn_dim, None)
    flexibleMoeLayer = FlexibleMOELayer(gate, experts, args)
    # *****************************************************************************
    flexibleMoeLayer.cuda()

    output, l_aux = flexibleMoeLayer(input)
    # print(output)

    print("Rank: {}, Output size:{}".format(args.local_rank, output.size()))

    a = output.mean() + l_aux.mean()
    a.backward()

    print("Rank: {}, Success!".format(args.local_rank))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--sequence-length', type=int, default=8)
    parser.add_argument('--decoder-embed-dim', type=int, default=5)
    parser.add_argument('--decoder-ffn-dim', type=int, default=5)
    parser.add_argument('--moe-expert-count', type=int, metavar='D', default=8,
                        help='Number of experts in each MoE Layer')
    parser.add_argument('--moe-gating-use-fp32', default=False, action='store_true',
                        help="Use FP32 computations in MoE top2 gating function")
    parser.add_argument('--moe-second-expert-policy', type=str, default='sampling',
                        help="policy for second expert, options: all/sampling/random")
    parser.add_argument('--moe-normalize-gate-prob-before-dropping', default=False, action='store_true',
                        help="whether to normalize gate probs before or after dropping experts for capacity and randomization")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    dist.init_process_group("nccl")
    torch.cuda.set_device(args.local_rank)
    testfmoe(args)