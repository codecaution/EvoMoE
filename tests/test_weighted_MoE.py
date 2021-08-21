import torch
import torch.nn as nn
import argparse
from fairseq.modules.moe import WeightedMOELayer

def testWeighted(args):
    num_experts = 2
    input_shape = [2, 2, args.decoder_embed_dim]
    input = torch.randn(input_shape)
    print(input)
    expert_list = []
    for i in range(num_experts):
        expert_list.append(nn.Linear(args.decoder_embed_dim, args.decoder_ffn_dim))
    experts = nn.ModuleList(expert_list)
    weightedMoe = WeightedMOELayer(experts, args)

    output = weightedMoe(input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder-embed-dim', type=int, default=5)
    parser.add_argument('--decoder-ffn-dim', type=int, default=5)
    args = parser.parse_args()
    testWeighted(args)