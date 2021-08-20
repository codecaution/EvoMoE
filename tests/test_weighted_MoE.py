import torch
import torch.nn as nn
from fairseq.modules.moe import WeightedMOELayer

def testWeighted():
    num_experts = 4
    input_shape = [2, 10, 20]
    input = torch.ones(input_shape)
    expert_list = []
    for i in range(4):
        expert_list.append(nn.Linear(20, 20))
    experts = nn.ModuleList(expert_list)
    weightedMoe = WeightedMOELayer(20, num_experts, experts)

    output = weightedMoe(input)
    # print(output[0].shape)
    # print(output[0][0])
    # print(output[0][1])

if __name__ == "__main__":
    testWeighted()