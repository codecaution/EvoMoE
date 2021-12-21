python -m torch.distributed.launch --nproc_per_node=4 \
    --nnodes=1 --node_rank=0 \
    --master_addr="localhost" --master_port=1234 tests/test_flexible_MoE.py 