#! /bin/bash
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1

DATA_NAME=cc100
BATCH_SIZE=8
NUM_EXPERTS=8
MAX_UPDATES=100000
UPDATE_FREQ=1
WARMUP_STEPS=5000
MOE_FREQ=0
LR=1e-4
CHECKPOINT_FREQUENCY=50000
TOKENS_PER_SAMPLE=512
echo -e "\n\n\n\n"
echo "=====================================ARGS======================================"
echo "data_name: ${DATA_NAME}"
echo "batch size: ${BATCH_SIZE}"
echo "num_experts: ${NUM_EXPERTS}"
echo "max updates: ${MAX_UPDATES}"
echo "update_freq: ${UPDATE_FREQ}"
echo "warmup_steps: ${WARMUP_STEPS}"
echo "moe_freq: ${MOE_FREQ}"
echo "gumbel_decay_iterations: ${GUMBEL_DECAY_ITERATIONS}"
echo "min_temperature: ${MIN_TEMP}"
echo "max_temperature: ${MAX_TEMP}"
echo "hard_gumbel_iterations: ${HARD_GUMBEL_ITERATIONS}"
echo "lr: ${LR}"
echo "checkpoint_frequency: ${CHECKPOINT_FREQUENCY}"

echo -e "\n\n\n\n"
echo "=====================================PATH========================================"

# Path
# Path
DATA_PATH=/jizhicfs/brendenliu/roberta_corpus/bertcorpus
#DATA_PATH=/jizhicfs/brendenliu/roberta_corpus/wiki
#DATA_PATH=/apdcephfs/private_brendenliu/fairseq/data-bin/wikitext-103
#CHECKPOINT_PATH="/jizhicfs/brendenliu/roberta"
CHECKPOINT_PATH="/jizhicfs/brendenliu/roberta_log/gpt_nomoe"
LOG_DIR="/jizhicfs/brendenliu/roberta_log/gpt_nomoe/log"
TENSORBOARD_PATH="/jizhicfs/brendenliu/roberta_log/gpt_nomoe/tensorboard"

if [ ! -d "$CHECKPOINT_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$CHECKPOINT_PATH"
fi

if [ ! -d "$LOG_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$LOG_DIR"
fi

if [ ! -d "$TENSORBOARD_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$TENSORBOARD_PATH"
fi

LOG_PATH=$LOG_DIR/rank_${INDEX}.log

echo "DATA_PATH: ${DATA_PATH}"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "LOG_PATH=: ${LOG_PATH}"
echo "TENSORBOARD_PATH =: ${TENSORBOARD_PATH}"

echo -e "\n\n\n\n"
echo "=====================================DDP Option========================================"
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=$CHIEF_IP
MASTER_PORT=60001
NNODES=1
NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
ddp_options="--distributed-world-size ${WORLD_SIZE} --nprocs-per-node ${GPUS_PER_NODE} --distributed-rank ${NODE_RANK} --distributed-init-method tcp://${MASTER_ADDR}:${MASTER_PORT}"
echo "ddp_options: ${ddp_options}"
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
DLWS_NUM_WORKER=8






python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
      /jizhicfs/brendenliu/stablemoe/train.py  \
      ${DATA_PATH} \
    --task language_modeling \
    --save-dir ${CHECKPOINT_PATH} \
    --arch transformer_lm_BaseGPT_x1_small \
    --moe-type base_layer \
    --moe-freq $MOE_FREQ \
    --moe-sublayers 16 \
    --two-stage-updates 6000 \
    --distill-assignment \
    --distilled-model wordemb \
    --distill-factor 0.3 \
    --criterion xentropy_aux \
    --balance-loss balance \
    --balance-factor 0.3 \
    --capacity-factor 2 \
    --assignment-algorithm GA \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.1 \
    --lr 0.0006 \
    --lr-scheduler polynomial_decay \
    --total-num-update ${MAX_UPDATES} \
    --warmup-updates ${WARMUP_STEPS} \
    --tokens-per-sample ${TOKENS_PER_SAMPLE} \
    --sample-break-mode none \
    --batch-size ${BATCH_SIZE} \
    --pad-to-fixed-length \
    --pad-to-fixed-bsz \
    --update-freq ${UPDATE_FREQ} \
    --max-update ${MAX_UPDATES} \
    --ddp-backend=legacy_ddp \
    --log-interval 1 \
    --log-format json \
    --ddp-backend=legacy_ddp \
    --validate-interval-updates 500 \
    --save-interval 5 \
    --tensorboard-logdir ../tblogs/$jobname \
    --distributed-no-spawn \
    --write-checkpoints-asynchronously \
    --save-dir ${CHECKPOINT_PATH} \
    --save-interval-updates ${CHECKPOINT_FREQUENCY} \
    --tensorboard-logdir ${TENSORBOARD_PATH} \
    --seed 1234 2>&1 | tee $LOG_PATH
