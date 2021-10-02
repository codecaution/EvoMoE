#! /bin/bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT

echo -e "\n\n\n\n"
echo "=====================================ARGS======================================"
echo "data_name: ${DATA_NAME}"
echo "batch size: ${BATCH_SIZE}"
echo "max updates: ${MAX_UPDATES}"
echo "update_freq: ${UPDATE_FREQ}"
echo "warmup_steps: ${WARMUP_STEPS}"
echo "lr: ${LR}"
echo "checkpoint_frequency: ${CHECKPOINT_FREQUENCY}"

echo -e "\n\n\n\n"
echo "=====================================PATH========================================"

# Path
shared_storage="/vc_data/users/v-xiaonannie"
blob_path="/home/v-xiaonannie/xiaonan"

if [ -d "${shared_storage}/Fairseq-Data/${DATA_NAME}" ]; then
  DATA_PATH="${shared_storage}/Fairseq-Data/${DATA_NAME}"
else
  DATA_PATH="${blob_path}/Fairseq-Data/${DATA_NAME}"
fi

CHECKPOINT_PATH="${shared_storage}/Fairseq-Results/checkpoints/${DATA_NAME}/${Version}"
LOG_DIR="${blob_path}/Fairseq-Results/logs/${DATA_NAME}/${Version}"

if [ ! -d "$CHECKPOINT_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$CHECKPOINT_PATH"
fi

if [ ! -d "$LOG_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$LOG_DIR"
fi

LOG_PATH=$LOG_DIR/rank_${NODE_RANK}.log

echo "DATA_PATH: ${DATA_PATH}"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "LOG_PATH=: ${LOG_PATH}"

echo -e "\n\n\n\n"
echo "=====================================DDP Option========================================"
GPU_PER_NODE_COUNT=$DLWS_NUM_GPU_PER_WORKER
MASTER_ADDR=$MASTER_IP
MASTER_PORT=$MASTER_PORT

if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  ddp_options="--distributed-world-size $GPU_PER_NODE_COUNT --distributed-rank 0 --distributed-init-method tcp://${MASTER_ADDR}:${MASTER_PORT}"
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
	ddp_options="--distributed-world-size $GPU_PER_NODE_COUNT --distributed-rank 0 --distributed-init-method tcp://${MASTER_ADDR}:${MASTER_PORT}"
  else
    local_rank=$(($GPU_PER_NODE_COUNT*$OMPI_COMM_WORLD_RANK))
    total_gpu=$(($OMPI_COMM_WORLD_SIZE*$GPU_PER_NODE_COUNT))
    ddp_options="--distributed-world-size $total_gpu --distributed-rank $local_rank --distributed-init-method tcp://${MASTER_ADDR}:${MASTER_PORT}"
  fi
fi
echo "ddp_options: ${ddp_options}"

python train.py ${ddp_options} \
      ${DATA_PATH} \
      --task masked_lm \
      --sample-break-mode complete \
      --arch roberta_large \
      --criterion masked_lm \
      --tokens-per-sample ${TOKENS_PER_SAMPLE} --batch-size ${BATCH_SIZE} --update-freq ${UPDATE_FREQ} \
      --lr ${LR} --lr-scheduler polynomial_decay --warmup-updates ${WARMUP_STEPS}  \
      --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
      --clip-norm 0.0 --weight-decay 0.01 --dropout 0.1 \
      --write-checkpoints-asynchronously \
      --save-dir ${CHECKPOINT_PATH} \
      --save-interval-updates ${CHECKPOINT_FREQUENCY} \
      --num-workers ${DLWS_NUM_WORKER}\
      --ddp-backend fully_sharded \
      --max-update ${MAX_UPDATES} \
      --total-num-update ${MAX_UPDATES} \
      --validate-interval-updates ${CHECKPOINT_FREQUENCY} \
      --log-format json --log-interval 100 \
      --symlink \
      --skip-invalid-size-inputs-valid-test \
      --seed 1234 2>&1 | tee -a $LOG_PATH