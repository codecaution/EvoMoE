DIR=/vc_data/users/v-xiaonannie
NAME="wmt17_en_de"
MOE_FREQ=2
NUM_EXPERTS=4
EXPERT_NORMALIZE=world_size
GUMBEL_DECAY_ITERATIONS=5000
MAX_TEMP=2.0
MIN_TEMP=0.5
HARD_GUMBEL_ITERATIONS=6000
CHECKPOINT_FREQUENCY=500

# DATA_PATH=$DIR/Fairseq-Data/translation/wmt14_en_fr/
# DATA_PATH=$DIR/Fairseq-Data/translation/wmt14_fr_en/
DATA_PATH=$DIR/Fairseq-Data/translation/wmt17_en_de/
# DATA_PATH=$DIR/Fairseq-Data/translation/wmt17_de_en/

LOG_PATH="$DIR/KDD22/logs/${NAME}_Gumbel_${NUM_EXPERTS}"
CHECKPOINT_PATH="$DIR/KDD22/checkpoints/${NAME}_Gumbel_${NUM_EXPERTS}"

if [ ! -d "$CHECKPOINT_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$CHECKPOINT_PATH"
fi

if [ ! -d "$LOG_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$LOG_PATH"
fi

fairseq-train \
    $DATA_PATH \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion moe_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --moe-gate-loss-wt 0 --moe-gate-loss-combine-method sum \
    --moe-expert-count $NUM_EXPERTS --moe-freq $MOE_FREQ \
    --moe-gating-use-fp32 --moe-top1-expert \
    --moe-normalize-expert-grad $EXPERT_NORMALIZE\
    --moe-eval-capacity-token-fraction -1.0 \
    --use-gumbel-softmax \
    --gumbel-decay-scheduler Linear \
    --gumbel-decay-factor ${GUMBEL_DECAY_ITERATIONS} \
    --max-temperature ${MAX_TEMP} \
    --min-temperature ${MIN_TEMP} \
    --switch-to-hard-gumbel-softmax ${HARD_GUMBEL_ITERATIONS} \
    --ddp-backend fully_sharded \
    --save-interval-updates ${CHECKPOINT_FREQUENCY} \
    --validate-interval-updates ${CHECKPOINT_FREQUENCY} \
    --save-dir ${CHECKPOINT_PATH} \
    --max-tokens 4096 2>&1| tee -a $LOG_PATH/training.log


# fairseq-generate $DATA_PATH \
#     --path $CHECKPOINT_PATH/checkpoint_best-rank-0.pt \
#     --batch-size 128 --beam 5 --remove-bpe
#     --moe-expert-count $NUM_EXPERTS --moe-freq $MOE_FREQ \
#     --moe-gating-use-fp32 --moe-top1-expert \
#     --moe-normalize-expert-grad $EXPERT_NORMALIZE\
#     --moe-train-capacity-token-fraction 1.0 \
#     --moe-eval-capacity-token-fraction -1.0 2 > &1| tee $LOG_PATH/test.log
