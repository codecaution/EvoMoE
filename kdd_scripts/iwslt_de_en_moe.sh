NAME="iwslt_de_en_moe_2"
MOE_FREQ=2
NUM_EXPERTS=4
EXPERT_NORMALIZE=world_size
GUMBEL_DECAY_ITERATIONS=1000
MAX_TEMP=2.0
MIN_TEMP=0.5
HARD_GUMBEL_ITERATIONS=2000
LOG_DIR="/home/storage/KDD22/logs/${NAME}"
CHECKPOINT_PATH="/home/storage/KDD22/checkpoints/${NAME}"

if [ ! -d "$CHECKPOINT_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$CHECKPOINT_PATH"
fi

if [ ! -d "$LOG_DIR" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$LOG_DIR"
fi

LOG_PATH=${LOG_DIR}/training.log

# python -u train.py \
#     /home/datasets/Fairseq-Data/iwslt14.tokenized.de-en \
#     --arch transformer_iwslt_de_en_moe --share-decoder-input-output-embed \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion moe_label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
#     --moe-expert-count $NUM_EXPERTS --moe-freq $MOE_FREQ \
#     --moe-gating-use-fp32 --moe-top1-expert \
#     --moe-normalize-expert-grad $EXPERT_NORMALIZE\
#     --moe-train-capacity-token-fraction 4.0 \
#     --moe-eval-capacity-token-fraction -1.0 \
#     --ddp-backend fully_sharded \
#     --save-dir ${CHECKPOINT_PATH} \
#     --max-tokens 4096 \
#     --distributed-world-size 4 2>&1 | tee $LOG_PATH

# python -u train.py \
#     /home/datasets/Fairseq-Data/iwslt14.tokenized.de-en \
#     --arch transformer_iwslt_de_en_moe --share-decoder-input-output-embed \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion moe_label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --moe-expert-count $NUM_EXPERTS --moe-freq $MOE_FREQ \
#     --moe-gating-use-fp32 --moe-topk-expert \
#     --moe-normalize-expert-grad sqrt_world_size \
#     --moe-eval-capacity-token-fraction -1.0 \
#     --use-gumbel-softmax \
#     --gumbel-decay-scheduler Linear \
#     --gumbel-decay-factor ${GUMBEL_DECAY_ITERATIONS} \
#     --max-temperature ${MAX_TEMP} \
#     --min-temperature ${MIN_TEMP} \
#     --switch-to-hard-gumbel-softmax ${HARD_GUMBEL_ITERATIONS} \
#     --ddp-backend fully_sharded \
#     --save-dir ${CHECKPOINT_PATH} \
#     --max-tokens 4096 \
#     --distributed-world-size 4 2>&1 | tee $LOG_PATH

    # --eval-bleu \
    # --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    # --eval-bleu-detok moses \
    # --eval-bleu-remove-bpe \
    # --eval-bleu-print-samples \
    # --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 

python -u train.py \
    /home/datasets/Fairseq-Data/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en_moe --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion moe_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --moe-gate-loss-wt 0.01 --moe-gate-loss-combine-method sum \
    --moe-expert-count $NUM_EXPERTS --moe-freq $MOE_FREQ \
    --moe-gating-use-fp32 --moe-top1-expert \
    --moe-normalize-expert-grad $EXPERT_NORMALIZE\
    --moe-train-capacity-token-fraction 2.0 \
    --moe-eval-capacity-token-fraction -1.0 \
    --save-dir ${CHECKPOINT_PATH} \
    --max-tokens 4096 \
    --distributed-world-size 4 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 2>&1 | tee $LOG_PATH 


# fairseq-validate \
#     /home/datasets/Fairseq-Data/iwslt14.tokenized.de-en \
#     --valid-subset test \
#     --path /home/storage/KDD22/checkpoints/iwslt_de_en_moe/checkpoint_best-rank-0.pt \
#     --batch-size 4 \
#     --task translation \
#     --criterion moe_label_smoothed_cross_entropy
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples

# fairseq-generate /home/datasets/Fairseq-Data/iwslt14.tokenized.de-en \
#     --path /home/storage/KDD22/checkpoints/iwslt_de_en_moe_gumbel/checkpoint_best-rank-0.pt \
#     --batch-size 128 --beam 5 --remove-bpe
#     --moe-expert-count $NUM_EXPERTS --moe-freq $MOE_FREQ \
#     --moe-gating-use-fp32 --moe-top1-expert \
#     --moe-normalize-expert-grad $EXPERT_NORMALIZE\
#     --moe-train-capacity-token-fraction 4.0 \
#     --moe-eval-capacity-token-fraction -1.0 \ 

# Generate test with beam=5: BLEU4 = 34.78, 68.7/43.1/28.9/19.9 (BP=0.963, ratio=0.964, syslen=126387, reflen=131161)
