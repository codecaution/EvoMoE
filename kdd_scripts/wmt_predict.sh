DIR=/vc_data/users/v-xiaonannie
NAME="wmt14_en_fr"
DATA_PATH=$DIR/Fairseq-Data/translation/wmt14_en_fr/
# DATA_PATH=$DIR/Fairseq-Data/translation/wmt14_fr_en/
# DATA_PATH=$DIR/Fairseq-Data/translation/wmt17_en_de/
# DATA_PATH=$DIR/Fairseq-Data/translation/wmt17_de_en/

LOG_PATH="$DIR/KDD22/logs/${NAME}"
CHECKPOINT_PATH="$DIR/KDD22/checkpoints/${NAME}"
CHECKPOINT_FREQUENCY=500

if [ ! -d "$CHECKPOINT_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$CHECKPOINT_PATH"
fi

if [ ! -d "$LOG_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p "$LOG_PATH"
fi

#fairseq-train \
#    $DATA_PATH \
#    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#    --dropout 0.3 --weight-decay 0.0001 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --save-dir ${CHECKPOINT_PATH} \
#    --save-interval-updates ${CHECKPOINT_FREQUENCY} \
#    --validate-interval-updates ${CHECKPOINT_FREQUENCY} \
#    --max-tokens 4096 2>&1| tee $LOG_PATH/training.log


fairseq-generate $DATA_PATH \
     --path $CHECKPOINT_PATH/checkpoint_best.pt \
     --batch-size 128 --beam 5 --remove-bpe 2>&1| tee -a $LOG_PATH/test.log
