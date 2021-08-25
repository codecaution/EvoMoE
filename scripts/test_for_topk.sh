NUM_EXPERTS=4
TOKENS_PER_SAMPLE=2048
python fairseq_cli/train.py \
  --ddp-backend fully_sharded --memory-efficient-fp16 --checkpoint-activations \
  --task dummy_lm --tokens-per-sample $TOKENS_PER_SAMPLE \
  --arch transformer_lm_gpt --share-decoder-input-output-embed \
  --decoder-layers 24 --decoder-embed-dim 2048 --decoder-ffn-embed-dim 8192 \
  --decoder-attention-heads 32 \
  --moe-expert-count $NUM_EXPERTS --moe-freq 2 \
  --moe-gating-use-fp32 --moe-second-expert-policy all \
  --moe-normalize-expert-grad sqrt_world_size \
  --moe-eval-capacity-token-fraction -1.0 \
  --max-sentences-valid 1 --num-workers-valid 0 \
  --criterion cross_entropy \
  --optimizer adam --fp16-adam-stats --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0005 --warmup-updates 750 \
  --dropout 0.1 --attention-dropout 0.1 \
  --batch-size 3 --update-freq 1 \
  --moe-topk-expert \
  --max-update 250 --disable-validation \
  --log-format json --log-interval 10
