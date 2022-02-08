fairseq-train \
    /home/datasets/Fairseq-Data/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric


# fairseq-generate data-bin/iwslt14.tokenized.de-en \
#     --path checkpoints/checkpoint_best.pt \
#     --batch-size 128 --beam 5 --remove-bpe

# Generate test with beam=5: BLEU4 = 34.78, 68.7/43.1/28.9/19.9 (BP=0.963, ratio=0.964, syslen=126387, reflen=131161)


fairseq-generate /home/datasets/Fairseq-Data/iwslt14.tokenized.de-en \
    --path /home/storage/KDD22/checkpoints/thor/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
