ID=47
ROBERTA_PATH=/jizhicfs/brendenliu/roberta_log/5000robertalarge_complete/checkpoint${ID}.pt
LOG_PATH=glue_log/test/${ID}
mkdir -p ${LOG_PATH}
QQP_PATH=/apdcephfs/private_brendenliu/fairseq/QQP-bin
CUDA_VISIBLE_DEVICES=4 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name qqp \
task.data=$QQP_PATH checkpoint.restore_file=$ROBERTA_PATH > ${LOG_PATH}/qqp.log 2>&1 &
