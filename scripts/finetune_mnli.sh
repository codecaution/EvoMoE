ID=47
ROBERTA_PATH=/jizhicfs/brendenliu/roberta_log/5000robertalarge_complete/checkpoint${ID}.pt
LOG_PATH=glue_log/test/${ID}
mkdir -p ${LOG_PATH}
mkdir -p ${LOG_PATH}/mnli_checkpoint
MNLI_PATH=/apdcephfs/private_brendenliu/fairseq/MNLI-bin
CUDA_VISIBLE_DEVICES=2 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name mnli \
task.data=$MNLI_PATH checkpoint.restore_file=$ROBERTA_PATH checkpoint.save_dir=/apdcephfs/private_brendenliu/EvoMoE/${LOG_PATH}/mnli_checkpoint > ${LOG_PATH}/mnli.log 2>&1 
RTE_ROBERTA=/apdcephfs/private_brendenliu/fairseq/${LOG_PATH}/mnli_checkpoint/checkpoint_best.pt
RTE_PATH=/apdcephfs/private_brendenliu/fairseq/RTE-bin
CUDA_VISIBLE_DEVICES=2 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name rte \
task.data=$RTE_PATH checkpoint.restore_file=$RTE_ROBERTA > ${LOG_PATH}/rte_mnli.log 2>&1
