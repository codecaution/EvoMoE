ID=47
ROBERTA_PATH=/jizhicfs/brendenliu/roberta_log/5000robertalarge_complete/checkpoint${ID}.pt
LOG_PATH=glue_log/test/${ID}
mkdir -p ${LOG_PATH}
mkdir -p ${LOG_PATH}/sts_b_checkpoint
#ROBERTA_PATH=/jizhicfs/brendenliu/EvoMoE/roberta.large/model.pt
#ROBERTA_PATH=/apdcephfs/private_brendenliu/fairseq/robert.base/roberta.base/model.pt
SST_PATH=/apdcephfs/private_brendenliu/fairseq/SST-2-bin
CUDA_VISIBLE_DEVICES=1 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name sst_2 \
task.data=$SST_PATH checkpoint.restore_file=$ROBERTA_PATH > ${LOG_PATH}/sst2.log 2>&1 
RTE_PATH=/apdcephfs/private_brendenliu/fairseq/RTE-bin
CUDA_VISIBLE_DEVICES=1 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name rte \
task.data=$RTE_PATH checkpoint.restore_file=$ROBERTA_PATH > ${LOG_PATH}/rte.log 2>&1
#sleep 60s
CoLA_PATH=/apdcephfs/private_brendenliu/fairseq/CoLA-bin
CUDA_VISIBLE_DEVICES=1 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name cola \
task.data=$CoLA_PATH checkpoint.restore_file=$ROBERTA_PATH > ${LOG_PATH}/cola.log 2>&1 
#sleep 60s
MRPC_PATH=/apdcephfs/private_brendenliu/fairseq/MRPC-bin
CUDA_VISIBLE_DEVICES=1 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name mrpc \
task.data=$MRPC_PATH checkpoint.restore_file=$ROBERTA_PATH > ${LOG_PATH}/mrpc.log 2>&1 
#sleep 60s
STS_PATH=/apdcephfs/private_brendenliu/fairseq/STS-B-bin
CUDA_VISIBLE_DEVICES=1 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name sts_b \
task.data=$STS_PATH checkpoint.restore_file=$ROBERTA_PATH checkpoint.save_dir=${LOG_PATH}/sts_b_checkpoint > ${LOG_PATH}/sts_b.log 2>&1 
#sleep 60s
QNLI_PATH=/apdcephfs/private_brendenliu/fairseq/QNLI-bin
CUDA_VISIBLE_DEVICES=1 fairseq-hydra-train --config-dir examples/roberta/config/finetuning --config-name qnli \
task.data=$QNLI_PATH checkpoint.restore_file=$ROBERTA_PATH > ${LOG_PATH}/qnli.log 2>&1 
