## EvoMoE: An Evolutional Mixture-of-Experts Training Framework via Dense-To-Sparse Gate
This code is for paper "EvoMoE: An Evolutional Mixture-of-Experts Training Framework via Dense-To-Sparse Gate" [[PDF]](https://arxiv.org/abs/2112.14397) (Under Review), which is implemented based on [Fairseq](https://github.com/pytorch/fairseq).

### Dependencies and Installation:
1. Please install the following packages at first.
```bash
# Apex
pip install -v --no-cache-dir --global-option="--cpp_ext" \
    --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" \
    --global-option="--xentropy" --global-option="--fast_multihead_attn" \
    git+git://github.com/NVIDIA/apex.git@e2083df5eb96643c61613b9df48dd4eea6b07690

# FairScale  
pip install fairscale==0.4.0

# Hydra
pip install hydra-core==1.0.7 omegaconf==2.0.6

#For large datasets, please install PyArrow
pip install pyarrow
```
2. Follow the fairseq installation instructions as [original repo](https://github.com/pytorch/fairseq/#requirements-and-installation). To install EvoMoE and develop locally:

``` bash
git clone https://github.com/pytorch/EvoMoE
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq
```
### Training
1. Prepare the training data by following the official example[link](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md).
2. Train a Dense-to-Sparse MoE model as:
```
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
      --task language_modeling \
      --arch transformer_lm_gptxl_moe \
      --share-decoder-input-output-embed \
      --tokens-per-sample ${TOKENS_PER_SAMPLE} --batch-size ${BATCH_SIZE} --update-freq ${UPDATE_FREQ} \
      --lr ${LR} --lr-scheduler polynomial_decay --warmup-updates ${WARMUP_STEPS}  \
      --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
      --clip-norm 5.0 --weight-decay 0.1 --dropout 0.1 \
      --criterion cross_entropy \
      --write-checkpoints-asynchronously \
      --save-dir ${CHECKPOINT_PATH} \
      --save-interval-updates ${CHECKPOINT_FREQUENCY} \
      --num-workers ${DLWS_NUM_WORKER}\
      --ddp-backend fully_sharded \
      --checkpoint-activations \
      --max-update ${MAX_UPDATES} \
      --total-num-update ${MAX_UPDATES} \
      --validate-interval-updates ${CHECKPOINT_FREQUENCY} \
      --log-format json --log-interval 100 \
      --symlink \
      --seed 1234 2>&1 | tee -a $LOG_PATH
```

### Citation

Please cite as:

``` bibtex
@article{nie2021densetosparse,
  title = {EvoMoE: An Evolutional Mixture-of-Experts Training Framework via Dense-To-Sparse Gate},
  author = {Nie, Xiaonan and Cao, Shijie and Miao, Xupeng and Ma, Lingxiao and Xue, Jilong and Miao, Youshan and Yang, Zichao and Yang, Zhi and Cui, Bin},
  journal = {arXiv preprint arXiv:2112.14397},
  year = {2021},
}
```
