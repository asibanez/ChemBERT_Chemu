gpu=$1

n_epochs=3

export TASK_NAME=ner
export TASK_DIR=./data/unl4bert.200k/

pre_model=bert-base-cased
# pre_model=./biobert/biobert_v1.1_pubmed/
# export OUTPUT_DIR=./bio-chembert_v2.0/
export OUTPUT_DIR=./chembert_v3.0/
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=${gpu} python finetune_mlm.py \
  --pregenerated_data ${TASK_DIR} \
  --bert_model ${pre_model} \
  --output_dir ${OUTPUT_DIR} \
  --epochs ${n_epochs} \
  --train_batch_size 32

