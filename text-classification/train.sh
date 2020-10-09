export DATA_DIR=./data
export TASK_NAME=SST-2

pretrained_model=$1
gpu=$2

# pretrained_model=/data/rsg/chemistry/sibanez/01_chem_nlp/00_pretrained/chembert_v3.0/
# pretrained_model=bert-base-uncased
# train
CUDA_VISIBLE_DEVICES=$gpu python run_glue.py \
  --model_name_or_path $pretrained_model \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir outputs/$pretrained_model \
  --overwrite_output_dir \
  --evaluate_during_training

