export DATA_DIR=./data
export TASK_NAME=SST-2

pretrained_model=$1
gpu=$2
# train
CUDA_VISIBLE_DEVICES=$gpu python run_glue.py \
  --model_name_or_path outputs/$pretrained_model \
  --task_name $TASK_NAME \
  --do_predict \
  --data_dir $DATA_DIR \
  --max_seq_length 512 \
  --per_device_eval_batch_size 8 \
  --output_dir outputs/$pretrained_model
