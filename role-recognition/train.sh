gpu=$3

# oversample_rate=5

# task: pos | ner
export TASK_NAME=ner
tagset=reaction
data_type=sent
export TASK_DIR=data_v3/${data_type}
# export MODEL_DIR=bert-base-cased
# model_dir: bert-base-cased, path_to_biobert, path_to_chembert
export MODEL_DIR=$1

output_dir=$2

n_epochs=20

CUDA_VISIBLE_DEVICES=${gpu} python run_tagging.py \
    --model_name_or_path ${MODEL_DIR} \
    --task_name $TASK_NAME \
    --tagset ${tagset} \
    --do_train \
    --do_eval \
    --data_dir $TASK_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs ${n_epochs} \
    --output_dir ${output_dir}/ \
    --overwrite_output_dir \
    --evaluate_during_training \
    --logging_steps 200 \
    --save_steps -1
    # --freeze_bert \
    # --local_rank 2

