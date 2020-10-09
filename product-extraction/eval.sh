gpu=$2

oversample_rate=5

#task: pos | ner
export TASK_NAME=ner
tagset=reaction
export TASK_DIR=data/ # /oversample_r${oversample_rate}/
export MODEL_DIR=bert-base-cased
# export MODEL_DIR=$1

n_epochs=3
OUTPUT_DIR=$1/oversample_r${oversample_rate}/
output_file="test.tags.preds"

CUDA_VISIBLE_DEVICES=${gpu} python run_tagging.py \
    --model_name_or_path ${MODEL_DIR} \
    --task_name $TASK_NAME \
    --tagset ${tagset} \
    --do_eval \
    --eval_on_test \
    --data_dir $TASK_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs ${n_epochs} \
    --output_dir ${OUTPUT_DIR} \
    --write_outputs \
    --output_file ${output_file} \
    --overwrite_output_dir
    # --local_rank 2

# merge "test.pred"
test_file=${TASK_DIR}/test.txt
python compile_outputs.py \
    --test_file ${test_file} \
    --tag_file ${OUTPUT_DIR}/${output_file} \
    --output ${OUTPUT_DIR}/test.preds
echo "Compiled outputs to:", ${OUTPUT_DIR}/test.preds

