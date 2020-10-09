task_name=ner

n_epochs=3

train_corpus=$1
output_dir=$2

python pregenerate_training_data_mlm.py \
    --train_corpus ${train_corpus} \
    --output_dir ${output_dir} \
    --task_name ${task_name} \
    --do_whole_word_mask \
    --bert_model bert-base-cased \
    --epochs_to_generate ${n_epochs} \
    --max_seq_len 128 \
    --no_nsp

