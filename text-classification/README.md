## Data Source Classification: Paper or Patent

### Environments
* Pytorch=1.5.0
* Transformers=2.11.0 (latest)

### Prepare training data
`cd data; python prepare_data.py`

### Train
`sh train.sh bert-base-uncased ${gpu_id}`

### Predict
`sh test.sh bert-base-uncased ${gpu_id}`

The predictions will be saved to `outputs/$pretrained_model/test_results_sst-2.txt` (0: Paper; 1: Patent)

### Performance
* `bert-base-uncased`: acc=93.9%
