""" This program prepare training/test data for Paper/Patent domain respectively.

    For Paper domain, we take passages from our own database (ACS publications)
    - xxx passages

    For Patent domain, we take passages from ChEMU challenges
    - xxx passages
"""

import csv

# "paper" domain:
# - train data: ACS journals
# - test data: DARPA test passages
paper_train_path = "./annotations/01_paper_style/ann_original_all"
paper_test_path = "./annotations/01_paper_style/ann_darpa_test"

# "patents" domain:
# - train data: ChEMU challenge (patents)
# - test data: DARPA test passages
patent_train_path = "./annotations/02_patent_style/ann_chemu_all"
patent_test_path = "./annotations/02_patent_style/ann_darpa_test"


ALL_DATA = [
    {
        'style': 'paper',
        'train': paper_train_path,
        'test': paper_test_path,
        'train_field': 'description',
        'test_field': 'description'
    },
    {
        'style': 'patent',
        'train': patent_train_path,
        'test': patent_test_path,
        'train_field': 'Text',
        'test_field': 'description'
    }
]

from chemdataextractor.nlp.tokenize import ChemWordTokenizer
cwt = ChemWordTokenizer()


def read_instances(path, field, label):
    data = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            text = row[field]
            tokens = cwt.tokenize(text)
            data.append((" ".join(tokens), label))
    return data


def write_instances(data, path):
    with open(path, "w") as f:
        f.write("sentence\tlabel\n")
        for text, label in data:
            f.write("{}\t{}\n".format(text, label))

def write_test_instances(data, path):
    with open(path, "w") as f:
        f.write("index\tsentence\n")
        for i, (text, label) in enumerate(data):
            f.write("{}\t{}\n".format(i, text))

def load_data():
    train_data = []
    test_data = []
    for i, d in enumerate(ALL_DATA):
        domain_label = i
        # train
        train_data += read_instances(d['train'], d['train_field'], domain_label)
        test_data += read_instances(d['test'], d['test_field'], domain_label)

    return train_data, test_data

if __name__ == '__main__':
    train_data, test_data = load_data()
    write_instances(train_data, "train.tsv")
    print("Train data saved to train.tsv")
    write_instances(test_data, "dev.tsv")
    print("Test data saved to dev.tsv") # should be dev data if available
    write_test_instances(test_data, "test.tsv")
    print("Test data saved to test.tsv")

