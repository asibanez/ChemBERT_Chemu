""" split data into train/dev/test
"""

import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor(object):
    """ Class for reading CoNLL-style data and split to train/test """

    def __init__(self, oversample_rate=3):
        self.oversample_rate = oversample_rate

    def get_full_dataset(self, filename):
        return self._create_examples(self._read_conll(filename))

    def over_sampling(self, X, Y):
        X_aug = []
        Y_aug = []
        for tokens, label in zip(X, Y):
            c = 1
            if "B-Prod" in label:
                c *= self.oversample_rate
            for i in range(c):
                X_aug.append(tokens)
                Y_aug.append(label)

        return (X_aug, Y_aug)

    def split_train_test(self, input_file, output_dir, ratio=0.1):
        examples = self.get_full_dataset(input_file)
        X, Y = list(zip(*examples))
        X = np.array(X)
        Y = np.array(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=ratio, random_state=42)

        X_train, Y_train = self.over_sampling(X_train, Y_train)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train.txt"), "w") as f:
            for tokens, label in zip(X_train, Y_train):
                f.write("\n".join([f"{token}\t{tag}" for (token, tag) in zip(tokens, label)]))
                f.write("\n\n")

        with open(os.path.join(output_dir, "valid.txt"), "w") as f:
            for tokens, label in zip(X_test, Y_test):
                f.write("\n".join([f"{token}\t{tag}" for (token, tag) in zip(tokens, label)]))
                f.write("\n\n")

    @classmethod
    def _read_conll(cls, input_file):
        with open(input_file) as f:
            data = []
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(sentence) > 0:
                        data.append(sentence)
                        sentence = []
                    continue
                token, tag = line.split('\t')
                sentence.append((token, tag))

        if len(sentence) > 0:
            data.append(sentence)
        return data

    def _create_examples(self, data):
        examples = []
        for i, sentence in enumerate(data):
            tokens = [t[0] for t in sentence]
            label = [t[1] for t in sentence]
            examples.append((tokens, label))
        return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--oversample-rate", type=int, default=1,
                        help="Oversampling rate for sentences that contain products.")
    parser.add_argument("--input-file", type=str, default="../data/all.txt",
                        help="Path to the full dataset.")
    parser.add_argument("--output-dir", type=str, default="./",
                        help="Directory for saving the output files.")
    args = parser.parse_args()
    processor = DataProcessor(oversample_rate=args.oversample_rate)
    processor.split_train_test(input_file=args.input_file, output_dir=args.output_dir)

