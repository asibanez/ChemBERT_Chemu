""" Generate training data for product recognition
"""

import argparse
import sys
import os
import csv
from tqdm import tqdm

from collections import defaultdict
from chemdataextractor.doc import Paragraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True,
                        help="The full annotation file.")
    parser.add_argument("--output-file", type=str, required=True,
                        help="The product recognition data file.")
    args = parser.parse_args()

    # data = []
    with open(args.output_file, "w") as fw:
        with open(args.annotation_file, "r") as fr:
            reader = csv.DictReader(fr, delimiter=',')
            for row in tqdm(reader):
                text = row["description"]
                sents = [s.text for s in Paragraph(text)]
                # data += sents
                for sent in sents:
                    fw.write(f"{sent}\n")

    # with open(args.output_file, "w") as f:
    #     for sent in data:
    #         f.write(f"{sent}\n")


if __name__ == '__main__':
    main()

