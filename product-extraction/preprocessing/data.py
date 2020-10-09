""" Generate training data for product recognition
"""

import argparse
import sys
import os
import csv

from collections import defaultdict
from chemdataextractor.doc import Paragraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True,
                        help="The full annotation file.")
    parser.add_argument("--output-file", type=str, required=True,
                        help="The product recognition data file.")
    args = parser.parse_args()

    data = defaultdict(list)
    with open(args.annotation_file, "r") as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            text = row["description"]
            tokens = text.split(' ')
            product = row["Products"]
            product_tags = row["Products-tag"]
            if product_tags == "":
                assert(product == "")
                continue
            product_tags = list(map(int, product_tags.split(',')))

            # verify the correspondence between ${product} and tags
            product_spans = []
            for i in range(int(len(product_tags) / 2)):
                product_spans.append((product_tags[i*2], product_tags[i*2+1]))
            val = " ".join([" ".join(tokens[start:end]) for (start, end) in product_spans])
            assert(val == product)

            for span in product_spans:
                if span not in data[text]:
                    data[text].append(span)

    n_sents = 0
    with open(args.output_file, "w") as f:
        # segment each paragraph into sentences, and map indexes correspondingly
        for text, prod_spans in data.items():
            print(prod_spans)
            sents = [s.text.split(' ') for s in Paragraph(text)]
            print("{} sentences detected".format(len(sents)))
            n_sents += len(sents)

            # make sure the indexes don't change
            # assert(text == " ".join([" ".join(sent) for sent in sents]))
            merged_text = " ".join([" ".join(sent) for sent in sents])
            if text != merged_text:
                print("text: %s (len: %d)" % (text, len(text)))
                print("merged text: %s (len: %d)" % (merged_text, len(merged_text)))

            # get sentence boundaries
            sent_boundaries = [0, ]
            for sent in sents:
                offset = sent_boundaries[-1]
                sent_boundaries.append(offset + len(sent))
            print(sent_boundaries)

            # initialize all tokens with tag "O"
            tagged_text = []
            for p, token in enumerate(text.split(' ')):
                tagged_text.append([token, 'O'])


            # check if a span (interval) crosses any sentence boundary
            def cross_boundary(interval, refs):
                # e.g., interval = [10, 12]
                for i, b in enumerate(refs):
                    if b > interval[0] and b <= interval[1] - 1:
                        return i
                return -1


            # tag Product tokens
            for span in prod_spans:
                # if the span crosses sentence boundaries, skip (or merge the two sents)
                if cross_boundary(span, sent_boundaries) >= 0:
                    print("cross_boundary!")
                    continue

                start, end = span
                tagged_text[start][1] = 'B-Prod'
                if end == start + 1:
                    continue
                for i in range(start+1, end):
                    tagged_text[i][1] = 'I-Prod'

            # split paragraph to sentences
            tagged_sents = []
            for i in range(len(sents)):
                bos = sent_boundaries[i]
                eos = sent_boundaries[i+1]
                tagged_sents.append(tagged_text[bos:eos])

            for ts in tagged_sents:
                for token, tag in ts:
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")


if __name__ == '__main__':
    main()

