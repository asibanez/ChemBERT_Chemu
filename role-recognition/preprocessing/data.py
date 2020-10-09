""" Generate training data for product recognition
"""

import argparse
import sys
import os
import csv

from collections import defaultdict
from chemdataextractor.doc import Paragraph

FOI = ["Reactants", "Yield", "Reaction", "Catalyst", "Solvent", "Temperature", "Time"]


def make_spans(tag_list):
    tag_list = list(map(int, tag_list.split(',')))
    spans = []
    for i in range(int(len(tag_list) / 2)):
        spans.append((tag_list[i*2], tag_list[i*2+1]))
    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True,
                        help="The full annotation file.")
    parser.add_argument("--output-file", type=str, required=True,
                        help="The product recognition data file.")
    args = parser.parse_args()

    data = []
    with open(args.annotation_file, "r") as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            text = row["description"]
            tokens = text.split(' ')
            product = row["Products"]
            product_tags = row["Products-tag"]

            if product_tags == "": # skip rows if products is empty
                assert(product == "")
                continue
            product_spans = make_spans(product_tags)

            # verify the correspondence between ${product} and tags
            val = " ".join([" ".join(tokens[start:end]) for (start, end) in product_spans])
            assert(val == product)

            sents = [s.text.split(' ') for s in Paragraph(text)]
            # make sure the indexes don't change
            # assert(text == " ".join([" ".join(sent) for sent in sents]))
            if (text != " ".join([" ".join(sent) for sent in sents])):
                print("text not matched after tokenization, skip.")
                continue
            # get sentence boundaries
            sent_boundaries = [0, ]
            for sent in sents:
                offset = sent_boundaries[-1]
                sent_boundaries.append(offset + len(sent))


            def get_segment(interval, boundaries, window=3):
                """ """
                cxt = int((window - 1) / 2)
                start, end = interval
                for i, b in enumerate(boundaries):
                    if start >= b and (end - 1 < boundaries[i+1]):
                        sent_id = i
                        break
                segment_start = boundaries[max(0, sent_id - cxt)]
                segment_end = boundaries[min(len(boundaries) - 1, sent_id + cxt + 1)]
                return (segment_start, segment_end)


            for span in product_spans: # for each product mention, create an individual instance
                seg_start, seg_end = get_segment(span, sent_boundaries, window=1)

                tagged_text = []
                for p, token in enumerate(tokens):
                    tagged_text.append([token, 'O'])

                # assign B/I- tags to each token
                for field in FOI:
                    fval = row[field]
                    fval_tags = row[field+"-tag"]
                    if fval_tags == "":
                        assert(fval == "")
                        continue
                    fval_spans = make_spans(fval_tags)

                    for fval_span in fval_spans:
                        start, end = fval_span
                        tagged_text[start][1] = f'B-{field}'
                        if end == start + 1:
                            continue
                        for i in range(start+1, end):
                            tagged_text[i][1] = f'I-{field}'

                prod_span_start, prod_span_end = span
                tagged_text.insert(prod_span_start, ["[P1]", "O"])
                tagged_text.insert(prod_span_end + 1, ["[P2]", "O"])

                tagged_segment = tagged_text[seg_start:(seg_end+2)]

                data.append(tagged_segment)

    with open(args.output_file, "w") as f:
        for tt in data:
            for token, tag in tt:
                f.write(f"{token}\t{tag}\n")
            f.write("\n")


if __name__ == '__main__':
    main()

