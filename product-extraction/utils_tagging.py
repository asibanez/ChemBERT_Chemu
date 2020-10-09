""" BERT sequence labeler fine-tuning: utilities to work with POS/NER tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from collections import namedtuple

from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

logger = logging.getLogger(__name__)

InputExample = namedtuple("InputExample", "guid text_a text_b label")
InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids label_ids")
Fields = namedtuple("Fields", "word_column tag_column")

ALL_FIELDS = {'ner': {'conll03': Fields(word_column=0, tag_column=-1),
                      'conll03_seg': Fields(word_column=0, tag_column=-1),
                      'onto': Fields(word_column=0, tag_column=-1),
                      'reaction': Fields(word_column=0, tag_column=-1)},
              'pos': {'upos': Fields(word_column=1, tag_column=3),
                      'ptb': Fields(word_column=1, tag_column=4)}}


def acc_and_f1(preds, labels):
    acc = accuracy_score(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    p = precision_score(y_true=labels, y_pred=preds)
    r = recall_score(y_true=labels, y_pred=preds)
    report = classification_report(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "precision:": p,
        "recall": r,
        # "report": report,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "pos":
        return {'acc': accuracy_score(preds, labels)}
    elif task_name == "ner":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


class DataProcessor(object):
    """Base class for data converters for sequence labeling data sets."""

    def __init__(self, task="ner", tagset="conll03"):
        self.task = task
        self.tagset = tagset
        self.fields = ALL_FIELDS[self.task][self.tagset]

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_conll(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # sentence-level, (doc-level on the way)
        with open(input_file, "r") as f:
            data = []
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(sentence) > 0:
                        data.append(sentence)
                        sentence = []
                    continue
                if line != "":
                    if line[0] == "#":
                        continue
                fields = line.split('\t')
                sentence.append(fields)

        if len(sentence) > 0:
            data.append(sentence)
        return data


class NerProcessor(DataProcessor):
    """Processor for CoNLL03-formated NER dataset."""
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        # only conll03 tags are implemented for now
        if self.tagset == "conll03_seg":
            return ["O", "B",  "I", "[CLS]", "[SEP]"]
        elif self.tagset == "conll03":
            return ["O",
                    "B-MISC", "I-MISC",
                    "B-PER", "I-PER",
                    "B-ORG", "I-ORG",
                    "B-LOC", "I-LOC",
                    "[CLS]",
                    "[SEP]"]
        elif self.tagset == "reaction":
            return ["O",
                    "B-Prod", "I-Prod",
                    "[CLS]",
                    "[SEP]"]
        else:
            raise KeyError("Unknown tagset: {}".format(self.tagset))

    def _create_examples(self, lines, set_type):
        examples = []
        for i, sentence in enumerate(lines):
            words = [t[self.fields.word_column] for t in sentence]
            label = [t[self.fields.tag_column] for t in sentence]
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(words)
            text_b = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PosProcessor(DataProcessor):
    """Processor for CoNLL-formated POS dataset."""
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_conll(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        if self.tagset == "ptb":
            return ["``", ",", ":", ".", "\'\'", "$", "AFX", "CC", "CD", "DT", "EX", "FW",
                    "HYPH", "IN", "JJ", "JJR", "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP",
                    "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP",
                    "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT",
                    "WP", "WP$", "WRB", "XX", "[CLS]", "[SEP]"]
        elif self.tagset == "upos":
            return [".", "ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRON", "PRT", "VERB", "X",
                    "[CLS]", "[SEP]"]
        else:
            raise KeyError("Unknown tagset: {}".format(self.tagset))

    def _create_examples(self, lines, set_type):
        examples = []
        for i, sentence in enumerate(lines):
            words = [t[self.fields.word_column] for t in sentence]
            label = [t[self.fields.tag_column] for t in sentence]
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(words)
            text_b = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


processors = {
    "pos": PosProcessor,
    "ner": NerProcessor,
}


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        BERT pattern: [CLS] + A + [SEP]
    """

    label_map = {label : i for i, label in enumerate(label_list, 0)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens = [cls_token] + tokens_a + [sep_token]
        if len(tokens) > max_seq_length:
            logger.info("Sentence length exceeds max_seq_length")
            max_seq_length = len(tokens)
        segment_ids = [sequence_a_segment_id] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_ids = [-1] * max_seq_length
        labels = [cls_token] + example.label + [sep_token]
        label_index = 0
        for tk_index, token in enumerate(tokens):
            if not token.startswith("##"):
                if labels[label_index] not in label_map:
                    label_ids[tk_index] = len(label_map) + 5 # a unique id
                else:
                    label_ids[tk_index] = label_map[labels[label_index]]
                label_index += 1
        assert label_index == len(labels)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))
    return features

