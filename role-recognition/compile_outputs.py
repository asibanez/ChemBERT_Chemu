import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_file", type=str, required=True,
                    help="Gold test file")
parser.add_argument("--tag_file", type=str, required=True,
                    help="Predicted tag file")
parser.add_argument("--output", type=str, required=True,
                    help="Compiled outputs")

args = parser.parse_args()

fr_test = open(args.test_file, "r")
fr_tag = open(args.tag_file, "r")

with open(args.output, "w") as fw:
    for l1, l2 in zip(fr_test, fr_tag):
        l1 = l1.strip()
        l2 = l2.strip()
        if l1 == "":
            fw.write("\n")
        else:
            fw.write(f"{l1}\t{l2}\n")

fr_test.close()
fr_tag.close()

