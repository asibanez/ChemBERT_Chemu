from seqeval.metrics import f1_score
import sys
from seqeval.metrics import classification_report

y_true = []
y_pred = []
ys_true = []
ys_pred = []

for line in open(sys.argv[1]):
    line = line.strip()
    if line == "":
        ys_true.append(y_true)
        ys_pred.append(y_pred)
        y_true = []
        y_pred = []
    else:
        line = line.split('\t')
        y_true.append(line[1])
        y_pred.append(line[2])

f1_score = f1_score(ys_true, ys_pred)
report = classification_report(ys_true, ys_pred)
print(f1_score)
print(report)

