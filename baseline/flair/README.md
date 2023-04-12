# Evaluation

`predict_flair.py` takes an IOB file as input and produces an output IOB file with NER predictions obtained using the Flair model `ner-spanish-large`.

```
❯ pip install flair seqeval
❯ python3 predict_flair.py dataset-es.iob > flair-baseline_predictions.txt
❯ python3 ../../utils/eval.py -c flair-baseline_predictions.txt dataset-es.iob
LOC,0.927,0.776,0.845
MISC,0.170,0.455,0.247
ORG,0.455,0.719,0.558
PER,0.702,0.867,0.776
micro avg,0.748,0.756,0.752
macro avg,0.563,0.704,0.606
weighted avg,0.848,0.756,0.791
```
