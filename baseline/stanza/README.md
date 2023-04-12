# Evaluation

`predict_stanza.py` takes an IOB file as input and produces an output IOB file with NER predictions obtained using the Stanza model `CoNLL02` for Spanish.

```
❯ pip install stanza seqeval
❯ python3 predict_stanza.py dataset-es.iob > stanza-baseline_predictions.txt
❯ python3 ../../utils/eval.py -c stanza-baseline_predictions.txt dataset-es.iob
LOC,0.924,0.719,0.809
MISC,0.120,0.242,0.160
ORG,0.266,0.662,0.380
PER,0.316,0.823,0.456
micro avg,0.683,0.689,0.686
macro avg,0.406,0.612,0.451
weighted avg,0.817,0.689,0.732
```
