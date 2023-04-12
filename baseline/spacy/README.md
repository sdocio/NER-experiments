# Evaluation

The same script for evaluating the customized spaCy models was used for getting NER predictions using the standard spaCy models with our dataset.

```
❯ pip install spacy seqeval
❯ python -m spacy download es_core_news_lg
❯ python3 ../../1-spacy/predict_spacy.py -m es_core_news_lg dataset-es.iob > spacy-baseline_predictions.txt
❯ python3 ../../utils/eval.py spacy-baseline_predictions.txt dataset-es.iob
              precision    recall  f1-score   support

         LOC      0.858     0.870     0.864     18387
        MISC      0.111     0.308     0.163      1606
         ORG      0.333     0.527     0.408       438
         PER      0.455     0.780     0.574      1206

   micro avg      0.683     0.816     0.744     21637
   macro avg      0.439     0.621     0.502     21637
weighted avg      0.769     0.816     0.786     21637
```
