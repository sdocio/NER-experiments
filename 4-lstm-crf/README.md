This folder contains a BiLSTM-CRF model for NER in Spanish, produced with [this implementation](https://github.com/yuhaozhang/neural-ner) of the architecture proposed by [Lample et al.](https://arxiv.org/abs/1603.01360). The dataset used contained interviews in the domain of tourism related to the Way of Saint Jacques.

It can be evaluated using `predict_lstm.py` and a testing dataset in IOB2 format. It produces the resulting predictions in IOB format and the evaluation metrics.

```bash
$ python3 lstm_predict.py -m model/best_model.pt test.iob > predictions.iob
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1518/1518 [00:32<00:00, 47.24it/s]
              precision    recall  f1-score   support

         LOC      0.983     0.987     0.985      3626
        MISC      0.825     0.825     0.825       325
         ORG      0.937     0.766     0.843        77
         PER      0.952     0.911     0.931       237

   micro avg      0.969     0.967     0.968      4265
   macro avg      0.924     0.872     0.896      4265
weighted avg      0.968     0.967     0.967      4265

$ python3 ../utils/eval.py -c predictions.iob test.iob                        
LOC,0.983,0.987,0.985
MISC,0.825,0.825,0.825
ORG,0.937,0.766,0.843
PER,0.952,0.911,0.931
micro avg,0.969,0.967,0.968
macro avg,0.924,0.872,0.896
weighted avg,0.968,0.967,0.967
```

The dataset used to train both models will be released once personal data is fully anonymised.
