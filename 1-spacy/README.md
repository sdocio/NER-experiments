Config files used to train the [spaCy](https://spacy.io/) models for the NER task in Spanish. The dataset used contained interviews in the domain of tourism related to the Way of Saint Jacques.

| folder | type                        | pipeline | trained model |
| ------ | --------------------------- | ---------------------------------------------------- | ----- |
| `cnn`  | [CNNs and Bloom embeddings](https://spacy.io/universe/project/video-spacys-ner-model) | `tok2vec` (`es_core_news_lg`) | [es_spacy_ner_cds](https://huggingface.co/sdocio/es_spacy_ner_cds)
| `trf`    | transformers              | `transformer` (`dccuchile/bert-base-spanish-wwm-cased`)    | [es_spacy_ner_cds_trf](https://huggingface.co/sdocio/es_spacy_ner_cds_trf)

In both cases the dataset was converted from IOB2 using spaCy utilities and trained with `run_train.sh`.

```bash
$ mkdir dataset/{test,train}
$ python -m spacy convert -c ner train.iob dataset/train
$ python -m spacy convert -c ner test.iob dataset/test
$ python -m spacy debug data config.cfg
$ bash run_train.sh
```

The script `predict_spacy.py` produces the resulting predictions in IOB format, which can be evaluated using `eval.py` and a testing dataset in IOB2 format.

```bash
$ python3 predict_spacy.py -m cnn/model/es_spacy_ner_cds ../datasets/test.iob > predictions.iob
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15178/15178 [03:52<00:00, 65.37it/s]
$ python ../utils/eval.py predictions.iob ../datasets/test.iob
              precision    recall  f1-score   support

         LOC      0.975     0.987     0.981      3626
        MISC      0.854     0.757     0.803       325
         ORG      0.869     0.688     0.768        77
         PER      0.942     0.890     0.915       237

   micro avg      0.963     0.958     0.961      4265
   macro avg      0.910     0.831     0.867      4265
weighted avg      0.962     0.958     0.960      4265
```

The dataset used to train both models will be released once personal data is fully anonymised.
