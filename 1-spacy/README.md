Config files used to train the [spaCy](https://spacy.io/) models for the NER task in Spanish. The dataset used contained interviews in the domain of tourism related to the Way of Saint Jacques.

| folder | type                        | pipeline | trained model |
| ------ | --------------------------- | ---------------------------------------------------- | ----- |
| `cnn`  | [CNNs and Bloom embeddings](https://spacy.io/universe/project/video-spacys-ner-model) | `tok2vec` (`es_core_news_lg`) | [es_spacy_ner_cds](https://huggingface.co/sdocio/es_spacy_ner_cds)
| `trf`    | transformers              | `transformer` (`es_dep_news_trf`)    | [es_spacy_ner_cds_trf](https://huggingface.co/sdocio/es_spacy_ner_cds_trf)

In both cases the dataset was converted from IOB2 using spaCy utilities and trained with `run_train.sh`.

```bash
$ mkdir dataset/{test,train}
$ python -m spacy convert -c ner train.iob dataset/train
$ python -m spacy convert -c ner test.iob dataset/test
$ python -m spacy debug data config.cfg
$ bash run_train.sh
```

Models can be evaluated using `predict_spacy.py` and a testing dataset in IOB2 format. It produces the resulting predictions in IOB format and the evaluation metrics.

```bash
$ python3 predict_spacy.py -m cnn/model/es_spacy_ner_cds ../datasets/test.iob > predictions.iob
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15178/15178 [03:52<00:00, 65.37it/s]
              precision    recall  f1-score   support

       B-LOC      0.977     0.989     0.983      3626
       I-LOC      0.926     0.944     0.935       808
      B-MISC      0.889     0.788     0.835       325
      I-MISC      0.777     0.721     0.748       280
       B-ORG      0.885     0.701     0.783        77
       I-ORG      0.876     0.842     0.859       101
       B-PER      0.946     0.895     0.920       237
       I-PER      0.954     0.856     0.902        97

   micro avg      0.951     0.944     0.947      5551
   macro avg      0.904     0.842     0.871      5551
weighted avg      0.949     0.944     0.946      5551

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
