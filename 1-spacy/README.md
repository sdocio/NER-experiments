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

Models can be evaluated using eval.py and a testing dataset in IOB2 format.

```bash
$ python eval.py -m es_spacy_ner_cds_trf test.iob
```

The dataset used to train both models will be released once personal data is fully anonymised.
