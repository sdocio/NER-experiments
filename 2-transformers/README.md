Files used to fine-tune transformer models for the NER task in Spanish. The dataset used contained interviews in the domain of tourism related to the Way of Saint Jacques.

| base model                                | finetuned model |
| ------------------------------------ | ---------------------------------------------------- |
| `PlanTL-GOB-ES/roberta-base-bne`     | [es_trf_ner_cds_bne-base](https://huggingface.co/sdocio/es_trf_ner_cds_bne-base) |
| `xlm-roberta-large`                  | [es_trf_ner_cds_xlm-large](https://huggingface.co/sdocio/es_trf_ner_cds_xlm-large) |

The script `predict_transformers.py` produces the resulting predictions in IOB format, which can be evaluated using `eval.py` and a testing dataset in IOB2 format.

```bash
$ python predict_transformers.py --model models/es_trf_ner_cds_bne-base --output_file predictions.iob --test_file dataset/test.json --train_file dataset/train.json
$ python ../utils/eval.py predictions.iob ../datasets/test.iob
```

The dataset used to train both models will be released once personal data is fully anonymised.
