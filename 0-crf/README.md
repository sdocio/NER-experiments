# Conditional Random Fields

The dataset is expected to be in IOB2 format, with 2 columns separated with one blank space.

**Example**

```
Me O
desvié O
un O
día O
hacia O
O B-LOC
Pindo I-LOC
. O

Fue O
antes O
de O
llegar O
a O
Sigüeiro B-LOC
. O
```

If you want to use PoS tags as feature as well, they must be included in the second column.

**Example**

```
Me PRON O
desvié VERB O
un DET O
día NOUN O
hacia ADP O
O PROPN B-LOC
Pindo PROPN I-LOC
. PUNCT O

Fue VERB O
antes ADV O
de ADP O
llegar VERB O
a ADP O
Sigüeiro PROPN B-LOC
. PUNCT O
```

## Training

The dataset is split 80-20% for training and test.

```
❯ mv sample_config.py config.py
❯ python train_crf.py dataset.iob -v -o crf.model
```

For training a model adding PoS tags as feature, option `-p` must be used.

```python train_crf.py dataset-with-pos.iob -p -v -o crf-pos.model```

## Test

```❯ python test_crf.py -m crf.model dataset.iob```

## Use the generated model

You can use the generated model with new texts. `predict_crf.py` can process a text file, and it relies on `spacy` for the tokenization.

**Example**

```
Mi amigo Juan visitó la ciudad de Santiago este año. El Ministerio de Industria y Competitividad financia el proyecto.
```

```
❯ pip install spacy
❯ python -m spacy download es_core_news_lg
❯ python3 predict_crf.py -t -m crf-model -s es_core_news_lg test.txt
```

***Results***

```
Mi O
amigo O
Juan B-PER
visitó O
la O
ciudad O
de O
Santiago B-LOC
este O
año O
. O

El O
Ministerio B-ORG
de I-ORG
Industria I-ORG
y I-ORG
Competitividad I-ORG
financia O
el O
proyecto O
. O
```

If you are using a model that takes into account PoS tags as features, you should use the option `with-pos`.

```
❯ echo "La escritora María Oruña firmará libros este jueves en Santiago.
Paulo Coelho escribió El diario de un mago." > test.txt
❯ python3 predict_crf.py -t --with-pos -m es_scq-ner_crf_sm_withpos -s es_core_news_lg test.txt
La O
escritora O
María B-PER
Oruña I-PER
firmará O
libros O
este O
jueves O
en O
Santiago B-LOC
. O

Paulo B-PER
Coelho I-PER
escribió O
El B-MISC
diario I-MISC
de I-MISC
un I-MISC
mago I-MISC
. O
```

The script can also be used with tokenized text, in IOB2 format. In this case, labels present in the IOB2 format are not taken into account.

```
❯ echo "Paulo O
Coelho O
escribió O
El O
diario O
de O
un O
mago O
. O" > test.iob

❯ python3 predict_crf.py -i -m es_scq-ner_crf_sm_tuned text.iob
Paulo B-PER
Coelho I-PER
escribió O
El B-MISC
diario I-MISC
de I-MISC
un I-MISC
mago I-MISC
. O
```
