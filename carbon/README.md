# Carbon Emissions Calculator script

The `calc.pl` script is designed to calculate the carbon emissions associated to the produced models. It provides an estimate of the environmental impact of the training/fine-tuning process. Please note that this estimation does not include the emissions produced to generate the resources used, such as LLMs or word embeddings.
    
## Usage

The program reads a TSV file with two columns from the standard input. The file should contain the name of the model and the training time, which is in the format "*MM*m*SS*", where *MM* represents minutes and *SS* represents seconds.

Example input from the file "training_time.csv":

```
❯ head -2 training_time.csv                 
crf     0m52
lstm-crf-glove  203m23
```

The program generates an output TSV file with four columns: model name, training time (converted to hours), carbon emissions, and the difference from the average carbon emissions calculated for the entire list.

```
❯ cat training_time.csv | perl calc.pl
crf             0.01    0.000058        -99.93144510
lstm-crf-glove  3.39    0.283855        234.46081977
lstm-crf-fast   3.38    0.282995        233.44672075
lstm-crf-w2vec  3.40    0.285135        235.96826426
spacy-cnn       0.23    0.019563        -76.94980337
trf-bert        0.14    0.012026        -85.83002181
trf-spanberta   0.14    0.011700        -86.21373495
trf-bne-base    0.15    0.012166        -85.66557332
trf-bne-large   0.30    0.025378        -70.09778297
trf-xlm-base    0.16    0.013491        -84.10331267
trf-xlm-large   0.35    0.029565        -65.16432829
spacy-bert      0.56    0.046987        -44.63567517
spacy-spanberta 0.40    0.033124        -60.97089180
spacy-bne-base  0.53    0.044499        -47.56833990
spacy-bne-large 1.45    0.121539        43.20722636
spacy-xlm-base  1.21    0.101721        19.85554084
spacy-xlm-large 1.42    0.118981        40.19233739
Total   17.24   1.442782

Average emissions: 0.084870
Average training time: 1.01
```
