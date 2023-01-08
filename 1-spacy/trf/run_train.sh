#!/bin/bash

BASEDIR="/path/to/NER-experiments/1-spacy"
CONFIGFILE="${BASEDIR}/trf/config.cfg"
OUTPUT_DIR="${BASEDIR}/trf/model"
GPU_ID=0


time python3 -m spacy train ${CONFIGFILE} --gpu-id ${GPU_ID} --output ${OUTPUT_DIR}
