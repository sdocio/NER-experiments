#!/bin/bash

NAME=$*
BASEDIR="/path/to/NER-experiments"
UTILSDIR="${BASEDIR}/utils"
TEST_FILE="${BASEDIR}/datasets/test.iob"
CONFIGFILE="${BASEDIR}/1-spacy/trf/config-${NAME}.cfg"
OUTPUT_DIR="${BASEDIR}/1-spacy/trf/model"
GPU_ID=0

[ -z "${NAME}" ] && echo "must supply an argument" && exit 1

sbatch <<EOT
#!/bin/sh

#SBATCH -J ${NAME}
#SBATCH --mem=24gb
#SBATCH --time=05:00:00
#SBATCH --output=spacy-${NAME}_%j.log
#SBATCH -e spacy-${NAME}_%j_error.log
#SBATCH --gres=gpu:A100_40:1


echo "Starting training: ${NAME}"
time python3 -m spacy train ${CONFIGFILE} --gpu-id ${GPU_ID} --output ${OUTPUT_DIR}/es_spacy_ner_cds_${NAME}

echo "Producing predictions to results/spacy-${NAME}_predictions.txt"
time python ../predict_spacy.py -m ${OUTPUT_DIR}/es_spacy_ner_cds_${NAME}/model-best ${TEST_FILE} > results/spacy-${NAME}_predictions.txt

echo "Saving evaluation results to results/results_spacy-${NAME}.csv"
time python ${UTILSDIR}/eval.py -o csv results/spacy-${NAME}_predictions.txt ${TEST_FILE} > results/results_spacy-${NAME}.csv

EOT
