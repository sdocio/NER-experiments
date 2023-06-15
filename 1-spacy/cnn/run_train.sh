#!/bin/bash

NAME="spacy-cnn"
BASEDIR="/path/to/NER-experiments"
UTILSDIR="${BASEDIR}/utils"
TEST_FILE="${BASEDIR}/datasets/test.iob"
CONFIGFILE="${BASEDIR}/1-spacy/cnn/config.cfg"
OUTPUT_DIR="${BASEDIR}/1-spacy/cnn/model"
GPU_ID=0

sbatch <<EOT
#!/bin/sh

#SBATCH -J ${NAME}
#SBATCH --mem=24gb
#SBATCH --time=05:00:00
#SBATCH --output=${NAME}_%j.log
#SBATCH -e ${NAME}_%j_error.log
#SBATCH --gres=gpu:A100_40:1

echo "Starting training: ${NAME}"
time python3 -m spacy train ${CONFIGFILE} --gpu-id ${GPU_ID} --output ${OUTPUT_DIR}

echo "Producing predictions to results/${NAME}_predictions.txt"
time python ../predict_spacy.py -m ${OUTPUT_DIR} -m ${OUTPUT_DIR}/model-best ${TEST_FILE} > results/${NAME}_predictions.txt

echo "Saving evaluation results to results/results_${NAME}.csv"
time python ${UTILSDIR}/eval.py -o csv results/${NAME}_predictions.txt ${TEST_FILE} > results/results_${NAME}.csv

EOT
