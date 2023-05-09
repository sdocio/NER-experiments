#!/bin/bash

##### defs ################################################################
MAX_LENGTH=128
BATCH_SIZE=32
NUM_EPOCHS=3
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=100
SEED=42
LOAD_BEST_MODEL_AT_END=True
EVAL_STRATEGY="no"
SAVE_STRATEGY="no"
TRAIN_FILE="dataset/train.json"
VALIDATION_FILE="dataset/validation.json"
###########################################################################



##### config ##############################################################
declare -A models=([bne-base]="PlanTL-GOB-ES/roberta-base-bne" [xlm-base]="xlm-roberta-base" [xlm-large]="xlm-roberta-large" [bne-large]="PlanTL-GOB-ES/roberta-large-bne" [spanberta]="chriskhanhtran/spanberta" [bert]="dccuchile/bert-base-spanish-wwm-cased")
declare -A output=([bne-base]="models/es_trf_ner_cds_bne-base" [xlm-base]="models/es_trf_ner_cds_xlm-base" [xlm-large]="models/es_trf_ner_cds_xlm-large" [bne-large]="models/es_trf_ner_cds_bne-large" [spanberta]="models/es_trf_ner_cds_spanberta" [bert]="models/es_trf_ner_cds_bert")

MODEL=$*
BASEDIR="/path/to/NER-experiments"
UTILSDIR="${BASEDIR}/utils"
TEST_FILE="dataset/test.json"
TEST_FILE_IOB="${BASEDIR}/datasets/test.iob"
###########################################################################

##### functions ###########################################################

die()
{
    echo $*
    exit 0
}

is_valid_file()
{
    [[ -f ${ifile} ]] && return 1
    return 0
}

is_valid_model()
{
    for validmodel in "${!models[@]}"
    do
        [[ ${MODEL} == ${validmodel} ]] && return 1
    done

    return 0
}

###########################################################################


[ -z "${MODEL}" ] && echo "must supply an argument" && exit 1

OUTPUT_DIR=${output[${MODEL}]}
TRANSFORMERS_MODEL=${models[${MODEL}]}

sbatch <<EOT
#!/bin/sh

#SBATCH -J ${MODEL}
#SBATCH --mem=24gb
#SBATCH --time=05:00:00
#SBATCH --output=trf-${MODEL}_%j.log
#SBATCH -e trf-${MODEL}_%j_error.log
#SBATCH --gres=gpu:A100_40:1

echo "Starting training: ${MODEL}"
time python3 run_ner.py \
  --model_name_or_path ${TRANSFORMERS_MODEL} \
  --train_file ${TRAIN_FILE} \
  --validation_file ${VALIDATION_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --max_seq_length ${MAX_LENGTH} \
  --num_train_epochs ${NUM_EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --logging_steps ${LOGGING_STEPS} \
  --seed ${SEED} \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --load_best_model_at_end ${LOAD_BEST_MODEL_AT_END} \
  --evaluation_strategy ${EVAL_STRATEGY} \
  --save_strategy ${SAVE_STRATEGY}

echo "Producing predictions to results/trf-${MODEL}_predictions.txt"
time python predict_transformers.py --model ${OUTPUT_DIR} --output_file results/trf-${MODEL}_predictions.txt --test_file ${TEST_FILE} --train_file ${TRAIN_FILE}

echo "Saving evaluation results to results/results_trf-${MODEL}.csv"
time python ${UTILSDIR}/eval.py -o csv results/trf-${MODEL}_predictions.txt ${TEST_FILE_IOB} > results/results_trf-${MODEL}.csv

EOT
