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
TRAIN_FILE="data/train.json"
VALIDATION_FILE="data/validation.json"
###########################################################################



##### config ##############################################################
declare -A models=([roberta-base-bne]="PlanTL-GOB-ES/roberta-base-bne" [xlm-roberta-large]="xlm-roberta-large" [roberta-large-bne]="PlanTL-GOB-ES/roberta-large-bne" [spanberta]="chriskhanhtran/spanberta" [bert-base-multilingual-cased]="bert-base-multilingual-cased")
declare -A output=([roberta-base-bne]="models/roberta-base-bne" [xlm-roberta-large]="models/xlm-roberta-large" [roberta-large-bne]="models/roberta-large-bne" [spanberta]="models/spanberta" [bert-base-multilingual-cased]="models/bert-base-multilingual-cased")
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

showhelp()
{
    echo "run_train.sh -m MODEL"
    echo "  -m|--model MODEL      Model to fine-tune (${!models[@]})."
    echo "  -t|--train FILE       Training corpus."
    echo "  -v|--validation FILE  Validation corpus."
    echo "  -h                    Show this help."
    exit 0
}
###########################################################################


while [[ "$#" > 0 ]]; do case $1 in
    -m|--model)
        [[ -z $2 ]] && die "With this option you must provide an argument with the model." 
        MODEL=$2 
        is_valid_model
        [ $? -ne 1 ] && die "Unsupported model. Please choose between: ${!models[@]}" 
        shift 
        shift
        ;;
    -t|--train)
        [[ -z $2 ]] && die "With this option you must provide an argument with the model." 
        ifile=$2
        is_valid_file
        [ $? -ne 1 ] && die "File '$2' does not exist."
        TRAIN_FILE=${ifile}
        shift
        shift
        ;;
    -v|--validation)
        [[ -z $2 ]] && die "With this option you must provide an argument with the model." 
        ifile=$2
        is_valid_file
        [ $? -ne 1 ] && die "File '$2' does not exist."
        VALIDATION_FILE=${ifile}
        shift
        shift
        ;;
    -h|--help)
        shift
        showhelp
        ;;
    *)
        break
esac; done

[ -z ${MODEL} ] && MODEL="roberta-base-bne"
OUTPUT_DIR=${output[${MODEL}]}

time python3 run_ner.py \
  --model_name_or_path ${MODEL} \
  --train_file ${TRAIN_FILE} \
  --validation_file ${VALIDATION_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --max_seq_length ${MAX_LENGTH} \
  --num_train_epochs ${NUM_EPOCHS} \
  --per_gpu_train_batch_size ${BATCH_SIZE} \
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
