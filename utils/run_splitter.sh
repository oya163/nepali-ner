#! /usr/bin/env bash

set -e
set -f

: '
    Bash script to run split datasets into k-fold
    
    How to run:
        bash utils/run_splitter.sh <k-fold> <output_dir_name>
        bash utils/run_splitter.sh 5 kfold
'

KFOLD=$1
DIRNAME=$2
SCRIPT=${NEPALI_POS}/utils/splitter.py
DATA_DIR=${NEPALI_POS}/data
DATA_FILE=${DATA_DIR}/dataset.txt
OUT_DIR=${DATA_DIR}/$DIRNAME


if [ -d "$OUT_DIR" ]
then
    echo "Directory exists. Deleting..."
    rm -rf ${OUT_DIR}
fi

echo "Creating directory ${OUT_DIR}"
mkdir ${OUT_DIR}


${SCRIPT} -i ${DATA_FILE} -o ${OUT_DIR} -k ${KFOLD} -v

exit
# eof
