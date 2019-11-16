#!/bin/bash
echo
echo **********************TRAIN EVALUATION**********************
./conlleval -r < ../results/lstm_1_train.txt
echo
echo **********************TEST EVALUATION**********************
./conlleval -r < ../results/lstm_1_test.txt
echo
echo **********************VAL EVALUATION**********************
./conlleval -r < ../results/lstm_1_val.txt
