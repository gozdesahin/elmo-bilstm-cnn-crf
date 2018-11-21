#!/bin/bash
echo 'Clean Training Performed'
DATASET='conll2000_data/clean'
RESULT='selmo_5e_ct_results'
MODEL='selmo_5e_ct_pos_models'

# (0) Remove previous caches and embeddings
#echo "Remove cached embeddings"
#rm embeddings/elmo_cache_conll2000_data_clean.pkl
#rm -r pkl/*

# (1) Create cache for ELMO embeddings for clean data
# Do it 10 times
echo "Test on p=0,1 simple"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
 -testFile 'data/conll2000_data/perturbed/01/test_simple.txt' \
 -testSetting 'CTP01_simple_'$id'.txt' \
 -model_save $MODEL \
 -result_save $RESULT \
 -cuda_device 0 \
 -task 'pos'

