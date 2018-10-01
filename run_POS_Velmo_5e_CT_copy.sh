#!/bin/bash
echo 'Clean Training Performed'
DATASET='conll2000_data/clean'
# (0) Remove previous caches and embeddings
echo "Remove cached embeddings"
#rm embeddings/velmo_cache_conll2000_data_clean.pkl
#rm -r pkl/*

echo "Create new cached embeddings"
# (1) Create cache for VELMO embeddings for clean data
#python Create_ELMo_Cache.py -datasetName $DATASET \
#-tokenColumnId 0 \
#-cuda_device 0 \
#-elmo_options 'pretrained/velmo_options.json' \
#-elmo_weights 'pretrained/velmo_weights.hdf5' \
#-pkl_path 'embeddings/velmo_cache_conll2000_data_clean.pkl'

mkdir results
mkdir results/pos
mkdir pkl/conll2000_data
# Do it 5 times
for id in 1 2 3 4 5
do
  echo "Train a POS tagger"
	# (2) Train and test a POS tagger
	#python Train_POS.py -datasetName $DATASET \
	#-tokenColumnId 0 \
	#-cuda_device 0 \
	#-elmo_options 'pretrained/velmo_options.json' \
	#-elmo_weights 'pretrained/velmo_weights.hdf5' \
	#-pkl_path 'embeddings/velmo_cache_conll2000_data_clean.pkl'
  echo "$id"
  echo $id
  echo "Test on clean data"
  python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/clean/test.txt' \
	-testSetting 'CTCT_'$id'.txt' \
	-cuda_device 0 \
	-task 'pos'
done
echo "Finished"
