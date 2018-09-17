#!/bin/bash
echo 'Clean Training Performed'
DATASET='conll2000_data/clean'
# (0) Remove previous caches and embeddings
echo "Remove cached embeddings"
rm embeddings/velmo_cache_conll2000_data_clean.pkl
rm -r pkl/

echo "Create new cached embeddings"
# (1) Create cache for VELMO embeddings for clean data
python Create_ELMo_Cache.py -datasetName $DATASET \
-tokenColumnId 0 \
-cuda_device 0 \
-elmo_options 'pretrained/velmo_options.json' \
-elmo_weights 'pretrained/velmo_weights.hdf5' \
-pkl_path 'embeddings/velmo_cache_conll2000_data_clean.pkl'

mkdir results
mkdir results/pos
mkdir pkl/conll2000_data
# Do it 5 times
for RUN in {1..5}
do
	echo "Train a POS tagger"
	# (2) Train and test a POS tagger
	python Train_POS.py -datasetName $DATASET \
	-tokenColumnId 0 \
	-cuda_device 0 \
	-elmo_options 'pretrained/velmo_options.json' \
	-elmo_weights 'pretrained/velmo_weights.hdf5' \
	-pkl_path 'embeddings/velmo_cache_conll2000_data_clean.pkl'

	echo "Test on clean data"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/clean/test.txt' \
	-testSetting 'CTCT_'$RUN'.txt' \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,1"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test.txt' \
	-testSetting 'CTP01_'$RUN'.txt' \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,2"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test.txt' \
	-testSetting 'CTP02_'$RUN'.txt' \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,3"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test.txt' \
	-testSetting 'CTP03_'$RUN'.txt' \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,4"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test.txt' \
	-testSetting 'CTP04_'$RUN'.txt' \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,5"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test.txt' \
	-testSetting 'CTP05_'$RUN'.txt' \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,6"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test.txt' \
	-testSetting 'CTP06_'$RUN'.txt' \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,7"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test.txt' \
	-testSetting 'CTP07_'$RUN'.txt' \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,8"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test.txt' \
	-testSetting 'CTP08_'$RUN'.txt' \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,9"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test.txt' \
	-testSetting 'CTP09_'$RUN'.txt' \
	-cuda_device 0 \
	-task 'pos'
done 
echo "Finished"