#!/bin/bash
echo 'Clean Training Performed'
DATASET='conll2000_data/clean'
# (0) Remove previous caches and embeddings
echo "Remove cached embeddings"
rm embeddings/velmo_cache_conll2000_data_perturbed_02.pkl
rm -r pkl/conll2000_data/perturbed

echo "Create new cached embeddings"
# (1) Create cache for VELMO embeddings for clean data
python Create_ELMo_Cache.py -datasetName $DATASET \
-tokenColumnId 0 \
-cuda_device 1 \
-elmo_options 'pretrained/velmo_options.json' \
-elmo_weights 'pretrained/velmo_weights.hdf5' \
-pkl_path 'embeddings/velmo_cache_conll2000_data_perturbed_02.pkl'

mkdir results
mkdir results/chunking
mkdir pkl/conll2000_data
mkdir pkl/conll2000_data/perturbed
mkdir pkl/conll2000_data/perturbed/02
mkdir models/conll2000_data/perturbed
mkdir models/conll2000_data/perturbed/02

# Do it 10 times
for id in 1 2 3 4 5 6 7 8 9 10
do
    # remove previous models for clean training
    rm models/conll2000_data/perturbed/*
	echo "Train a Chunker"
	# (2) Train and test a POS tagger
	python Train_Chunking.py -datasetName $DATASET \
	-tokenColumnId 0 \
	-cuda_device 1 \
	-elmo_options 'pretrained/velmo_options.json' \
	-elmo_weights 'pretrained/velmo_weights.hdf5' \
	-pkl_path 'embeddings/velmo_cache_conll2000_data_perturbed_02.pkl'

	echo "Test on clean data"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/clean/test.txt' \
	-testSetting 'ATCT_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,1 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_org.txt' \
	-testSetting 'ATP01_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,1 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_ood.txt' \
	-testSetting 'ATP01_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,2 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_org.txt' \
	-testSetting 'ATP02_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,2 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_ood.txt' \
	-testSetting 'ATP02_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,3 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_org.txt' \
	-testSetting 'ATP03_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,3 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_ood.txt' \
	-testSetting 'ATP03_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,4 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_org.txt' \
	-testSetting 'ATP04_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,4 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_ood.txt' \
	-testSetting 'ATP04_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,5 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_org.txt' \
	-testSetting 'ATP05_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'
	
	echo "Test on p=0,5 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_ood.txt' \
	-testSetting 'ATP05_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,6 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_org.txt' \
	-testSetting 'ATP06_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,6 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_ood.txt' \
	-testSetting 'ATP06_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,7 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_org.txt' \
	-testSetting 'ATP07_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,7 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_ood.txt' \
	-testSetting 'ATP07_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,8 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_org.txt' \
	-testSetting 'ATP08_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'


	echo "Test on p=0,8 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_ood.txt' \
	-testSetting 'ATP08_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,9 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_org.txt' \
	-testSetting 'ATP09_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'

	echo "Test on p=0,9 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_ood.txt' \
	-testSetting 'ATP09_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'chunking'
done 
echo "Finished"