#!/bin/bash
echo 'Clean Training Performed'
DATASET='conll2000_data/clean'
RESULT='velmo_5e_ct_results'
MODEL='velmo_5e_ct_chunk_models'

# (0) Remove previous caches and embeddings
#echo "Remove cached embeddings"
#rm embeddings/elmo_cache_conll2000_data_clean.pkl
#rm -r pkl/*

echo "Create new cached embeddings"
# (1) Create cache for VELMO embeddings for clean data
python Create_ELMo_Cache.py -datasetName $DATASET \
-tokenColumnId 0 \
-cuda_device 0 \
-elmo_options 'pretrained/velmo_options.json' \
-elmo_weights 'pretrained/velmo_weights.hdf5' \
-pkl_path 'embeddings/velmo_cache_conll2000_data_clean.pkl'

mkdir $RESULT
mkdir $RESULT/chunking
mkdir pkl/conll2000_data

# Do it 10 times
for id in 1 2 3 4 5 6 7 8 9 10
do
    # remove previous models for clean training
    rm $MODEL/conll2000_data/*
	echo "Train a Chunker"
	# (2) Train and test a POS tagger
	python Train_Chunking.py -datasetName $DATASET \
	-tokenColumnId 0 \
	-cuda_device 0 \
    -elmo_options 'pretrained/velmo_options.json' \
    -elmo_weights 'pretrained/velmo_weights.hdf5' \
	-model_save $MODEL \
	-pkl_path 'embeddings/velmo_cache_conll2000_data_clean.pkl'

	echo "Test on clean data"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/clean/test.txt' \
	-testSetting 'CTCT_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 original"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_org.txt' \
	-testSetting 'CTP01_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_ood.txt' \
	-testSetting 'CTP01_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_odd.txt' \
	-testSetting 'CTP01_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_simple.txt' \
	-testSetting 'CTP01_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_org.txt' \
	-testSetting 'CTP02_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_ood.txt' \
	-testSetting 'CTP02_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_odd.txt' \
	-testSetting 'CTP02_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_simple.txt' \
	-testSetting 'CTP02_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_org.txt' \
	-testSetting 'CTP03_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_ood.txt' \
	-testSetting 'CTP03_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_odd.txt' \
	-testSetting 'CTP03_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_simple.txt' \
	-testSetting 'CTP03_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_org.txt' \
	-testSetting 'CTP04_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_ood.txt' \
	-testSetting 'CTP04_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_odd.txt' \
	-testSetting 'CTP04_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_simple.txt' \
	-testSetting 'CTP04_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,5 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_org.txt' \
	-testSetting 'CTP05_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'
	
	echo "Test on p=0,5 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_ood.txt' \
	-testSetting 'CTP05_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,5 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_odd.txt' \
	-testSetting 'CTP05_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,5 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_simple.txt' \
	-testSetting 'CTP05_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_org.txt' \
	-testSetting 'CTP06_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_ood.txt' \
	-testSetting 'CTP06_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_odd.txt' \
	-testSetting 'CTP06_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_simple.txt' \
	-testSetting 'CTP06_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_org.txt' \
	-testSetting 'CTP07_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_ood.txt' \
	-testSetting 'CTP07_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_odd.txt' \
	-testSetting 'CTP07_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_simple.txt' \
	-testSetting 'CTP07_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_org.txt' \
	-testSetting 'CTP08_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_ood.txt' \
	-testSetting 'CTP08_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_odd.txt' \
	-testSetting 'CTP08_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_simple.txt' \
	-testSetting 'CTP08_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_org.txt' \
	-testSetting 'CTP09_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_ood.txt' \
	-testSetting 'CTP09_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_odd.txt' \
	-testSetting 'CTP09_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_simple.txt' \
	-testSetting 'CTP09_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

done 
echo "Finished"