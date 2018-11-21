#!/bin/bash
echo 'POS Tagging'
echo 'Adversarial Training for Random Perturbation - SELMO'
DATASET='conll2000_data/perturbed/random'
RESULT='selmo_pos_random_results'
MODEL='selmo_pos_random_models'

# (0) Remove previous caches and embeddings
#echo "Remove cached embeddings"
rm embeddings/elmo_cache_conll2000_data_perturbed_random.pkl
rm -r pkl/conll2000_data/perturbed/random

echo "Create new cached embeddings"
# (1) Create cache for VELMO embeddings for clean data
python Create_ELMo_Cache.py -datasetName $DATASET \
-tokenColumnId 0 \
-cuda_device 0 \
-elmo_options 'pretrained/selmo.30k.5ep.options.json' \
-elmo_weights 'pretrained/selmo.30k.5ep.weights.hdf5' \
-pkl_path 'embeddings/elmo_cache_conll2000_data_perturbed_random.pkl'

mkdir $RESULT
mkdir $RESULT/pos
mkdir pkl/conll2000_data
mkdir pkl/conll2000_data/perturbed
mkdir pkl/conll2000_data/perturbed/random
mkdir $MODEL
mkdir $MODEL/conll2000_data
mkdir $MODEL/conll2000_data/perturbed
mkdir $MODEL/conll2000_data/perturbed/random

# Do it 1 time
for id in 1
do
    # remove previous models for training
    rm $MODEL/conll2000_data/perturbed/*
	echo "Train a POS Tagger"
	# (2) Train and test a POS tagger
	python Train_POS.py -datasetName $DATASET \
	-tokenColumnId 0 \
	-cuda_device 1 \
	-elmo_options 'pretrained/selmo.30k.5ep.options.json' \
	-elmo_weights 'pretrained/selmo.30k.5ep.weights.hdf5' \
	-model_save $MODEL \
	-pkl_path 'embeddings/elmo_cache_conll2000_data_perturbed_random.pkl'

	echo "Test on clean data"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/clean/test.txt' \
	-testSetting 'ATCT_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,1 original"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_org.txt' \
	-testSetting 'ATP01_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,1 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_ood.txt' \
	-testSetting 'ATP01_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,1 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_odd.txt' \
	-testSetting 'ATP01_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,1 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_simple.txt' \
	-testSetting 'ATP01_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,2 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_org.txt' \
	-testSetting 'ATP02_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,2 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_ood.txt' \
	-testSetting 'ATP02_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,2 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_odd.txt' \
	-testSetting 'ATP02_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,2 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_simple.txt' \
	-testSetting 'ATP02_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,3 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_org.txt' \
	-testSetting 'ATP03_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,3 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_ood.txt' \
	-testSetting 'ATP03_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,3 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_odd.txt' \
	-testSetting 'ATP03_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,3 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_simple.txt' \
	-testSetting 'ATP03_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,4 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_org.txt' \
	-testSetting 'ATP04_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,4 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_ood.txt' \
	-testSetting 'ATP04_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,4 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_odd.txt' \
	-testSetting 'ATP04_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,4 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_simple.txt' \
	-testSetting 'ATP04_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,5 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_org.txt' \
	-testSetting 'ATP05_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'
	
	echo "Test on p=0,5 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_ood.txt' \
	-testSetting 'ATP05_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,5 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_odd.txt' \
	-testSetting 'ATP05_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,5 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_simple.txt' \
	-testSetting 'ATP05_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,6 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_org.txt' \
	-testSetting 'ATP06_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,6 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_ood.txt' \
	-testSetting 'ATP06_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,6 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_odd.txt' \
	-testSetting 'ATP06_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,6 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_simple.txt' \
	-testSetting 'ATP06_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,7 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_org.txt' \
	-testSetting 'ATP07_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,7 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_ood.txt' \
	-testSetting 'ATP07_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,7 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_odd.txt' \
	-testSetting 'ATP07_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,7 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_simple.txt' \
	-testSetting 'ATP07_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,8 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_org.txt' \
	-testSetting 'ATP08_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,8 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_ood.txt' \
	-testSetting 'ATP08_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,8 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_odd.txt' \
	-testSetting 'ATP08_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,8 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_simple.txt' \
	-testSetting 'ATP08_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,9 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_org.txt' \
	-testSetting 'ATP09_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,9 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_ood.txt' \
	-testSetting 'ATP09_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,9 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_odd.txt' \
	-testSetting 'ATP09_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,9 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_simple.txt' \
	-testSetting 'ATP09_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

done 
echo "Finished SELMO POS Random"


echo 'Adversarial Training Performed POS Random VELMO'
DATASET='conll2000_data/perturbed/random'
RESULT='velmo_pos_random_results'
MODEL='velmo_pos_random_models'

# (0) Remove previous caches and embeddings
#echo "Remove cached embeddings"
rm embeddings/velmo_cache_conll2000_data_perturbed_random.pkl
rm -r pkl/conll2000_data/perturbed

echo "Create new cached embeddings"
# (1) Create cache for VELMO embeddings for clean data
python Create_ELMo_Cache.py -datasetName $DATASET \
-tokenColumnId 0 \
-cuda_device 0 \
-elmo_options 'pretrained/velmo_options.json' \
-elmo_weights 'pretrained/velmo_weights.hdf5' \
-pkl_path 'embeddings/velmo_cache_conll2000_data_perturbed_random.pkl'

mkdir $RESULT
mkdir $RESULT/pos
mkdir pkl/conll2000_data
mkdir pkl/conll2000_data/perturbed
mkdir pkl/conll2000_data/perturbed/random
mkdir $MODEL
mkdir $MODEL/conll2000_data
mkdir $MODEL/conll2000_data/perturbed
mkdir $MODEL/conll2000_data/perturbed/random

# Do it 10 times
for id in 1
do
    # remove previous models for clean training
    rm $MODEL/conll2000_data/perturbed/*
	echo "Train a POS tagger"
	# (2) Train and test a POS tagger
	python Train_POS.py -datasetName $DATASET \
	-tokenColumnId 0 \
	-cuda_device 1 \
    -elmo_options 'pretrained/velmo_options.json' \
    -elmo_weights 'pretrained/velmo_weights.hdf5' \
	-model_save $MODEL \
	-pkl_path 'embeddings/velmo_cache_conll2000_data_perturbed_random.pkl'

	echo "Test on clean data"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/clean/test.txt' \
	-testSetting 'ATCT_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,1 original"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_org.txt' \
	-testSetting 'ATP01_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,1 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_ood.txt' \
	-testSetting 'ATP01_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,1 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_odd.txt' \
	-testSetting 'ATP01_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,1 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_simple.txt' \
	-testSetting 'ATP01_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,2 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_org.txt' \
	-testSetting 'ATP02_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,2 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_ood.txt' \
	-testSetting 'ATP02_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,2 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_odd.txt' \
	-testSetting 'ATP02_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,2 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_simple.txt' \
	-testSetting 'ATP02_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,3 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_org.txt' \
	-testSetting 'ATP03_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,3 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_ood.txt' \
	-testSetting 'ATP03_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,3 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_odd.txt' \
	-testSetting 'ATP03_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,3 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_simple.txt' \
	-testSetting 'ATP03_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,4 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_org.txt' \
	-testSetting 'ATP04_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,4 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_ood.txt' \
	-testSetting 'ATP04_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,4 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_odd.txt' \
	-testSetting 'ATP04_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,4 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_simple.txt' \
	-testSetting 'ATP04_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,5 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_org.txt' \
	-testSetting 'ATP05_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'
	
	echo "Test on p=0,5 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_ood.txt' \
	-testSetting 'ATP05_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,5 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_odd.txt' \
	-testSetting 'ATP05_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,5 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_simple.txt' \
	-testSetting 'ATP05_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,6 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_org.txt' \
	-testSetting 'ATP06_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,6 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_ood.txt' \
	-testSetting 'ATP06_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,6 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_odd.txt' \
	-testSetting 'ATP06_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,6 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_simple.txt' \
	-testSetting 'ATP06_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,7 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_org.txt' \
	-testSetting 'ATP07_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,7 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_ood.txt' \
	-testSetting 'ATP07_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,7 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_odd.txt' \
	-testSetting 'ATP07_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,7 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_simple.txt' \
	-testSetting 'ATP07_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,8 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_org.txt' \
	-testSetting 'ATP08_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,8 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_ood.txt' \
	-testSetting 'ATP08_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,8 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_odd.txt' \
	-testSetting 'ATP08_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,8 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_simple.txt' \
	-testSetting 'ATP08_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,9 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_org.txt' \
	-testSetting 'ATP09_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,9 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_ood.txt' \
	-testSetting 'ATP09_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,9 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_odd.txt' \
	-testSetting 'ATP09_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'

	echo "Test on p=0,9 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_simple.txt' \
	-testSetting 'ATP09_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'pos'
done 


echo 'Chunking'
echo 'Adversarial Training for Random Perturbation - SELMO'
DATASET='conll2000_data/perturbed/random'
RESULT='selmo_chunk_random_results'
MODEL='selmo_chunk_random_models'

# (0) Remove previous caches and embeddings
#echo "Remove cached embeddings"
rm embeddings/elmo_cache_conll2000_data_perturbed_random.pkl
rm -r pkl/conll2000_data/perturbed/random

echo "Create new cached embeddings"
# (1) Create cache for VELMO embeddings for clean data
python Create_ELMo_Cache.py -datasetName $DATASET \
-tokenColumnId 0 \
-cuda_device 0 \
-elmo_options 'pretrained/selmo.30k.5ep.options.json' \
-elmo_weights 'pretrained/selmo.30k.5ep.weights.hdf5' \
-pkl_path 'embeddings/elmo_cache_conll2000_data_perturbed_random.pkl'

mkdir $RESULT
mkdir $RESULT/chunking
mkdir pkl/conll2000_data
mkdir pkl/conll2000_data/perturbed
mkdir pkl/conll2000_data/perturbed/random
mkdir $MODEL
mkdir $MODEL/conll2000_data
mkdir $MODEL/conll2000_data/perturbed
mkdir $MODEL/conll2000_data/perturbed/random

# Do it 1 time
for id in 1
do
    # remove previous models for training
    rm $MODEL/conll2000_data/perturbed/*
	echo "Train Chunking"
	# (2) Train and test a POS tagger
	python Train_Chunking.py -datasetName $DATASET \
	-tokenColumnId 0 \
	-cuda_device 1 \
	-elmo_options 'pretrained/selmo.30k.5ep.options.json' \
	-elmo_weights 'pretrained/selmo.30k.5ep.weights.hdf5' \
	-model_save $MODEL \
	-pkl_path 'embeddings/elmo_cache_conll2000_data_perturbed_random.pkl'

	echo "Test on clean data"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/clean/test.txt' \
	-testSetting 'ATCT_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 original"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_org.txt' \
	-testSetting 'ATP01_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_ood.txt' \
	-testSetting 'ATP01_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_odd.txt' \
	-testSetting 'ATP01_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_simple.txt' \
	-testSetting 'ATP01_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_org.txt' \
	-testSetting 'ATP02_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_ood.txt' \
	-testSetting 'ATP02_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_odd.txt' \
	-testSetting 'ATP02_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_simple.txt' \
	-testSetting 'ATP02_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_org.txt' \
	-testSetting 'ATP03_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_ood.txt' \
	-testSetting 'ATP03_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_odd.txt' \
	-testSetting 'ATP03_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_simple.txt' \
	-testSetting 'ATP03_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_org.txt' \
	-testSetting 'ATP04_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_ood.txt' \
	-testSetting 'ATP04_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_odd.txt' \
	-testSetting 'ATP04_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_simple.txt' \
	-testSetting 'ATP04_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,5 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_org.txt' \
	-testSetting 'ATP05_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'
	
	echo "Test on p=0,5 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_ood.txt' \
	-testSetting 'ATP05_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,5 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_odd.txt' \
	-testSetting 'ATP05_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,5 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_simple.txt' \
	-testSetting 'ATP05_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_org.txt' \
	-testSetting 'ATP06_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_ood.txt' \
	-testSetting 'ATP06_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_odd.txt' \
	-testSetting 'ATP06_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_simple.txt' \
	-testSetting 'ATP06_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_org.txt' \
	-testSetting 'ATP07_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_ood.txt' \
	-testSetting 'ATP07_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_odd.txt' \
	-testSetting 'ATP07_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_simple.txt' \
	-testSetting 'ATP07_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_org.txt' \
	-testSetting 'ATP08_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_ood.txt' \
	-testSetting 'ATP08_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_odd.txt' \
	-testSetting 'ATP08_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_simple.txt' \
	-testSetting 'ATP08_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_org.txt' \
	-testSetting 'ATP09_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_ood.txt' \
	-testSetting 'ATP09_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_odd.txt' \
	-testSetting 'ATP09_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_simple.txt' \
	-testSetting 'ATP09_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

done 
echo "Finished SELMO Chunking Random"


echo 'Adversarial Training Performed Chunking Random VELMO'
DATASET='conll2000_data/perturbed/random'
RESULT='velmo_chunk_random_results'
MODEL='velmo_chunk_random_models'

# (0) Remove previous caches and embeddings
#echo "Remove cached embeddings"
rm embeddings/velmo_cache_conll2000_data_perturbed_random.pkl
rm -r pkl/conll2000_data/perturbed

echo "Create new cached embeddings"
# (1) Create cache for VELMO embeddings for clean data
python Create_ELMo_Cache.py -datasetName $DATASET \
-tokenColumnId 0 \
-cuda_device 0 \
-elmo_options 'pretrained/velmo_options.json' \
-elmo_weights 'pretrained/velmo_weights.hdf5' \
-pkl_path 'embeddings/velmo_cache_conll2000_data_perturbed_random.pkl'

mkdir $RESULT
mkdir $RESULT/chunking
mkdir pkl/conll2000_data
mkdir pkl/conll2000_data/perturbed
mkdir pkl/conll2000_data/perturbed/random
mkdir $MODEL
mkdir $MODEL/conll2000_data
mkdir $MODEL/conll2000_data/perturbed
mkdir $MODEL/conll2000_data/perturbed/random

# Do it 10 times
for id in 1
do
    # remove previous models for clean training
    rm $MODEL/conll2000_data/perturbed/*
	echo "Train Chunking"
	# (2) Train and test a POS tagger
	python Train_Chunking.py -datasetName $DATASET \
	-tokenColumnId 0 \
	-cuda_device 1 \
    -elmo_options 'pretrained/velmo_options.json' \
    -elmo_weights 'pretrained/velmo_weights.hdf5' \
	-model_save $MODEL \
	-pkl_path 'embeddings/velmo_cache_conll2000_data_perturbed_random.pkl'

	echo "Test on clean data"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/clean/test.txt' \
	-testSetting 'ATCT_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 original"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_org.txt' \
	-testSetting 'ATP01_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_ood.txt' \
	-testSetting 'ATP01_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_odd.txt' \
	-testSetting 'ATP01_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,1 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_simple.txt' \
	-testSetting 'ATP01_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_org.txt' \
	-testSetting 'ATP02_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_ood.txt' \
	-testSetting 'ATP02_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_odd.txt' \
	-testSetting 'ATP02_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,2 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_simple.txt' \
	-testSetting 'ATP02_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_org.txt' \
	-testSetting 'ATP03_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_ood.txt' \
	-testSetting 'ATP03_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_odd.txt' \
	-testSetting 'ATP03_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,3 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_simple.txt' \
	-testSetting 'ATP03_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_org.txt' \
	-testSetting 'ATP04_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_ood.txt' \
	-testSetting 'ATP04_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_odd.txt' \
	-testSetting 'ATP04_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,4 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_simple.txt' \
	-testSetting 'ATP04_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,5 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_org.txt' \
	-testSetting 'ATP05_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'
	
	echo "Test on p=0,5 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_ood.txt' \
	-testSetting 'ATP05_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,5 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_odd.txt' \
	-testSetting 'ATP05_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,5 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_simple.txt' \
	-testSetting 'ATP05_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_org.txt' \
	-testSetting 'ATP06_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_ood.txt' \
	-testSetting 'ATP06_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_odd.txt' \
	-testSetting 'ATP06_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,6 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_simple.txt' \
	-testSetting 'ATP06_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_org.txt' \
	-testSetting 'ATP07_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_ood.txt' \
	-testSetting 'ATP07_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_odd.txt' \
	-testSetting 'ATP07_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,7 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_simple.txt' \
	-testSetting 'ATP07_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_org.txt' \
	-testSetting 'ATP08_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_ood.txt' \
	-testSetting 'ATP08_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_odd.txt' \
	-testSetting 'ATP08_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,8 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_simple.txt' \
	-testSetting 'ATP08_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_org.txt' \
	-testSetting 'ATP09_org_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_ood.txt' \
	-testSetting 'ATP09_ood_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 odd"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_odd.txt' \
	-testSetting 'ATP09_odd_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

	echo "Test on p=0,9 simple"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_simple.txt' \
	-testSetting 'ATP09_simple_'$id'.txt' \
	-model_save $MODEL \
	-result_save $RESULT \
	-cuda_device 0 \
	-task 'chunking'

done 



echo "Finished"