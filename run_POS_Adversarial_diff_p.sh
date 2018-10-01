#!/bin/bash
echo 'Adversarial Training Performed on SELMO'
DATASET='conll2000_data/perturbed/04'
# (0) Remove previous caches and embeddings
echo "Remove cached embeddings"
rm embeddings/elmo_cache_conll2000_data_perturbed_04.pkl
rm -r pkl/conll2000_data/perturbed

echo "Create new cached embeddings"
# (1) Create cache for VELMO embeddings for clean data
python Create_ELMo_Cache.py -datasetName $DATASET \
-tokenColumnId 0 \
-cuda_device 1 \
-elmo_options 'pretrained/selmo.30k.5ep.options.json' \
-elmo_weights 'pretrained/selmo.30k.5ep.weights.hdf5' \
-pkl_path 'embeddings/elmo_cache_conll2000_data_perturbed_04.pkl'

mkdir results
mkdir results/pos
mkdir pkl/conll2000_data
mkdir pkl/conll2000_data/perturbed
mkdir pkl/conll2000_data/perturbed/04
mkdir models/conll2000_data/perturbed
mkdir models/conll2000_data/perturbed/04
# Do it 10 times
for id in 1 2 3 4 5
do
    # remove previous models for clean training
    rm models/conll2000_data/perturbed/*
	echo "Train a POS tagger"
	# (2) Train and test a POS tagger
	python Train_POS.py -datasetName $DATASET \
	-tokenColumnId 0 \
	-cuda_device 1 \
	-elmo_options 'pretrained/selmo.30k.5ep.options.json' \
	-elmo_weights 'pretrained/selmo.30k.5ep.weights.hdf5' \
	-pkl_path 'embeddings/elmo_cache_conll2000_data_perturbed_04.pkl'

	echo "Test on clean data"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/clean/test.txt' \
	-testSetting 'SATCT_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,1 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_org.txt' \
	-testSetting 'SATP01_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,1 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_ood.txt' \
	-testSetting 'SATP01_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,2 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_org.txt' \
	-testSetting 'SATP02_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,2 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_ood.txt' \
	-testSetting 'SATP02_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,3 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_org.txt' \
	-testSetting 'SATP03_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,3 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_ood.txt' \
	-testSetting 'SATP03_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,4 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_org.txt' \
	-testSetting 'SATP04_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,4 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_ood.txt' \
	-testSetting 'SATP04_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,5 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_org.txt' \
	-testSetting 'SATP05_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'
	
	echo "Test on p=0,5 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_ood.txt' \
	-testSetting 'SATP05_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,6 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_org.txt' \
	-testSetting 'SATP06_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,6 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_ood.txt' \
	-testSetting 'SATP06_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,7 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_org.txt' \
	-testSetting 'SATP07_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,7 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_ood.txt' \
	-testSetting 'SATP07_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,8 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_org.txt' \
	-testSetting 'SATP08_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'


	echo "Test on p=0,8 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_ood.txt' \
	-testSetting 'SATP08_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,9 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_org.txt' \
	-testSetting 'SATP09_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,9 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_ood.txt' \
	-testSetting 'SATP09_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'
done 
echo "Finished for SELMO"

#!/bin/bash
echo 'Adversarial Training Performed for VELMO'
DATASET='conll2000_data/perturbed/04'
# (0) Remove previous caches and embeddings
echo "Remove cached embeddings"
rm embeddings/velmo_cache_conll2000_data_perturbed_04.pkl
rm -r pkl/conll2000_data/perturbed

echo "Create new cached embeddings"
# (1) Create cache for VELMO embeddings for clean data
python Create_ELMo_Cache.py -datasetName $DATASET \
-tokenColumnId 0 \
-cuda_device 1 \
-elmo_options 'pretrained/velmo_options.json' \
-elmo_weights 'pretrained/velmo_weights.hdf5' \
-pkl_path 'embeddings/velmo_cache_conll2000_data_perturbed_04.pkl'

mkdir results
mkdir results/pos
mkdir pkl/conll2000_data
mkdir pkl/conll2000_data/perturbed
mkdir pkl/conll2000_data/perturbed/04
mkdir models/conll2000_data/perturbed
mkdir models/conll2000_data/perturbed/04
# Do it 10 times
for id in 1 2 3 4 5
do
    # remove previous models for clean training
    rm models/conll2000_data/perturbed/*
	echo "Train a POS tagger"
	# (2) Train and test a POS tagger
	python Train_POS.py -datasetName $DATASET \
	-tokenColumnId 0 \
	-cuda_device 1 \
	-elmo_options 'pretrained/velmo_options.json' \
	-elmo_weights 'pretrained/velmo_weights.hdf5' \
	-pkl_path 'embeddings/velmo_cache_conll2000_data_perturbed_04.pkl'

	echo "Test on clean data"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/clean/test.txt' \
	-testSetting 'VATCT_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,1 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_org.txt' \
	-testSetting 'VATP01_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,1 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/01/test_ood.txt' \
	-testSetting 'VATP01_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,2 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_org.txt' \
	-testSetting 'VATP02_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,2 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/02/test_ood.txt' \
	-testSetting 'VATP02_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,3 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_org.txt' \
	-testSetting 'VATP03_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,3 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/03/test_ood.txt' \
	-testSetting 'VATP03_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,4 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_org.txt' \
	-testSetting 'VATP04_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,4 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/04/test_ood.txt' \
	-testSetting 'VATP04_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,5 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_org.txt' \
	-testSetting 'VATP05_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'
	
	echo "Test on p=0,5 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/05/test_ood.txt' \
	-testSetting 'VATP05_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,6 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_org.txt' \
	-testSetting 'VATP06_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,6 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/06/test_ood.txt' \
	-testSetting 'VATP06_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,7 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_org.txt' \
	-testSetting 'VATP07_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,7 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/07/test_ood.txt' \
	-testSetting 'VATP07_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,8 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_org.txt' \
	-testSetting 'VATP08_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'


	echo "Test on p=0,8 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/08/test_ood.txt' \
	-testSetting 'VATP08_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,9 org"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_org.txt' \
	-testSetting 'VATP09_org_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'

	echo "Test on p=0,9 ood"
	python EvalModel_CoNLL_Format.py -datasetName $DATASET \
	-testFile 'data/conll2000_data/perturbed/09/test_ood.txt' \
	-testSetting 'VATP09_ood_'$id'.txt' \
	-cuda_device 1 \
	-task 'pos'
done 
echo "Finished"