#!/bin/bash
DATASET='conll2000_data/clean'
RESULT='selmo_5e_ct_results'
MODEL='selmo_5e_ct_pos_models'

echo "Test on p=0,1 ood"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/01/test_ood_corrected.txt' \
-testSetting 'MTP01_org_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,1 simple"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/01/test_simple_corrected.txt' \
-testSetting 'MTP01_ood_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,2 ood"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/02/test_ood_corrected.txt' \
-testSetting 'MTP02_org_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,2 simple"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/02/test_simple_corrected.txt' \
-testSetting 'MTP02_ood_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,3 ood"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/03/test_ood_corrected.txt' \
-testSetting 'MTP03_org_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,3 simple"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/03/test_simple_corrected.txt' \
-testSetting 'MTP03_ood_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,4 ood"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/04/test_ood_corrected.txt' \
-testSetting 'MTP04_org_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,4 simple"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/04/test_simple_corrected.txt' \
-testSetting 'MTP04_ood_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,5 ood"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/05/test_ood_corrected.txt' \
-testSetting 'MTP05_org_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,5 simple"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/05/test_simple_corrected.txt' \
-testSetting 'MTP05_ood_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,6 ood"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/06/test_ood_corrected.txt' \
-testSetting 'MTP06_org_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,6 simple"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/06/test_simple_corrected.txt' \
-testSetting 'MTP06_ood_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,7 ood"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/07/test_ood_corrected.txt' \
-testSetting 'MTP07_org_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,7 simple"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/07/test_simple_corrected.txt' \
-testSetting 'MTP07_ood_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,8 ood"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/08/test_ood_corrected.txt' \
-testSetting 'MTP08_org_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,8 simple"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/08/test_simple_corrected.txt' \
-testSetting 'MTP08_ood_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,9 ood"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/09/test_ood_corrected.txt' \
-testSetting 'MTP02_org_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'

echo "Test on p=0,9 simple"
python EvalModel_CoNLL_Format.py -datasetName $DATASET \
-testFile 'data/conll2000_data/perturbed/09/test_simple_corrected.txt' \
-testSetting 'MTP09_ood_'$id'.txt' \
-model_save $MODEL \
-result_save $RESULT \
-cuda_device 0 \
-task 'pos'