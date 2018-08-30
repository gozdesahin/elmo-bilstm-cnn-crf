#!/bin/bash
MODEL=models/conll2000_data/clean_0.9704_0.9650_14.h5
echo "clean"
python EvalModel_CoNLL_Format.py $MODEL data/conll2000_data/clean/test.txt
echo "p=0,1"
python EvalModel_CoNLL_Format.py $MODEL data/conll2000_data/perturbed/01/test.txt
echo "p=0,2"
python EvalModel_CoNLL_Format.py $MODEL data/conll2000_data/perturbed/02/test.txt
echo "p=0,3"
python EvalModel_CoNLL_Format.py $MODEL data/conll2000_data/perturbed/03/test.txt
echo "p=0,4"
python EvalModel_CoNLL_Format.py $MODEL data/conll2000_data/perturbed/04/test.txt
echo "p=0,5"
python EvalModel_CoNLL_Format.py $MODEL data/conll2000_data/perturbed/05/test.txt
echo "p=0,6"
python EvalModel_CoNLL_Format.py $MODEL data/conll2000_data/perturbed/06/test.txt
echo "p=0,7"
python EvalModel_CoNLL_Format.py $MODEL data/conll2000_data/perturbed/07/test.txt
echo "p=0,8"
python EvalModel_CoNLL_Format.py $MODEL data/conll2000_data/perturbed/08/test.txt
echo "p=0,9"
python EvalModel_CoNLL_Format.py $MODEL data/conll2000_data/perturbed/09/test.txt
echo "Finished"