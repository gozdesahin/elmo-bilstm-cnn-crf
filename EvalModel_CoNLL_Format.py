#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
import nltk
import sys
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation, addEmbeddings, readCoNLL
from neuralnets.ELMoBiLSTM import ELMoBiLSTM
from neuralnets.ELMoWordEmbeddings import ELMoWordEmbeddings

if len(sys.argv) < 3:
    print("Usage: python RunModel_CoNLL_Format.py modelPath inputPathToConllFile")
    exit()

modelPath = sys.argv[1]
inputPath = sys.argv[2]
inputColumns = {0: "tokens", 1:'POS', 2:'chunk_BIO'}

# :: Load the model ::
lstmModel = ELMoBiLSTM.loadModel(modelPath)

# :: Prepare the input ::
sentences = readCoNLL(inputPath, inputColumns)
addCharInformation(sentences)
addCasingInformation(sentences)

# :: Map casing and character information to integer indices ::
dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Perform the word embedding / ELMo embedding lookup ::
embLookup = lstmModel.embeddingsLookup
embLookup.elmo_cuda_device = 0         #Cuda device for pytorch - elmo embedding, -1 for CPU
addEmbeddings(dataMatrix, embLookup.sentenceLookup)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)

#test_pre, test_rec, test_f1 = lstmModel.computeF1('conll2000_data/clean',dataMatrix)
#print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (test_pre, test_rec, test_f1))

test_acc = lstmModel.computeAcc('conll2000_data/clean',dataMatrix)
print("Test-Data: Accuracy: %.3f" % (test_acc))
