# Comprehensive machine learning based study of the chemical space of herbicides
*Laboratory for Machine Learning and Knowledge Representation, Rudjer Boskovic Intitute*

## General Information
Several modeling approaches were tested for characterisation of structure-activity relationships according to manually currated MoA/SoA and weed selectivity labels.
This repository contains code for hyperparamter tuning with 10x10-fold cross-validation and training of optimized models (randomForest, XGBoost, SVM, NaiveBayes) on whole training sets.

## Datasets 

### HRAC
HRAC dataset contains 346 synthetic herbicides available in original HRAC list, and extended sample of 163 herbicides collected from literature and free-access online databases - with additional 131 phytotoxic natural products collected from literature. Chemical space for HRAC classification approach was represented with 166-bit MACCS structural fingeprints

### Weed Selectivity
Dataset used for weed selectivity inference was collected from a subset of 221 herbicides, respectively, with assigned labels. Weed selectivity datasets were all preprocessed and standardized [(x - mean(x)) / sd(x)] prior to upload.


## Requirements
R version 3.6.3  
randomForest 4.6-14  
caret 6.0-84  
klaR 0.6-15  

## Instructions 

1) Clone github repository:    
git clone https://github.com/davoors/Herbicide-Classification.git

2) For HRAC or Weed-Selectivity approach go to "models" folder and run each model seperately with "Rscript" function, and pretrained models will be saved in the same folder as a .rds file:  
ex: Rscript svm_classifier.R

3) To test pretrained model over a set of test sets, run a Predict.R script - Output files will be saved in the output folder for each dedicated approach (ex: "/output/svm/logD or ./logP"):  
ex: Rscript svm_pred.R
