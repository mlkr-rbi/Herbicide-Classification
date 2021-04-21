# Comprehensive machine learning based study of the chemical space of herbicides

Several modeling approaches were tested for characterisation of structure-activity relationships according to manually currated MoA/SoA and weed selectivity labels.
This repository contains code for hyperparamter tuning with 10x10-fold cross-validation and training of optimized models (randomForest, XGBoost, SVM, NaiveBayes) on whole training sets.

##  - HRAC
HRAC dataset contains 346 synthetic herbicides available in original HRAC list, and extended sample of 163 herbicides collected from literature and free-access online databases - with additional 131 phytotoxic natural products collected from literature. Chemical space for HRAC classification approach were represented with 166-bit MACCS structural fingeprints

## Datasets - Weed Selectivity
Dataset used for weed selectivity and application stage inference were collected from subsets of 221 and 323 herbicides, respectively, with assigned labels. Weed selectivity datasets were all preprocessed and standardized [(x - mean(x)) / sd(x)] prior to upload.
