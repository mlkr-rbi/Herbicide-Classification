# Comprehensive machine learning based study of the chemical space of herbicides

Several modeling approaces were tested for characterisation of structure-activity relationships according to manually currated MoA and weed selectivity.
This repository contains code for hyperparamter tuning with 10x10-fold cross-validation and training of optimized models (randomForest, XGBoost, SVM, NaiveBayes) on whole training sets.

Available dataset - including 166-bit MACCS structural fingeprints for HRAC classification approach, and pre-selected physicochemical descriptors with LogP or LogD variables - were all preprocesed and standardized [(x - mean(x)) / sd(x)] prior to upload.
