# Hackathon Framework

Objective of this project to enable quick experimentation in Data Analytics projects with minimal cookie cutter programming.
Getting rid of all the fit_transforms.! 

---
***NOTE***

- This is a work in progress. Underlying modules are in process of development.
- As this project matures there will be changes in the scripts such as `train.py` and `predict.py`
- TODO
    * Create modules for `tuning`, `stacking`
    * Removal of some of the modules that are redundant


---

The framework is designed to make the Data Science flow easier to perform, by encapsulating different techniques for each step within 1 method.
There are classes for each of the below listed steps:

 - Feature Evaluation
    * Report to give an intution of the dataset
 
 - Feature Engineering
    * Modules to perform feature transformations on Categorical and Numerical Dataset.
    * Various applicable techniques are encoded within these modules and are accesed with an argument.
 
 - Fefature Generation
    * Module to create new features based on different techniques
   
 - Cross Validation
    * Stratified Folding both for Regression and Classification

 - Training
    * Run multiple models using 1 class.
    * Evaluating and Saving the results in an organized manner

 - Tuning
    * Hyper-parameter tuning of multiple models, based on json arguments for parameter values.
 
 - Prediction

 - Evaluating the model

  
  

## Steps to use the framework

  

1. Clone the repo.
2. Create 3 folders `input` and `model` and `tuneq`.
3. Save the training, testing and sample submission file in `input` folder. 
4. The outputs generated from training such as trained model, encoders and oof_preds will be saved in `model` folder.
5. The parameters for fine tuning the models should be saved in the `tune` folder.
6. Update the `config.py` to point it to the correct path for data, model and tuning.
7. Update the `dispatcher.py` with model/models you want to run your dataset on.
8. Use the sample notebook to understand how to use this framework after this intial configuration is completed.




## Description of Files and their Purpose

- `config.py`: Config file to give path of all the datasets and other standard configuration items. Such as csv files path, random seed etc.
 
- `feature_eval.py`: This script and the class inside is used to analyze the dataframe and its columns to get the following output:
	 - min, max and unique values of each column
	 - histogram/ distribution of each column
	 - corelation of columns using a heat map
	 
- `feature_gen.py`: Encapsulates method to generate new features. Currently implemented the `Polynomial features` method from sklearn.
    Returns Dataframe with new features. 

- `feature_impute.py`: Encapsulates the method to impute blank values in a dataframe.
    Currently, it supports 3 imputation methods:
    - Simple Imputer
    - Model Based Imputer: Extra Trees or knn
    - Knn based imputer
    - Returns updated Dataframe

- `cross_validation.py`: This class is used to perform cross validation on any dataframe based on the type of problem statement. It is used to create cross validated dataset.

- `categorical.py`: This class can be used for encoding of categorical features in a given dataframe.
	- Inputs : Dataframe, Categorical Columns List, Type of Encoding
	- Output: Encoded Dataframe
	- Supported Encoding Techniques:
	    - Lable Encoding
	    - Binary Encoding
	    - One Hot Encoding

- `numerical.py`: This class can be used for encoding of numerical features in a given dataframe.
	- Inputs : Dataframe, Categorical Columns List, Type of Encoding
	- Output: Encoded Dataframe, Transformer Object for later use. 
	- Support Techniques:
	    - Standard Scaler
	    - Min-Max Scaler
	    - Power Tranformer
	    - Log Transformer
	
- `metrics.py`: This class can be used to evaluate the results of given predictions and actual value. 

- `dispatcher.py`: Python File with Models and parameters. They have been designed to supply the models to `engine.py` for training on a given dataset

- `engine.py`: This script encapsulates the method to train and evaluate the multiple models simultaneously
    - Leverages on `dispatcher.py` and `metrics.py` for model and metrics
    - The results for each fold are also saved in the `models` folder as `oof_predictions.csv` for each model.
    - **To Do** Stacking module to suporrt stacking of multiple models

- **Scripts to be ignored for now**:
    - `train.py`: For training
    - `predict.py`: For prediction
    - `tune.py`: For tuning h-parameter
    - `create_folds.py`: To create folded datframe