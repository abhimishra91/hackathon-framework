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



Framework that will be used for Data Hackathon
  

This is an ML framework in progress that is going to be used for datahackathon events.

It has been designed to encapsulate the different processes during a data hackathon challenge.

  

 - Cross Validation

 - Training

 - Prediction

 - Evaluating the model

  
  

## Steps to use the framework

  

1. Clone the repo.
2. Create 2 empty folders `input` and `model`.
3. Save the training, testing and sample submission file in `input` folder. The outputs generated from training such as trained model, encoders and oof_preds will be saved in `model` folder.
4. Update the `config.py` to point it to the correct path of the training and test files.
5. Run the `create_folds.py`
6. Update the `dispatcher.py` with model/models you want to run your dataset on.
7. Open Command Prompt and go to the local of the repo.
8. Run `python train.py <model_name>`
9. The results will be saved in the `model` folder.




## Description of Files and their Purpose

 - `config.py`: Config file to give path of all the datasets and other standard configuration items. Such as csv files path, random seed etc.
 
 -  `feature_eval.py`: This script and the class inside is used to analyze the dataframe and its columns to get the following output:
	 - min, max and unique values of each column
	 - histogram/ distribution of each column
	 - corelation of columns using a heat map

- `create_folds.py`: Creates cross validated dataset for any given dataframe, and then saves it in the given location. Ready for any modelling task. It uses `cross_validation.py`  to perform this action.

- `cross_validation.py`: This class is used to perform cross validation on any dataframe based on the type of problem statement. It is used in the `create_folds.py` script 

- `categorical.py`: This class can be used for encoding of categorical features in a given dataframe.
	- Inputs : Dataframe, Categorical Columns List, Type of Encoding
	- Output: Encoded Dataframe

- `metrics.py`: This class can be used to evaluate the results of given predictions and actual value. 

- `dispatcher.py`: Python File with Models and parameters. They have been designed to supply the models to `train.py` for training on a given dataset

- `train.py`: Python script to train data in a cross validated fashion using the folds created in the dataset. 
	- Currently the script performs following functions:
		- Input: Test dataframe with folds, specified model from the dispatcher
		- Performs label encoding on the categorical dataset `## TO DO ##` 
		- Trains the model on different folds and saves the each model, label encoder and out of fold prediction.

- `predict.py`: Loads all the saved models, label encoders and columns from the folder and runs its on the validation set. Finally the predictions from different models are averaged to generate the final submission file. 
