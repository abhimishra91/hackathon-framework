# Hackathon Framework
 Framework that will be used for Data Hackathon


This is an ML framework in progress that is going to be used for datahackathon events. 
It has been designed to encapsulate the different processes during a data hackathon challenge. 

    * Cross Validation
    * Training
    * Prediction
    * Evaluating the model

It's a work in progress.

To use the framework, create 2 empty folders `input` and `model`.

Save the training, testing and sample submission file in `input` folder.
The outputs generated from training such as trained model, encoders and oof_preds will be saved in `model` folder

Update the `config.py` to point it to the correct path of the training and test files. 