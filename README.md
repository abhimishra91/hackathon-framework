# Hackathon Framework
 Framework that will be used for Data Hackathon


This is an ML framework in progress that is going to be used for datahackathon events. 
It has been designed to encapsulate the different processes during a data hackathon challenge. 

    * Cross Validation
    * Training
    * Prediction
    * Evaluating the model


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



**It's a work in progress.**

## TO DO:
    - Update `dispatcher.py` to feed multiple models to `train.py`
    - Create a hyper parameter tuning script
    - Update `create_folds.py` to read multiple files, concat them and then create folds
    - Update `categorical.py` for more encoding methods.
    - Update the framework to include models other than sklearn. Eg:
        - Factorization Machine
        - Neural Networks
        - Vowpal Wabbit