import argparse
import numpy as np
import pandas as pd

import joblib

from src import config

TRAINING_DATA = config.TRAINING_DATA
TEST_DATA = config.TEST_DATA
FOLDS = config.FOLDS



def predict(MODEL, FOLDS):
    MODEL = MODEL
    df = pd.read_csv(TEST_DATA)
    text_idx = df["id"].values
    predictions = None

    for FOLD in range(FOLDS):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(f'models/{MODEL}_{FOLD}_label_encoder.pkl')
        cols = joblib.load(f'models/{MODEL}_{FOLD}_columns.pkl')
        for c in encoders:
            print(c)
            lbl = encoders[c]
            df.loc[:,c] = lbl.transform(df[c].values.tolist())

        clf = joblib.load(f'models/{MODEL}_{FOLD}_.pkl')
        df = df[cols]
        preds = clf.predict_proba(df)[:,1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((text_idx, predictions)), columns=['id', 'target'])
    return sub



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Type in the model you want to run", type=str)
    args = parser.parse_args()

    MODEL = args.model

    submission = predict(MODEL, FOLDS)
    submission.id = submission.id.astype(int)
    submission.to_csv(f'models/{MODEL}.csv', index=False)


    
