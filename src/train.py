import argparse
import pandas as pd

from sklearn import preprocessing
from sklearn import metrics

import joblib

from src import dispatcher, config

TRAINING_DATA = config.TRAINING_DATA
TEST_DATA = config.TEST_DATA
FOLDS = config.FOLDS

FOLD_MAPPING = {
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

if __name__ == "__main__":
    # TODO Update the train.py to use the ML Framework for all the work
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Type in the model you want to run", type=str)
    args = parser.parse_args()

    MODEL = args.model


    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    idx = []
    predictions = []

    for FOLD in range(FOLDS):
        train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
        valid_df = df[df.kfold==FOLD]

        ytrain = train_df.target.values
        yvalid = valid_df.target.values

        idx.extend(valid_df["id"].values.tolist())

        train_df = train_df.drop(["id", "kfold", "target"], axis=1)
        valid_df = valid_df.drop(["id", "kfold", "target"], axis=1)
        valid_df = valid_df[train_df.columns]

        label_encoder = {}
        for c in train_df.columns:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(train_df[c].values.tolist()+valid_df[c].values.tolist()+df_test[c].values.tolist())
            train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
            valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
            label_encoder[c] = lbl

        clf = dispatcher.MODELS[MODEL]
        clf.fit(train_df, ytrain)
        preds = clf.predict_proba(valid_df)[:,1]
        predictions.extend(preds.tolist())
        print(metrics.roc_auc_score(yvalid, preds))

        joblib.dump(label_encoder, f'models/{MODEL}_{FOLD}_label_encoder.pkl')
        joblib.dump(clf, f'models/{MODEL}_{FOLD}_.pkl')
        joblib.dump(train_df.columns, f'models/{MODEL}_{FOLD}_columns.pkl')
    oof_dict = {'id': idx, 'Predictions': predictions}
    oof_pred = pd.DataFrame(oof_dict)
    oof_pred.to_csv(f'models/{MODEL}_oof_predictions.csv')