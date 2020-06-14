import pandas as pd
from src import config
from src.cross_validation import CrossValidation

RAW_DATA = config.RAW_DATA
FOLDS_DATA = config.DATA_PATH + r'\diamonds_folds.csv'
# REG_DATA = config.REG_DATA
# FOLDS_DATA_REG = config.DATA_PATH+r'\train_folds_reg.csv'


if __name__ == "__main__":
    # df = pd.read_csv(REG_DATA)
    # df['kfold'] = -1
    df = pd.read_csv(r'C:\Users\abhis\Documents\01_proj\input_data\diamonds.csv')
    df = df.sample(frac=1).reset_index(drop=True)

    # kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    # for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
    #     print(len(train_idx), len(val_idx))
    #     df.loc[val_idx, 'kfold'] = fold

    # cross_val = CrossValidation(df = df, target_cols=['price'], problem_type='single_col_regression',
    # stratified_regression = True)
    cross_val = CrossValidation(df=df, target_cols=['target'], problem_type='single_col_regression', stratified_regression=True)
    df_folds = cross_val.split()
    # df.to_csv(FOLDS_DATA_REG, index=False)
    df_folds.to_csv(FOLDS_DATA, index=False)