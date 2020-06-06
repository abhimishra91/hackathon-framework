import numpy as np
from sklearn import model_selection


"""

Categories to tackle
- binary classification
- multi class classification
- multi label classification
- single column regression
- multi column regression
- hold out

"""


class CrossValidation():
    def __init__(
        self,
        df,
        target_cols,
        shuffle=False,
        problem_type='binary_classification',
        stratified_regression=False,
        multilabel_delimiter=',',
        n_folds=5,
        random_state=42
    ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.shuffle = shuffle
        self.problem_type = problem_type
        self.stratified_regression = stratified_regression
        self.num_folds = n_folds
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe['kfold'] = -1

    def _sort_partition(self, y, num_folds):
        n = len(y)
        cats = np.empty(n, dtype='u4')
        div, mod = divmod(n, num_folds)
        cats[:n-mod] = np.repeat(range(div), num_folds)
        cats[n-mod:] = div + 1
        return cats[np.argsort(np.argsort(y))]

    def split(self):
        if self.problem_type in ('binary_classification', 'multiclass_classification'):
            if self.num_targets != 1:
                raise Exception('Invalid number of target for this type of Problem statement')
            target = self.target_cols[0]
            unqiue_values = self.dataframe[target].nunique()
            if unqiue_values == 1:
                raise Exception('Only one class present in the data, No ML is needed')
            elif unqiue_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, shuffle=False)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ('single_col_regression', 'multi_col_regression'):
            if self.num_targets != 1 and self.problem_type == 'single_col_regression':
                raise Exception('Invalid number of targets for this type of problem')
            if self.num_targets < 2 and self.problem_type == 'multi_col_regression':
                raise Exception('Invalid number of targets for this type of problem')
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ('single_col_regression') and self.stratified_regression:
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            y = self.dataframe[target].values
            y_categorized = self._sort_partition(y, self.num_folds)
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds,shuffle=False)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=y_categorized)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1
        
        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, 'kfold'] = fold
        
        else:
            raise Exception("Problem type not understood!")
        
        return self.dataframe

# if __name__ == "__main__":
#     import config
#     import pandas as pd
#     REG_DATA = config.RAW_DATA
#     df = pd.read_csv(REG_DATA)
#     cross_val = CrossValidation(df=df, target_cols=['price'], problem_type='single_col_regression', stratified_regression = True)
#     df_folds = cross_val.split()
#     print(df_folds.head())


