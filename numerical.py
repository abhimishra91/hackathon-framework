from sklearn import preprocessing
import numpy as np


class NumericalFeatures:
    def __init__(self, df, numerical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: List of Columns that we want to encode
        encoding_type: standard, min-max, power
        """
        self.df = df
        self.num_feats = numerical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.stan_scaler = dict()
        self.min_max_encoder = dict()
        self.power_transform_encoder = dict()
        self.log_transform = dict()

        if self.handle_na:
            for c in self.num_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna('-9999999999')

        self.output_df = self.df.copy(deep=True)

    def _standard_scaler(self):
        for c in self.num_feats:
            ss = preprocessing.StandardScaler()
            ss.fit(self.df[c].values.reshape(-1, 1))
            self.output_df.loc[:, c] = ss.transform(self.df[c].values.reshape(-1, 1))
            self.stan_scaler[c] = ss
        return self.output_df

    def _min_max_scaler(self):
        for c in self.num_feats:
            mms = preprocessing.MinMaxScaler()
            mms.fit(self.df[c].values.reshape(-1, 1))
            self.output_df.loc[:, c] = mms.transform(self.df[c].values.reshape(-1, 1))
            self.min_max_encoder[c] = mms
        return self.output_df

    def _power_transform(self):
        for c in self.num_feats:
            powt = preprocessing.PowerTransformer()
            powt.fit(self.df[c].values.reshape(-1, 1))
            self.output_df.loc[:, c] = powt.transform(self.df[c].values.reshape(-1, 1))
            self.power_transform_encoder[c] = powt
        return self.output_df

    def _log_transform(self):
        for c in self.num_feats:
            logt = preprocessing.FunctionTransformer(np.log1p, validate=True)
            logt.fit(self.df[c].values.reshape(-1,1))
            self.output_df.loc[:, c] = logt.transform(self.df[c].values.reshape(-1,1))
            self.log_transform[c] = logt
        return self.output_df

    def fit_transform(self):
        if self.enc_type == "min-max":
            return self._min_max_scaler()
        elif self.enc_type == "standard":
            return self._standard_scaler()
        elif self.enc_type == "power":
            return self._power_transform()
        elif self.enc_type == "log":
            return self._log_transform()
        else:
            raise Exception('Transformation Type not understood')

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.num_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna('-9999999')

        if self.enc_type == 'min-max':
            for c, mms in self.min_max_encoder.items():
                dataframe.loc[:, c] = mms.transform(dataframe[c].values.reshape(-1, 1))
            return dataframe
        elif self.enc_type == 'standard':
            for c, ss in self.stan_scaler.items():
                dataframe.loc[:, c] = ss.transform(dataframe[c].values.reshape(-1, 1))
            return dataframe
        elif self.enc_type == "power":
            for c, powt in self.power_transform_encoder.items():
                dataframe.loc[:, c] = powt.transform(dataframe[c].values.reshape(-1, 1))
            return dataframe
        elif self.enc_type == "log":
            for c, logt in self.log_transform.items():
                dataframe.loc[:, c] = logt.transform(dataframe[c].values.reshape(-1,1))
        else:
            raise Exception('Transformation not understood')

# if __name__ == "__main__":
#     import pandas as pd
#
#     df = pd.read_csv(r'C:\Users\abhis\OneDrive - IHS Markit\Python\00_practice\00_practice\diamonds.csv',
#                      encoding='latin-1')
#     num_cols = ['price']
#     num_feat_transform = NumericalFeatures(df, num_cols, encoding_type='log', handle_na=False)
#     transformed_df = num_feat_transform.fit_transform()
#     print(transformed_df.head())
