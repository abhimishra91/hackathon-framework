from sklearn import preprocessing
import config


class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: List of Columns that we want to encode
        encoding_type: label, binary, ohe-hot-encoding (ohe)
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.handle_na = handle_na

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna('-9999999999')

        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:,c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def _label_binarizer(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values) #array
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}" 
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        self.ohe = ohe
        return ohe.transform(self.df[self.cat_feats].values)

    def fit_transform(self):
        if self.enc_type == 'label':
            return self._label_encoding()
        elif self.enc_type == 'binary':
            return self._label_binarizer()
        elif self.enc_type == 'ohe':
            return self._one_hot()
        else:
            raise Exception('Encoding Type not understood')

    def transform(self, dataframe):
        if self.handle_na: 
            for c in self.cat_feats:
                dataframe.loc[:,c] = dataframe.loc[:,c].astype(str).fillna('-9999999999')

        if self.enc_type == 'label':
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:,c] = lbl.transform(dataframe[c].values)
            return dataframe        
        elif self.enc_type == 'binary':
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values) #array
                dataframe = dataframe.drop(c, axis=1)
                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}" 
                    dataframe[new_col_name] = val[:, j]        
            return dataframe
        elif self.enc_type == 'ohe':
            return self.ohe.transform(dataframe[self.cat_feats].values)
        else:
            raise Exception('Encoding Type not understood')



# if __name__ == "__main__":
#     import pandas as pd
#     from sklearn import linear_model
#     DATA_PATH = config.DATA_PATH
#     TRAINING_DATA = DATA_PATH + r'\train_cat.csv'
#     TEST_DATA = DATA_PATH + r'\test_cat.csv'
#     df = pd.read_csv(TRAINING_DATA)#.head(50)
#     df_test = pd.read_csv(TEST_DATA)#.head(50)
#     submission = pd.read_csv(r'C:\Users\abhis\Documents\01_proj\input_data\submission.csv')
#     train_len = len(df)

#     df_test['target'] = -1

#     full_data = pd.concat([df, df_test])

#     cols = [c for c in df.columns if c not in ['id', 'target']]
#     print(cols)
#     cat_feats = CategoricalFeatures(full_data, categorical_features=cols, encoding_type='ohe', handle_na=True)
    
#     full_data_transformed = cat_feats.fit_transform()
#     # test_transformed = cat_feats.transform(df_test)
    
#     train_transformed = full_data_transformed[:train_len, :]
#     test_transformed = full_data_transformed[train_len:, :]
    
#     print(train_transformed.shape)
#     print(test_transformed.shape)

#     clf = linear_model.LogisticRegression()
#     clf.fit(train_transformed, df.target.values)
#     pred = clf.predict_proba(test_transformed)[:,1]

#     submission.loc[:, 'target'] = pred
#     submission.to_csv(r'C:\Users\abhis\Documents\01_proj\input_data\submission.csv', index = False)



