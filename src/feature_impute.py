from sklearn import impute
from sklearn.experimental import enable_iterative_imputer
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors


class FeatureImpute:
    def __init__(
            self,
            dataframe,
            target_col: list,
            impute_method: str = 'simple',
            impute_model: str = 'lr',
            impute_stratergy: str = 'mean',
    ):
        """

        :param dataframe: Dataframe that is to be imputed
        :param target_col: List of columns on which imputation is to be performed
        :param impute_method: String to define the possible imputation stratergy. 'simple', 'model', 'knn'
        :param impute_model: String to define if any model is to be used for imputation. Values: 'lr', 'et', 'knn'
        :param impute_stratergy: String to define what strategy to be used for imputing
        """

        self.df = dataframe
        self.target = target_col
        self.impute_method = impute_method
        self.model = impute_model
        self.stratergy = impute_stratergy

        if self.model == 'et':
            self.estimator = ensemble.ExtraTreesRegressor(n_estimators=50, random_state=42)
        elif self.model == 'knn':
            self.estimator = neighbors.KNeighborsRegressor(n_neighbors=15)
        else:
            self.estimator = linear_model.LinearRegression()

        self.output_df = self.df.copy(deep=True)

    def _simple_impute(self):
        for col in self.target:
            s_impute = impute.SimpleImputer()
            s_impute.fit(self.df[col].values)
            self.output_df.loc[:, col] = s_impute.fit_transform(self.df[col].values)
        return self.output_df

    def _model_impute(self):
        for col in self.target:
            m_impute = impute.IterativeImputer(estimator=self.estimator, random_state=42)
            m_impute.fit(self.df[col].values)
            self.output_df.loc[:, col] = m_impute.fit_transform(self.df[col].values)
        return self.output_df

    def _knn_impute(self):
        for col in self.target:
            k_impute = impute.KNNImputer()
            k_impute.fit(self.df[col].values)
            self.output_df.loc[:, col] = k_impute.fit_transform(self.df[col].values)
        return self.output_df

    def fit_transfom(self):
        if self.impute_method == 'simple':
            return self._simple_impute()
        elif self.impute_method == 'model':
            return self._model_impute()
        elif self.impute_method == 'knn':
            return self._knn_impute()
        else:
            raise Exception('Imputation Type not defined')
