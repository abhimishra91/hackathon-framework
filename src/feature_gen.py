import pandas as pd
from sklearn import preprocessing


class FeatureGen:
    def __init__(
        self,
        df,
        target_cols: list = None,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = True,
        feature_gen: str = "poly",
    ):
        """
        :param df: Dataframe that needs to be used for generation of feature
        :param target_cols: List of columns that the method needs to be applied on
        :param feature_gen: Method to be used to generate features, ploy=Polynomical Feature generator from sklearn
        """
        self.df = df
        self.feature_gen = feature_gen
        self.target_cols = target_cols
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        if self.target_cols is None:
            self.target_cols = self.df.columns

    def fit_transform(self):
        if self.feature_gen == "poly":
            polynomial = preprocessing.PolynomialFeatures(
                self.degree, self.interaction_only, self.include_bias
            )
            new_features = polynomial.fit_transform(self.df[self.target_cols].values)
            new_features = pd.DataFrame(new_features)
            output_df = pd.concat([self.df, new_features], axis=1)
            return output_df


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(
        r"C:\Users\abhis\Documents\01_proj\kaggle_comp\sample_ihsm\input\train_sample.csv"
    )
    poly = FeatureGen(df, degree=2)
    new_df = poly.fit_transform()
    print(new_df.head())
