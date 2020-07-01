import matplotlib.pyplot as plt
import seaborn as sns


class FeatEvaluation:
    def __init__(self, df, target_col: str = None):
        """
        :param df: Dataframe which will be analysed
        :param target_col: String of the colummn name that is the target for this analysis in the dataframe
        """
        self.df = df
        self.target = target_col

    def stat_desc(self, col):
        if self.df[col].dtype == "O":
            return "Categorical Data"
        else:
            return self.df[col].describe().loc[["min", "max"]]

    def feature_report(self):
        print("Feature Report Generated for all the columns in the Dataframe")
        for col in self.df.columns:
            print("\n")
            print(f"Feature Report for Column: {col}")
            print("~~~~~~==================~~~~~~")
            print(str(self.stat_desc(col)))
            print(f"No of Unique Values: {self.df[col].nunique()}")
            print(f"No of Values in the column: {self.df[col].value_counts()}")
        return

    def feature_plot(self):
        for col in self.df.columns:
            print("Plotting the Distribution for: {0}".format(col))
            if self.df[col].dtype == "O":
                plt.figure(figsize=(16, 9))
                sns.boxplot(x=col, y=self.target, data=self.df)
                plt.show()
            else:
                plt.figure(figsize=(16, 9))
                sns.distplot(self.df[col].values)
                plt.show()
        return

    def corelation_plot(self):
        corr = self.df.corr()
        plt.figure(figsize=(16, 9))
        sns.heatmap(
            corr,
            annot=True,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="coolwarm",
            linewidths=1.5,
            linecolor="black",
        )
        plt.show()
        return


# if __name__ == "__main__":
#     import config
#     import pandas as pd
#     RAW_TRAIN_DATA = config.RAW_DATA
#     TEST_DATA = config.TEST_DATA
#
#     train_df = pd.read_csv(RAW_TRAIN_DATA)
#     test_df = pd.read_csv(TEST_DATA)
#     test_df['target'] = -99999
#     eval = FeatEvaluation(train_df, 'price')
#     print(eval.feature_report())
