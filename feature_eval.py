import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FeatEval:
    def __init__(self, df, target_col=None, feature_report=True, distribution=True, corelation=True):
        """
        df: pandas dataframe to be evaluated
        target_col: Target column in the dataframe for estimation
        """
        self.df = df
        self.target = target_col
        self.feat_report = feature_report
        self.dist = distribution
        self.corr = corelation

    def stat_desc(self, df, col):
        if self.df[col].dtype == 'O':
            return 'Categorical Data'
        else:
            return self.df[col].describe().loc[['min', 'max']]

    def feat_plot(self, df, col):
        if df[col].dtype == 'O':
            return 'To be decided'
        else:
            plt.figure(figsize=(16,9))
            sns.distplot(df[col])
            plt.show()
            return

    def corr_plt(self, df):
        corr = df.corr()
        plt.figure(figsize=(16,9))
        sns.heatmap(corr, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=1.5, linecolor='black')
        plt.show()
        return

    def __call__(self):
        print('Feature Report Generated for all the columns in the Dataframe')
        for col in self.df.columns:
            print('\n')
            if self.feat_report:
                print(f'Feature Report for Column: {col}')
                print('~~~~~~==================~~~~~~')
                print(str(self.stat_desc(self.df, col)))
                print(f'No of Unique Values: {self.df[col].nunique()}')
            if self.dist:
                self.feat_plot(self.df, col)
        if self.corr_plt:
            self.corr_plt(self.df)
        return


# if __name__ == "__main__":
    # import config
    # RAW_TRAIN_DATA = config.RAW_DATA
    # TEST_DATA = config.TEST_DATA

    # train_df = pd.read_csv(RAW_TRAIN_DATA)
    # test_df = pd.read_csv(TEST_DATA)
    # test_df['target'] = -99999
    # feat_eval = FeatEval(train_df, 'price', feature_report=True, distribution=True, corelation=True)()