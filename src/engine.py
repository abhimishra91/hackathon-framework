import pandas as pd
from src import dispatcher
import joblib
from pathlib import Path
from src.metrics import RegressionMetric, ClassificationMetric


class Engine:
    def __init__(
        self,
        dataframe,
        id_col: str,
        target_col: str,
        folds: int,
        unused_col: list = None,
        problem_type: str = "regression",
        model_list: list = None,
        save_model: bool = False,
    ):
        """

        :param dataframe: Dataframe for the folded dataset to be used for training/tuning
        :param id_col: String for the column name that is used for identfying the rows in dataframe
        :param target_col: String of Column name that is to be used for prediction
        :param folds: Number of folds in the dataframe
        :param unused_col: List of columns that is not to be considered for training
        :param model_list: List of models that are to be run for comparison, tuning or training.
        :param problem_type: This will define the type of models to pick.Possible values: regression/classification
        :param save_model: True or False to specify if you want the save the model file
        """
        self.df = dataframe
        self.id = id_col
        self.target = target_col
        self.folds = folds
        self.problem = problem_type
        self.models = model_list
        self.save_model = save_model
        self.model_dict = dict()
        self.result_dict = dict()

        if unused_col is None:
            self.unused_col = self.target.split()
        else:
            self.unused_col = unused_col

        if self.problem == "regression":
            self.models_used = dispatcher.REGRESSION_MODELS
        else:
            self.models_used = dispatcher.CLASSIFICATION_MODELS

        self.output_path = Path("./models/")

    @staticmethod
    def _generate_mapping(folds):
        fold_dict = dict()
        for i in range(folds):
            fold_dict[i] = [x for x in range(folds) if x != i]
        return fold_dict

    @staticmethod
    def _save_model(self, model: str, fold: int, clf):
        joblib.dump(clf, f"{self.output_path}/{str(model)}__{str(fold)}__.pkl")
        return

    @staticmethod
    def _save_result(self):
        for model_result in self.result_dict.keys():
            result_df = pd.DataFrame(self.result_dict[model_result])
            result_df.to_csv(
                f"{self.output_path}/{str(model_result)}__oof_predictions.csv",
                index=False,
            )
        return

    @staticmethod
    def _train_model(self, model, fold_dict: dict):
        idx = list()
        actuals = list()
        predictions = list()
        fold_list = list()
        for fold in range(self.folds):
            train_df = self.df[self.df.kfold.isin(fold_dict.get(fold))]
            valid_df = self.df[self.df.kfold == fold]
            ytrain = train_df[self.target].values
            yvalid = valid_df[self.target].values
            idx.extend(valid_df[self.id].values.tolist())
            train_df = train_df.drop(self.unused_col, axis=1)
            valid_df = valid_df.drop(self.unused_col, axis=1)
            valid_df = valid_df[train_df.columns]
            clf = self.models_used[model]
            clf.fit(train_df, ytrain)
            preds = clf.predict(valid_df)
            predictions.extend(preds.tolist())
            actuals.extend(yvalid.tolist())
            fold_list.extend([fold for num in range(len(yvalid))])
            if self.save_model:
                self._save_model(self, model, fold, clf)
        result = {
            self.id: idx,
            "Predictions": predictions,
            "Actuals": actuals,
            "Fold": fold_list,
        }
        return result

    def train_models(self):
        fold_dict = self._generate_mapping(self.folds)
        if self.models is None:
            self.models = list(self.models_used.keys())
        for model in self.models:
            if self.save_model:
                print(
                    f"Training model: {str(model)}, and saving the model and results at: {str(self.output_path)}"
                )
            else:
                print(
                    f"Training model: {str(model)}, and saving only the results at: {str(self.output_path)}"
                )
            self.result_dict[model] = self._train_model(self, model, fold_dict)
        print(f"Saving the results of Trained Models")
        self._save_result(self)
        return

    def stack(self):
        #   TODO Implement stacking in Engine. IP: model_preds, meta_model | OP: Pred, result
        return

    def evaluate(
        self, model_list: list, metric: str = None, target_transformer: dict = None
    ):
        """

        :param model_list: List of models which needs to be evaluated
        :param metric: Metric that is to be calcuated for the models
        :param target_transformer: If any transformer is used on the target, pass that transformer object here.
        :return: dataframe with metric and mean
        """
        for model in model_list:
            print("Evaluating the result for model {0}".format(str(model)))
            result_df = pd.read_csv(
                f"{self.output_path}/{str(model)}__oof_predictions.csv"
            )
            if self.problem == "regression":
                metric_type = RegressionMetric()
            else:
                metric_type = ClassificationMetric()
            if target_transformer is None:
                result = metric_type(
                    metric,
                    result_df["Actuals"].values.reshape(-1, 1),
                    result_df["Predictions"].values.reshape(-1, 1),
                )
            else:
                result = metric_type(
                    metric,
                    target_transformer[self.target].inverse_transform(
                        result_df["Actuals"].values.reshape(-1, 1)
                    ),
                    target_transformer[self.target].inverse_transform(
                        result_df["Predictions"].values.reshape(-1, 1)
                    ),
                )

        return result


# if __name__ == "__main__":
#     import pandas as pd
#
#     df = pd.read_csv(r'C:\Users\abhis\Documents\01_proj\kaggle_comp\sample_ihsm\input\train_folds.csv')
#     engine = Engine(
#         dataframe=df,
#         id_col='vehicle_id',
#         target_col='Price_USD',
#         folds=5,
#         save_model=True,
#         model_list=['lr']
#     )
#     engine.train_models()
#     my_result = engine.evaluate(model_list=['lr'], metric='mape')
#     print(my_result)
