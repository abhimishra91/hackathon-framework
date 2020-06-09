from sklearn import metrics as skmetrics
import numpy as np


class RegressionMetric:
    def __init__(self):
        self.metrics = {
            "mae": self._mae,
            "mse": self._mse,
            "rmse": self._rmse,
            "msle": self._msle,
            "rmsle": self._rmsle,
            "r2": self._r2,
            "mape": self._mape
        }

    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception("Metric not implemented")
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _mae(y_true, y_pred):
        return skmetrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)  

    @staticmethod
    def _mse(y_true, y_pred):
        return skmetrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def _rmse(self, y_true, y_pred):
        return np.sqrt(self._mse(y_true=y_true, y_pred=y_pred))

    @staticmethod
    def _msle(y_true, y_pred):
        return skmetrics.mean_squared_log_error(y_true=y_true, y_pred=y_pred)

    def _rmsle(self, y_true, y_pred):
        return np.sqrt(self._msle(y_true=y_true, y_pred=y_pred))

    @staticmethod
    def _r2(y_true, y_pred):
        return skmetrics.r2_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        y_val = np.maximum(y_true, 1e-8)
        return (np.abs(y_true - y_pred) / y_val).mean()


class ClassificationMetric:
    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "f1": self._f1,
            "recall": self._recall,
            "precision": self._precision,
            "auc": self._auc,
            "logloss": self._logloss
        }

    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception('Metric not implemented')
        if metric == 'auc':
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_proba)
            else: 
                raise Exception("y_proba cannot be None for AUC")
        elif metric == 'logloss':
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_proba)
            else: 
                raise Exception("y_proba cannot be None for LogLoss")
        else: 
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)
