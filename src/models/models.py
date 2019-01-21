# coding:utf-8

import numpy as np

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb
import lightgbm as lgb

R_SEED = 1234


class ManshionBrothers:
    def __init__(self):
        self.models = []
        self.model_initializer = {
                "Lasso"       : self.init_Lasso,
                "ElasticNet"  : self.init_ElasticNet,
                "KernelRidge" : self.init_KernelRidge,
                "GB"          : self.init_GradientBoosting,
                "XGB"         : self.init_XGBoost,
                "LGB"         : self.init_LGB
                }

    def compile(self, models=["LGB"]):
        self.models = []
        for mod in models:
            if not isinstance(mod, list):
                self.models.append(self.model_initializer[mod]())
            else:
                self.models.append(self.model_initializer[mod[0]](mod[1]))

    def train(self, x, y):
        for mod in self.models:
            mod.fit(x, y)

    def predict(self, x, model_weights=None):
        if len(self.models)==0:
            print("You must run 'compile' before predict")
            return

        if len(self.models)>1:
            return self.ensemble(x, model_weights)
        else:
            return self.models[0].predict(x)

    def ensemble(self, x, model_weights):
        if model_weights==None:
            model_weights = [1 for _ in self.models]

        ## predict
        preds = []
        for mod in self.models:
            preds.append(mod.predict(x))

        ## weighting
        result = np.zeros(shape=preds[0].shape)
        for p, w in zip(preds, model_weights):
            result += p * w

        return result / sum(model_weights)

    def evaluate(self, y, y_pred):
        mse = mean_squared_error(y, y_pred)
        msle = mean_squared_log_error(y, y_pred)
        print("MSE :", mse)
        print("MSLE:", msle)
        return mse, msle


    ## ---------------------------------------------------------- ##
    def init_Lasso(self, config={}):
        ## default
        p = {"alpha":0.0005, "random_state":R_SEED}

        ## overwrite parameters
        for k, v in config.items():
            p[k] = v

        ## instance
        mod = Lasso(alpha=p["alpha"], random_state=p["random_state"])

        return mod

    def init_ElasticNet(self, config={}):
        ## default
        p = {"alpha":0.0005, "l1_ratio":0.9, "random_state":R_SEED}

        ## overwrite parameters
        for k, v in config.items():
            p[k] = v

        ## instance
        mod = ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"],
                         random_state=p["random_state"])

        return mod

    def init_KernelRidge(self, config={}):
        ## default
        p = {"alpha":0.6, "kernel":"polynomial", "degree":2, "coef0":2.5}

        ## overwrite parameters
        for k, v in config.items():
            p[k] = v

        ## instance
        mod = KernelRidge(alpha=p["alpha"], kernel=p["kernel"],
                          degree=p["degree"], coef0=p["coef0"])

        return mod

    def init_GradientBoosting(self, config={}):
        ## default
        p = {"n_estimators":3000, "learning_rate":0.05, "max_depth":4,
            "max_features":"sqrt", "min_samples_leaf":15, "min_samples_split":10,
            "loss":"huber", "random_state":R_SEED}

        ## overwrite parameters
        for k, v in config.items():
            p[k] = v

        ## instance
        mod = GradientBoostingRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                       max_depth=p["max_depth"], max_features=p["max_features"],
                                       min_samples_leaf=p["min_samples_leaf"], min_samples_split=p["min_samples_split"],
                                       loss=p["loss"], random_state=p["random_state"])

        return mod

    def init_XGBoost(self, config={}):
        ## default
        p = {"n_estimators":3000, "learning_rate":0.01, "max_depth":6}

        ## overwrite parameters
        for k, v in config.items():
            p[k] = v

        ## instance
        mod = xgb.XGBRegressor(learning_rate=p["learning_rate"], n_estimators=p["n_estimators"],
                               max_depth=p["max_depth"])

        return mod

    def init_LGB(self, config={}):
        ## default
        p = {"objective":"regression", "num_leaves":5, "learning_rate":0.05,
             "n_estimators":3000, "max_bin":55, "bagging_fraction":0.8,
             "bagging_freq":5, "feature_fraction":0.2319, "feature_fraction_seed":9,
             "bagging_seed":9, "min_data_in_leaf":6, "min_sum_hessian_in_leaf":11}

        ## overwrite parameters
        for k, v in config.items():
            p[k] = v

        ## instance
        mod = lgb.LGBMRegressor(objective=p["objective"], num_leaves=p["num_leaves"],
                                learning_rate=p["learning_rate"], n_estimators=p["n_estimators"],
                                max_bin=p["max_bin"], bagging_fraction=p["bagging_fraction"],
                                bagging_freq=p["bagging_freq"], feature_fraction=p["feature_fraction"],
                                feature_fraction_seed=p["feature_fraction_seed"], bagging_seed=p["bagging_seed"],
                                min_data_in_leaf=p["min_data_in_leaf"],
                                min_sum_hessian_in_leaf=p["min_sum_hessian_in_leaf"])

        return mod
    ## ---------------------------------------------------------- ##


if __name__ == "__main__":
    import pandas as pd

    ### regressor
    mb = ManshionBrothers()


    ### load data
    fp = "../../data/train_set.csv"
    df = pd.read_csv(fp, low_memory=False).head(100)


    ### prepare x and y
    df_x = df[["BATHRM","EYB"]]
    df_y = df["PRICE"]


    ### prepare models
    mb.compile(models=["LGB"])

    ## ensembling models
    # mb.compile(models=["LGB"]) # <- single model
    # mb.compile(models=["LGB", "XGB"]) # <- using two models
    # mb.compile(models=["Lasso","ElasticNet","KernelRidge","GB","XGB","LGB"]) # <- using all models

    ## passing model parameters
    # mb.compile(models=[["LGB",{"n_estimators":1000}]])
    # mb.compile(models=[ ["LGB",{"n_estimators":1000}], ["XGB", "n_estimators":1000] ])


    ### train
    mb.train(df_x, np.log1p(df_y))


    ### predict
    pred = mb.predict(df_x)

    ## if more than 1 models, you can pass model weights
    # mb.compile(models=["LGB", "XGB"]) # <- using two models
    # mb.train(df_x, np.log1p(df_y))
    # pred = mb.predict(df_x, model_weights=[1,0.5])


    ### evaluate
    mb.evaluate(np.log1p(df_y), pred)


