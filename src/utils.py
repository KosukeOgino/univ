# coding:utf-8

from datetime import datetime as dt

import pandas as pd

from sklearn.model_selection import train_test_split

R_SEED = 1234


class DataPreprocessor:
    def __init__(self):
        pass

    def load_csv(self, fp_train, fp_test):
        self.train_org = pd.read_csv(fp_train, low_memory=False)
        self.test_org = pd.read_csv(fp_test, low_memory=False)

    def load_dataframe(self, df_train, df_test):
        self.train_org = df_train.copy()
        self.test_org = df_test.copy()


    def set_cols(self, num_cols=[], cat_cols=[]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols


    def compile(self):
        ## xy
        self.split_xy()

        ## ec
        self.extract_columns()

        ## de
        self.dummy_encoding()

        ## fmv
        self.fill_missing_vals()

        ## cnf
        self.create_new_features()


    def get_validation_data(self):
        x_train, x_valid, y_train, y_valid = train_test_split(self.x_train, self.y_train, test_size=0.2, random_state=R_SEED)
        return x_train, x_valid, y_train, y_valid


    ## -------------------------------------------------- ##
    def split_xy(self):
        self.x_train = self.train_org.iloc[:, :-1]
        self.y_train = self.train_org["PRICE"]

        self.x_test = self.test_org.copy()

    def extract_columns(self):
        self.x_train = self.x_train[self.num_cols + self.cat_cols]
        self.x_test = self.x_test[self.num_cols + self.cat_cols]

    def dummy_encoding(self):
        if len(self.cat_cols)==0:
            return

        x_tmp = pd.concat([self.x_train, self.x_test], axis=0)
        x_tmp = pd.concat([x_tmp[self.num_cols], pd.get_dummies(x_tmp[self.cat_cols])], axis=1)

        self.x_train = x_tmp.iloc[:self.x_train.shape[0], :]
        self.x_test = x_tmp.iloc[self.x_train.shape[0]:, :]

    def treat_outliers(self):
        pass

    def fill_missing_vals(self):
        pass

    def create_new_features(self):
        pass

    ## -------------------------------------------------- ##



class Postprocessor:
    def __init__(self):
        pass

    @classmethod
    def make_submit_csv(cls, y_pred, df_test, save_path=None):
        df_submit = pd.DataFrame(y_pred, index=df_test.index, columns=["PRICE"])

        if save_path != None:
            fn = dt.now().strftime("%Y%m%d_%H%M%S") + ".csv"
            df_submit.to_csv(save_path + fn, header=True, index_label="Id")

        return df_submit



if __name__ == "__main__":
    print("hoge")


