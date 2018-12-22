
import pandas as pd

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

cols = ['GrLivArea','YearBuilt','OverallCond']

df_test_x = df_test[cols]

df_x = df_train[cols]
df_y = df_train["SalePrice"]

# clf = LinearRegression()
clf = SVR()
clf.fit(df_x, df_y)

y_pred = clf.predict(df_test_x)

pred_df = pd.DataFrame(y_pred, index=df_test["Id"], columns=["SalePrice"])
pred_df.to_csv('./mb.csv', header=True, index_label='Id')
