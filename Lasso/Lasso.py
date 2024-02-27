from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error # 平均二乗誤差の計算モジュール

df = pd.read_csv('normalized_dataset.csv',header = None, names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'])

df_train, df_test = train_test_split(df, test_size=0.3, random_state = 4) # トレーニングデータとテストデータに分割

#目的関数と説明変数の設定
x_train = df_train[['E']]
x_test = df_test[['E']]

y_train = df_train[['F']]
y_test = df_test[['F']] 

# モデルの構築
LssModel = Lasso(alpha=0.01)

#モデルの学習
LssModel.fit(x_train, y_train)

#予測
y_pred = LssModel.predict(x_test)
print(y_pred)


plt.scatter(x_test, y_test) #正解のプロット

plt.scatter(x_test, y_pred, color = 'red') #予測のプロット

plt.show()









