import numpy as np
import matplotlib.pyplot as plt

# X：30個の入力データ（身長 [cm]）
X = np.array([131, 132, 132, 133.5, 135, 142, 143.8, 144, 148, 149, 150, 152,
              153, 157, 158, 158, 162, 164, 166, 169, 169.5, 170, 172, 173, 173, 176, 180, 184, 186, 190])

# Y：Xに対応する正解データ（体重 [kg]）
Y = np.array([31, 28, 35, 40, 31, 40, 42, 45, 50, 48, 56, 50, 51, 56, 65, 61, 66,
              61.5, 69, 71, 63, 68, 80, 74, 76.5, 82, 68, 75, 92, 90])

# データの標準化(平均0, 分散1に変換)
def normalize(x):
    M = x.mean()
    S = x.std()
    x = (x - M) / S
    return x
    
X = normalize(X)
Y = normalize(Y)

m = X.shape[0] #データ数

#初期値を１として、パラメータa,bを設定する
a = 1
b = 1

#繰り返し回数
iterations = 1000
#学習率
alpha = 0.01

#目的関数の値を保存するリスト
cost = []

for i in range(iterations):
    #仮説を定義
    h = a * X + b
    
    a = a - alpha * (1/m) * np.sum((h - Y) * X)
    b = b - alpha * (1/m) * np.sum(h - Y) 
    
    #更新後のパラメータで仮説と目的関数の値を計算
    h = a * X + b
    J = (1/(2*m)) * np.sum((h - Y)**2)
    
    #学習によって目的関数の値がどう変化しているかを後で見れるように、現在の目的関数の値をリストに保存します
    cost.append(J)
    
# 学習した結果を表示
print("学習後のa: %f,"% a)
print("学習後のb: %f,"% b)
print("学習後の目的関数の値: %f,"% J) 

##################################################################

#学習曲線をプロット
h = a * X + b
plt.scatter(X, Y) #正解のプロット
plt.plot(X, h, c='orange') #仮説をプロット
plt.legend((u'Data',u'Regression line')) #凡例
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()