from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data       # 特徴（花の長さなど）
y = iris.target     # 答え（花の種類）

# データを「学習用」と「テスト用」に分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

# 決定木モデルを作る
model = DecisionTreeClassifier()

# 学習用データで学習
model.fit(X_train, y_train)

# テストデータを使って予想させる
y_pred = model.predict(X_test)

# 正解率を計算
accuracy = accuracy_score(y_test, y_pred)
print(f"学習量10%正解率: {accuracy * 100:.2f}%")

