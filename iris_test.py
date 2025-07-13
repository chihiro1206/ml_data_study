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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 決定木モデルを作る
model = DecisionTreeClassifier()

# 学習用データで学習
model.fit(X_train, y_train)

# テストデータを使って予想させる
y_pred = model.predict(X_test)

"""# Split the dataset
data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, test_size=0.5, random_state=0)

# Define Neural Neowork model
clf = MLPClassifier(hidden_layer_sizes=10, activation='relu',
                    solver='adam', max_iter=1000)

# Lerning model
clf.fit(data_train, target_train)


# Show loss curve
plt.plot(clf.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()"""


# 決定木を図にする
plt.figure(figsize=(14, 8))
plot_tree(
    model,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    rounded=True
)
plt.show()

# 正解率を計算
accuracy = accuracy_score(y_test, y_pred)
print(f"正解率: {accuracy * 100:.2f}%")

# 自分で花の情報を入れてみる（例）
my_flower = [[5.0, 3.6, 1.4, 0.2]]  # Setosaっぽいデータ

# 予想させる
prediction = model.predict(my_flower)
print("予想された花の種類:", iris.target_names[prediction[0]])