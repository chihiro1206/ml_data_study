from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
wine = load_wine()
X = wine.data
y = wine.target

# テストデータの固定　ここで学習データとテストは分離する
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 上で分離した学習データの中からさらに学習に使う割合を指定する
train_frac = 0.1

sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=0)
for train_idx, _ in sss.split(X_train_full, y_train_full):
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 固定テストデータで予測・評価
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"学習割合{int(train_frac*100)}%のとき正解率: {acc*100:.2f}%")

"""plt.figure(figsize=(12,8))
plot_tree(model,filled=True,feature_names=wine.feature_names,class_names=wine.target_names)
plt.show()"""