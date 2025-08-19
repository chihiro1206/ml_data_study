from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data
y = iris.target

# 学習データとテストデータの分離
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 上で分離した学習データの中からさらに学習に使う割合を指定する
train_frac = 0.1  

sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=0)
for train_idx, _ in sss.split(X_train_full, y_train_full):
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]


model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"学習割合{int(train_frac*100)}%のとき正解率: {acc*100:.2f}%")
