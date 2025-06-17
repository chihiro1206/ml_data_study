"""from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#必要なライブラリをインポートしてる

from sklearn import datasets

digits = datasets.load_digits()

# 特徴量データ（説明変数）を表示
print(digits.data)

# 正解ラベルを表示
print(digits.target)

# データの形状（サンプル数×特徴量数）
print(digits.data.shape)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

print(iris.keys())
#irisのデータセットのキーを表示

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

# Load as pandas.DataFrame
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target
print(df_iris.describe())
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)"""


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# アヤメのデータを読み込む
iris = load_iris()
X = iris.data       # 特徴（花の長さなど）
y = iris.target     # 答え（花の種類）

# データを「学習用」と「テスト用」に分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 決定木（はい・いいえで分ける木）のモデルを作る
model = DecisionTreeClassifier()

# 学習用データで学習させる（覚えさせる）
model.fit(X_train, y_train)

# テストデータを使って予想させる
y_pred = model.predict(X_test)

print(iris.feature_names)
print(iris.target_names)
"""iris = load_iris()
model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(iris.data, iris.target)

# 決定木を図にする
plt.figure(figsize=(14, 8))
plot_tree(
    model,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    rounded=True
)
plt.show()"""

"""# 正解率を計算
accuracy = accuracy_score(y_test, y_pred)
print(f"正解率: {accuracy * 100:.2f}%")

# 自分で花の情報を入れてみる（例）
my_flower = [[5.0, 3.6, 1.4, 0.2]]  # Setosaっぽいデータ

# 予想させる
prediction = model.predict(my_flower)
print("予想された花の種類:", iris.target_names[prediction[0]])"""