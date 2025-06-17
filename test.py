from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#　必要なライブラリをインポートしてる

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

print(iris.keys())