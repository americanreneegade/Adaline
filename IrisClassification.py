import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

import matplotlib.pyplot as plt
import numpy as np

# grabs the first 100 class labels corresponding to setosa and versicolor.
y = df.iloc[0:100, 4].values

# coverts class labels to -1 (for setosa) and 1 (for the rest, i.e., versicolor).
y = np.where(y == 'Iris-setosa', -1, 1)

# grabs sepal length and petal length, two features in columns 0 and 2.
X = df.iloc[0:100, [0,2]].values

# creates Adaline using the class we made
from Adaline import *
ada = Adaline(eta=0.01, n_iter=15)

# standardizes dataset X
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()

# trains the Adaline on our standardized Iris data
ada.fit(X_std, y)

# plots the decision regions and data
from DecisionRegionPlotter import *
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# plots cost function decreasing over epochs
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

# note: sum-squared-error remains non-zero even though
# all samples were correctly classified