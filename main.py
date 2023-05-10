import Hoodhod
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,confusion_matrix,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# X_train = np.array([[0], [0],[0]])
# y_train = np.array([3, 3,5 ])

# X_test = np.array([[0]])
# y_test = np.array([5])

# model = Hoodhod.Linear_Model.LinearRegressionModel()
# model.fit(X_train, y_train)
# y2=model.predict(np.array([[1]]))
# print("predictions for x=1",y2)


# y_pred = model.predict(X_test)

# r2 = r2_score(y_test, y_pred)

# print("Predicted value:", y_pred)
# print("R-squared score:", r2)

# X_train = np.array([[0], [0], [0], [0]])
# y_train = np.array([3, 3, 4, 5])
# X_test = np.array([[1]])
# model = linear_model.LinearRegression()
# model.fit(X_train, y_train)
# y2=model.predict(X_test)
# print(y2)
# X_train = np.array([[1]])
# y_train = np.array([2])
# X_test = np.array([[3]])
# model = Hoodhod.Linear_Model.LinearRegressionModel()
# model.fit(X_train, y_train)
# y2=model.predict(X_test)
# print(y2)

# X_train = np.array([[1],[2],[3],[5],[6]])
# y_train = np.array([2,3.5,4,6.7,7.3])
# X_test = np.array([[7]])
# model = Hoodhod.Linear_Model.LinearRegressionModel()
# model.fit(X_train, y_train)
# y2=model.predict(X_test)
# print(y2)

# model = linear_model.LinearRegression()
# model2 = Hoodhod.Linear_Model.LinearRegressionModel()

# model.fit(X_train.values.reshape(-1, 1), y_train)
# model2.fit(X_train.values.reshape(-1, 1), y_train)

# print("Test score Hoodhod ",model2.score(X_test.values.reshape(-1, 1),y_test))
# print("Test score sklearn:",model.score(X_test.values.reshape(-1, 1), y_test))

# y1=model.predict(X_test.values.reshape(-1, 1))
# y2=model2.predict(X_test.values.reshape(-1, 1))

# print("mean squared error with sklearn",mean_squared_error(y_test, y1))
# print("mean squared error with Hoodhod",Hoodhod.metrices.mean_squared_error(y_test, y2))

# print("r2 score with sklearn",r2_score(y_test, y1))
# 


# print('----------------------------------------------------')

X_train = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
y_train = np.array([0, 1, 0, 1])
X_test = np.array([[1, 1], [0, 0],[1,0]])

clf = Hoodhod.BinaryDecisionTree(max_depth=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Predictions:", y_pred)

X_train = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0], [1, 0, 1, 1],
                    [0, 1, 1, 1], [0, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 0]])
y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])



clf = Hoodhod.BinaryDecisionTree(max_depth=3)
clf.fit(X_train, y_train)
X_test = np.array([[1, 1, 0, 0]])
y_pred = clf.predict(X_test)
print("second pred",y_pred)


