import Hoodhod
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,confusion_matrix,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.read_csv("TPFDD2_data.csv")
X_train, X_test, y_train, y_test = train_test_split(df.X, df.y, test_size=0.20, random_state=0)

model = linear_model.LinearRegression()
model2 = Hoodhod.Linear_Model.LinearRegressionModel()

model.fit(X_train.values.reshape(-1, 1), y_train)
model2.fit(X_train.values.reshape(-1, 1), y_train)

print("Test score Hoodhod ",model2.score(X_test.values.reshape(-1, 1),y_test))
print("Test score sklearn:",model.score(X_test.values.reshape(-1, 1), y_test))

y1=model.predict(X_test.values.reshape(-1, 1))
y2=model2.predict(X_test.values.reshape(-1, 1))

print("mean squared error with sklearn",mean_squared_error(y_test, y1))
print("mean squared error with Hoodhod",Hoodhod.metrices.mean_squared_error(y_test, y2))

print("r2 score with sklearn",r2_score(y_test, y1))
print("r2 score with Hoodhod",Hoodhod.metrices.r_squared(y_test, y2))

print('----------------------------------------------------')

X = iris.data 
y = iris.target 
y = np.where(y == 0, 0, 1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier=DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

clf = Hoodhod.BinaryDecisionTree(max_depth=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)


 