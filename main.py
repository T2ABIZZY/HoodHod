import Hoodhod
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("TPFDD2_data.csv")
model = linear_model.LinearRegression()
model2 = Hoodhod.Linear_Model.LinearRegressionModel()
X_train, X_test, y_train, y_test = train_test_split(df.X, df.y, test_size=0.20, random_state=0)
model.fit(X_train.values.reshape(-1, 1), y_train)
model2.fit(X_train.values.reshape(-1, 1), y_train)
print("Test score Hoodhod ",model2.score(X_test.values.reshape(-1, 1),y_test))
print("Test score sklearn:",model.score(X_test.values.reshape(-1, 1), y_test))
y1=model.predict(X_test.values.reshape(-1, 1))
y1=model2.predict(X_test.values.reshape(-1, 1))
print("mean squared error with sklearn",mean_squared_error(y_test, y1))
print("mean squared error with Hoodhod",Hoodhod.metrices.mean_squared_error(y_test, y1))
print("r2 score with sklearn",r2_score(y_test, y1))
print("r2 score with Hoodhod",Hoodhod.metrices.r_squared(y_test, y1))