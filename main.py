import Hoodhod
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

df = pd.read_csv("TPFDD2_data.csv")
model = linear_model.LinearRegression()
model2 = Hoodhod.Linear_Model.LinearRegressionModel()
X_train, X_test, y_train, y_test = train_test_split(df.X, df.y, test_size=0.20, random_state=0)
model.fit(X_train.values.reshape(-1, 1), y_train)
model2.fit(X_train.values.reshape(-1, 1), y_train)
print("Test score Hoodhod ",model2.score(X_test.values.reshape(-1, 1),y_test))
print("Test score sklearn:",model.score(X_test.values.reshape(-1, 1), y_test))
