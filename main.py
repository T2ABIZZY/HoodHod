import Hoodhod
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

# df = pd.read_csv("TPFDD2_data.csv")
# model = linear_model.LinearRegression()
# model2 = Hoodhod.LinearRegressionModel()
# X_train, X_test, y_train, y_test = train_test_split(df.X, df.y, test_size=0.20, random_state=0)
# model.fit(X_train.values.reshape(-1, 1), y_train)
# model2.fit(X_train.values.reshape(-1, 1), y_train)
# print("Test R2 Hoodhod ",model2.r_squared(X_test.values.reshape(-1, 1),y_test))
# print("Test mean squared error Hoodhod ",model2.mean_squared_error(X_test.values.reshape(-1, 1),y_test))
# print("Test R2 sklearn:",model.score(X_test.values.reshape(-1, 1), y_test))
# print("Test mean squared error sklearn:",mean_squared_error(X_test.values.reshape(-1, 1), y_test))


df = pd.read_csv("TPFDD2_data.csv")
model = linear_model.LinearRegression()
model2 = Hoodhod.LinearRegressionModel()
model.fit(df['X'].values.reshape(-1, 1),df['y'])
model2.fit(df['X'].values.reshape(-1, 1),df['y'])
y_pred=model.predict(df['X'].values.reshape(-1, 1))
print("Test R2 Hoodhod ",model2.r_squared(df['X'].values.reshape(-1, 1),df['y']))
print("Test mean squared error Hoodhod ",model2.mean_squared_error(df['X'].values.reshape(-1, 1),df['y']))
print("Test R2 sklearn:",model.score(df['X'].values.reshape(-1, 1),df['y']))
print("Test mean squared error sklearn:",mean_squared_error(df['y'],y_pred))





# model.fit(df[['x']], df['y'])
# model2.fit(df[['x']], df['y'])
# y_2 = model.predict(df[['x']])
# y_3 = model2.predict(df[['x']])
# df['y_predicted'] = y_2
# df['y_predicted2'] = y_3
# print(df.head())
# mse = mean_squared_error(df['y'], df['y_predicted'])
# mse2 = model2.mean_squared_error(df['x'], df['y'])
# print(mse)
# print(mse2)