from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import (BayesianRidge, Lasso, LinearRegression,
                                  Ridge, ridge_regression)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

df_ministerio = df_ministerio.reset_index(drop=True)

X = df_ministerio.drop("Sueldo neto", axis=1)
y = df_ministerio["Sueldo neto"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = []

for model in models:
  model.fit(X_train, y_train)
  y_pred = pd.Series(model.predict(X_test))
  y_pred.index = y_test.index
  print(str(model))
  print("MAE: ", mean_absolute_error(y_test, y_pred))
  print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
  print("R2: ", r2_score(y_test, y_pred))