import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn import metrics


boston_data = datasets.load_boston(return_X_y=False)
X = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
y = pd.Series(boston_data.target)

train_size = 0.3

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.3, shuffle=False)

test_size = 0.8571
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rem, y_rem, test_size=0.8571, shuffle=False
)

epochs = 10000

# # Pipeline to normalise
# reg = make_pipeline(StandardScaler(),
#                     linear_model.SGDRegressor(max_iter = epochs))

reg = linear_model.SGDRegressor(max_iter=epochs)

reg.fit(X_train, y_train)

# Validation set
y_valid_pred = reg.predict(X_valid)

valid_mse = metrics.mean_squared_error(y_valid, y_valid_pred)

print("Valid MSE: ", valid_mse.round(3))

# Test set
y_test_pred = reg.predict(X_test)

test_mse = metrics.mean_squared_error(y_test, y_test_pred)

print("Test MSE: ", test_mse.round(3))
