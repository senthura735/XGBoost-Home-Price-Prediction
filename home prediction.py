import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load the data from CSV file
data = pd.read_csv('home_prices.csv')

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing
# Separate features and target variable
X = data[['income', 'schools', 'hospitals', 'crime_rate']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the data into DMatrix format required by XGBoost
train_dmatrix = xgb.DMatrix(data=X_train_scaled, label=y_train)
test_dmatrix = xgb.DMatrix(data=X_test_scaled, label=y_test)

# Set up the parameters for XGBoost
params = {
    'objective': 'reg:squarederror', # for regression tasks
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10
}

# Train the model
xg_reg = xgb.train(params=params, dtrain=train_dmatrix, num_boost_round=100)

# Predict on the test data
y_pred = xg_reg.predict(test_dmatrix)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')

