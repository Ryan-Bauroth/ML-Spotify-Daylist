import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Reading data from a CSV file
data = pd.read_csv('data.csv')

df = pd.DataFrame(data)

df = df.drop(columns=['songname','artist'])

df["popularity"] = df["popularity"].fillna(df["popularity"].median())

# Define input features and target variables
X = df[['time', 'dayofweek', 'month', 'temp']]
y = df.drop(columns=['time', 'dayofweek', 'month', 'temp', 'genres'])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y ,random_state=42)

# Initialize the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Wrap it in a MultiOutputRegressor to handle multiple targets
multi_output_gbr = MultiOutputRegressor(gbr)

# Train the model
multi_output_gbr.fit(X_train, y_train)

# time, dow, month, temp
#predict_value = X_test
predict_value = pd.DataFrame([["65224", "2", "9", "74.65280151367188"]], columns=["time", "dayofweek", "month", "temp"])

# Make predictions
y_pred = multi_output_gbr.predict(predict_value)

print(y_pred)
print((df.drop(columns=['time', 'dayofweek', 'month', 'temp', 'genres'])).keys())

# Evaluate the model
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(f"Mean Squared Errors for each target: {mse}")

# If you want to see the overall performance, you can use the average MSE
avg_mse = mean_squared_error(y_test, y_pred)
print(f"Average Mean Squared Error: {avg_mse}")

