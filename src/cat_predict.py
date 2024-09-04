import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# Reading data from a CSV file
data = pd.read_csv('data.csv')

df = pd.DataFrame(data)

df = df.drop(columns=['songname','artist'])


df["genres"] = df["genres"].fillna("")
df["genres"] = df["genres"].str.split(".")
df["popularity"] = df["popularity"].fillna(df["popularity"].median())

# Define features and target
X = df[['temp', 'time', 'dayofweek', 'month']]
y = df['genres']

# Handle multi-label target (genres)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

# Preprocessing pipeline for features
preprocessor = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Define the model pipeline with multi-label support
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(RandomForestClassifier()))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))