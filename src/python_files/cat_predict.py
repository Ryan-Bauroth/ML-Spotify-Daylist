import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def train_random_forest_classifier(features, target):

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
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the model
    model.fit(x_train, y_train)

    return model, x_test, y_test

if __name__ == "__main__":
    # Reading data from a CSV file
    data = pd.read_csv('../data.csv')

    df = pd.DataFrame(data)

    df = df.drop(columns=['songname', 'artist'])

    df["genres"] = df["genres"].fillna("")
    df["genres"] = df["genres"].str.split(".")
    df["popularity"] = df["popularity"].fillna(df["popularity"].median())

    # Define features and target
    x = df[['temp', 'time', 'dayofweek', 'month']]
    y = df['genres']

    # Handle multi-label target (genres)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)

    model, x_test, y_test = train_random_forest_classifier(x, y)

    # Predict and evaluate
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))