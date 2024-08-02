import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """Load the dataset from a CSV file."""
    file_path = "C:/Users/User/OneDrive/Desktop/ML_Model/Iris/data/Iris.csv"
    return pd.read_csv(file_path)

def train_model(data):
    """Train a logistic regression model."""
    X = data.drop(columns=['Species'])
    y = data['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Write scores to a file
    with open("metrics.txt", 'w') as outfile:
            outfile.write("Model Accuracy: {accuracy * 100:.2f}%")
           
    return model

if __name__ == "__main__":
    data = load_data("data/iris.csv")
    model = train_model(data)
