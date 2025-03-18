from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load data
path = 'data/creditcard_2023.csv'
try:
    df = pd.read_csv(path)
except Exception as e:
    raise RuntimeError(f"Failed to load data: {e}")

# Divide our dataset into features (X) and target (y)
x = df.drop(['id', 'Class'], axis=1)
y = df['Class']

# Standardize the data
stn_scaler = StandardScaler()
x_scaled = stn_scaler.fit_transform(x)

X = pd.DataFrame(x_scaled, columns=x.columns)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
lin_reg = LogisticRegression()
lin_reg.fit(x_train, y_train)

# Save the trained model to a file
model_file = "ml/fraud_model.joblib"
try:
    model_target_tuple = (lin_reg, list(y_train.unique()))  # Save the unique target values
    joblib.dump(model_target_tuple, model_file)
except Exception as e:
    raise RuntimeError(f"Failed to save model: {e}")

print("Model training complete and saved successfully.")