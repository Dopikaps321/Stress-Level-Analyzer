import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\USER\Desktop\stress\stress_level.csv")

# Check for missing values and duplicates
print(data.isnull().sum())
data.dropna(inplace=True)
print(data.duplicated().sum())
data.drop_duplicates(inplace=True)

# Verify data cleaning
print(data.info())

# Define features and target
features = ['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sh', 'hr']
X = data[features]
y = data['sl']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and scaler
joblib.dump(model, 'C:\\Users\\USER\\Desktop\\stress\\stress_level_model.pkl')
joblib.dump(scaler, 'C:\\Users\\USER\\Desktop\\stress\\scaler.pkl')

print("Model and scaler saved successfully.")
