import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. Load Data
df = pd.read_csv('creditcard.csv')

# 2. Separate Features and Target
# 'Class' 0 = Legit, 1 = Fraud
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Original class distribution: {y.value_counts()}")

# 3. Handle Imbalance using SMOTE
# It creates synthetic examples of the minority class (fraud)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print(f"Resampled class distribution: {y_res.value_counts()}")

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 5. Train Model (Random Forest is great for this)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
