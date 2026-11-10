import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pickle

np.random.seed(42)
n = 500
data = {
    'attendance': np.random.randint(40, 101, n),
    'midterm_score': np.random.randint(20, 101, n),
    'homework': np.random.randint(0, 11, n),
    'study_hours': np.random.randint(0, 11, n),
}
df = pd.DataFrame(data)
df['result'] = ((df['attendance']*0.3 + df['midterm_score']*0.5 + df['homework']*5 + df['study_hours']*3) > 65).astype(int)
df.to_csv('students.csv', index=False)
print("Created students.csv")

X = df[['attendance', 'midterm_score', 'homework', 'study_hours']]
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, pred)*100:.1f}%")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as model.pkl - Ready!")