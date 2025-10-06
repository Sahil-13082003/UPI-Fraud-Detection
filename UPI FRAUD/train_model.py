import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv('onlinefraud.csv')
data.fillna(0, inplace=True)

# Encode categorical features
type_encoder = LabelEncoder()
nameOrig_encoder = LabelEncoder()
nameDest_encoder = LabelEncoder()

data['type'] = type_encoder.fit_transform(data['type'])
data['nameOrig'] = nameOrig_encoder.fit_transform(data['nameOrig'])
data['nameDest'] = nameDest_encoder.fit_transform(data['nameDest'])

# Save encoders
joblib.dump(type_encoder, 'type_encoder.sav')
joblib.dump(nameOrig_encoder, 'nameOrig_encoder.sav')
joblib.dump(nameDest_encoder, 'nameDest_encoder.sav')

# Prepare data
X = data.drop(columns=['isFraud', 'isFlaggedFraud'])  # Drop only target & irrelevant
y = data['isFraud']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgb_model.sav')

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, 'lr_model.sav')

# Evaluation
y_pred = xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

