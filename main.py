import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv("student_cgpa_dataset.csv")

x = df.drop(columns=["CGPA"])
y = df["CGPA"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = mean_squared_error(y_test, y_pred)
print(metrics)
if metrics < 0.80:
    joblib.dump(model, "counsel_model.pkl")