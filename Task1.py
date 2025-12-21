import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv("train.csv")
features = ['GrLivArea', 'FullBath', 'BedroomAbvGr']
a = df[features]
b = df['SalePrice']
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(a_train, b_train)
b_pred = model.predict(a_test)
mse = mean_squared_error(b_test, b_pred)
rmse = mse ** 0.5
print("Mean Squared Error:", mse)
print("RMSE:", rmse)
print("Model Coefficients:", dict(zip(features, model.coef_)))
print("Model Intercept:", model.intercept_)
plt.figure(figsize=(7, 5))
plt.scatter(b_test, b_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([b_test.min(), b_test.max()],[b_test.min(), b_test.max()],linewidth=2)
plt.show()
plt.figure(figsize=(8, 5))
plt.scatter(df["GrLivArea"], df["SalePrice"],color="skyblue", alpha=0.5, label="Actual Data")
x_line = np.linspace(df["GrLivArea"].min(), df["GrLivArea"].max(), 200)
y_line = (model.coef_[0] * x_line) + model.intercept_
plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression Line")
plt.title("Linear Regression Trend: Living Area (sq ft) vs Sale Price")
plt.xlabel("GrLivArea (Square Footage)")
plt.ylabel("SalePrice (Price)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
test_df = pd.read_csv("test.csv")
a_test_final = test_df[features].fillna(0) 
test_pred = model.predict(a_test_final)
submission = pd.DataFrame({"Id": test_df["Id"],"SalePrice": test_pred})
submission.to_csv("submission.csv", index=False)
print("Prediction successfull!")


