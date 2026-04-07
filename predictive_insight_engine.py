# =========================================================
# IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_excel("RealEstate_HousePrice_Dataset_4200.xlsx")
df.columns = df.columns.str.strip()

print("Columns:", df.columns)


# =========================================================
# PART B: DATA UNDERSTANDING & PREPARATION
# =========================================================

# Handle missing values
df = df.dropna()

# Identify target variable (last column safest)
target_column = df.columns[-1]

# Independent & Dependent Variables
X = df.drop(target_column, axis=1)
y = df[target_column]

print("\nIndependent Variables:\n", X.columns)
print("\nDependent Variable:", target_column)

# Relationship Visualization
sns.pairplot(df)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================================
# SCALING (IMPORTANT FOR GD)
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# PART C: SIMPLE LINEAR REGRESSION
# =========================================================

feature_name = X.columns[0]

X_train_slr = X_train[[feature_name]]
X_test_slr = X_test[[feature_name]]

slr = LinearRegression()
slr.fit(X_train_slr, y_train)

y_pred_slr = slr.predict(X_test_slr)

# Plot regression line
plt.scatter(X_test_slr, y_test)
plt.plot(X_test_slr, y_pred_slr, color='red')
plt.xlabel(feature_name)
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.show()

print("\nSLR Coefficient:", slr.coef_[0])
print("SLR Intercept:", slr.intercept_)


# =========================================================
# PART D: MODEL EVALUATION
# =========================================================

def evaluate(y_true, y_pred, p):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return mse, mae, rmse, r2, adj_r2

slr_metrics = evaluate(y_test, y_pred_slr, 1)

print("\nSLR Metrics:", slr_metrics)


# =========================================================
# PART E: MULTIPLE LINEAR REGRESSION
# =========================================================

mlr = LinearRegression()
mlr.fit(X_train, y_train)

y_pred_mlr = mlr.predict(X_test)

mlr_metrics = evaluate(y_test, y_pred_mlr, X.shape[1])

print("\nMLR Metrics:", mlr_metrics)


# =========================================================
# PART F: POLYNOMIAL REGRESSION
# =========================================================

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

poly_metrics = evaluate(y_test, y_pred_poly, X_train_poly.shape[1])

print("\nPolynomial Metrics:", poly_metrics)


# =========================================================
# PART F (VISUAL COMPARISON)
# =========================================================

plt.scatter(y_test, y_pred_mlr, label="MLR")
plt.scatter(y_test, y_pred_poly, label="Polynomial", alpha=0.6)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear vs Polynomial")
plt.legend()
plt.show()


# =========================================================
# PART G: GRADIENT DESCENT
# =========================================================

X_b = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]
y_b = y_train.values.reshape(-1, 1)

X_test_b = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]


# -------- Batch Gradient Descent --------
theta = np.random.randn(X_b.shape[1], 1)
lr = 0.01
epochs = 500

loss_history = []

for i in range(epochs):
    gradients = (2 / len(X_b)) * X_b.T.dot(X_b.dot(theta) - y_b)
    theta -= lr * gradients
    
    loss = np.mean((X_b.dot(theta) - y_b)**2)
    loss_history.append(loss)

y_pred_gd = X_test_b.dot(theta).flatten()

gd_metrics = evaluate(y_test.values, y_pred_gd, X.shape[1])
print("\nBatch GD Metrics:", gd_metrics)


# Plot convergence
plt.plot(loss_history)
plt.title("Batch GD Convergence")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()


# -------- Stochastic GD --------
theta_sgd = np.random.randn(X_b.shape[1], 1)

for epoch in range(50):
    for i in range(len(X_b)):
        rand_idx = np.random.randint(len(X_b))
        xi = X_b[rand_idx:rand_idx+1]
        yi = y_b[rand_idx:rand_idx+1]

        gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
        theta_sgd -= 0.01 * gradients

y_pred_sgd = X_test_b.dot(theta_sgd).flatten()
sgd_metrics = evaluate(y_test.values, y_pred_sgd, X.shape[1])

print("\nSGD Metrics:", sgd_metrics)


# -------- Mini-Batch GD --------
theta_mgd = np.random.randn(X_b.shape[1], 1)
batch_size = 32

for epoch in range(100):
    for i in range(0, len(X_b), batch_size):
        xi = X_b[i:i+batch_size]
        yi = y_b[i:i+batch_size]

        gradients = (2 / len(xi)) * xi.T.dot(xi.dot(theta_mgd) - yi)
        theta_mgd -= 0.01 * gradients

y_pred_mgd = X_test_b.dot(theta_mgd).flatten()
mgd_metrics = evaluate(y_test.values, y_pred_mgd, X.shape[1])

print("\nMini-Batch GD Metrics:", mgd_metrics)


# =========================================================
# PART G: COMPARISON
# =========================================================

print("\nGradient Descent Comparison:")
print("Batch GD:", gd_metrics[2])
print("SGD:", sgd_metrics[2])
print("Mini-Batch GD:", mgd_metrics[2])


# =========================================================
# PART H: BIAS-VARIANCE ANALYSIS
# =========================================================

print("\nBias-Variance Indicators:")

models = {
    "SLR": slr_metrics,
    "MLR": mlr_metrics,
    "Polynomial": poly_metrics
}

for name, m in models.items():
    print(f"{name} -> R2: {m[3]:.4f}, RMSE: {m[2]:.2f}")


# =========================================================
# FINAL MODEL COMPARISON
# =========================================================

names = list(models.keys())
r2_scores = [models[m][3] for m in names]

plt.bar(names, r2_scores)
plt.title("Final Model Comparison")
plt.ylabel("R2 Score")
plt.show()