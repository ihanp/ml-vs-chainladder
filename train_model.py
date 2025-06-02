from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Split into training and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale inputs and targets
input_scaler = StandardScaler()
target_scaler = StandardScaler()

X_tr_scaled = input_scaler.fit_transform(X_tr)
X_val_scaled = input_scaler.transform(X_val)

# Reshape y to be 2D for scaler
y_tr_scaled = target_scaler.fit_transform(y_tr.values.reshape(-1, 1))
y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))

# Train MLP model
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                   max_iter=500, random_state=42)
mlp.fit(X_tr_scaled, y_tr_scaled.ravel())

# Evaluate on validation set
y_pred_scaled = mlp.predict(X_val_scaled)
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"Validation MAE (residual): {mae:.2f}")
print(f"Validation RMSE (residual): {rmse:.2f}")
