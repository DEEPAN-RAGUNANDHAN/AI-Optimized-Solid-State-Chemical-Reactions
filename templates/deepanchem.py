import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("solid_state_reaction_large.csv")

# Encode categorical variables
label_encoders = {}
for col in ["Reactants", "Product", "Catalyst Used", "Crystal Structure"]:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Define features and target
X = df.drop(columns=["Reaction Yield (%)", "Reaction Efficiency (%)"])
y = df["Reaction Yield (%)"]  # Predicting reaction yield

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train an MLP Regressor (Neural Network) model
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# Predict and evaluate
rf_pred = rf_model.predict(X_test)
mlp_pred = mlp_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
mlp_mae = mean_absolute_error(y_test, mlp_pred)
mlp_r2 = r2_score(y_test, mlp_pred)

# Visualization: 3D scatter plot of Temperature, Pressure, and Reaction Yield
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["Temperature (°C)"], df["Pressure (atm)"], df["Reaction Yield (%)"], c=df["Reaction Yield (%)"], cmap='viridis', marker='o')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Pressure (atm)')
ax.set_zlabel('Reaction Yield (%)')
ax.set_title('3D Visualization of Reaction Conditions')
plt.show()

# Crystal Structure Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="Crystal Structure", data=df, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Distribution of Crystal Structures in Dataset")
plt.show()

# Display results
print(f"Random Forest - Mean Absolute Error: {rf_mae}, R-squared Score: {rf_r2}")
print(f"MLP Regressor - Mean Absolute Error: {mlp_mae}, R-squared Score: {mlp_r2}")
print(df.head())
