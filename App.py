from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

app = Flask(__name__)

# 📌 Load trained model and scaler for reaction yield prediction
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# 📌 Simulated dataset for FCC optimization
np.random.seed(42)
data = pd.DataFrame({
    "Temperature": np.random.uniform(450, 550, 100),  # °C
    "Catalyst_to_Oil_Ratio": np.random.uniform(4, 7, 100),  # wt/wt
    "Residence_Time": np.random.uniform(1, 6, 100),  # sec
})
data["Gasoline_Yield"] = (
    0.4 * np.tanh((data["Temperature"] - 500) / 50) +
    0.3 * np.tanh((data["Catalyst_to_Oil_Ratio"] - 5) / 1.5) +
    0.2 * np.tanh((data["Residence_Time"] - 3) / 2) +
    np.random.normal(0, 0.02, 100)
) * 100  # Convert to percentage

# 📌 Train a predictive model for FCC optimization
X = data[["Temperature", "Catalyst_to_Oil_Ratio", "Residence_Time"]]
y = data["Gasoline_Yield"]
fcc_model = RandomForestRegressor(n_estimators=100, random_state=42)
fcc_model.fit(X, y)

# 📌 Define parameter search space for FCC
space = [
    Real(450, 550, name="Temperature"),
    Real(4, 7, name="Catalyst_to_Oil_Ratio"),
    Real(1, 6, name="Residence_Time"),
]

@use_named_args(space)
def objective(**params):
    X_new = np.array([[params["Temperature"], params["Catalyst_to_Oil_Ratio"], params["Residence_Time"]]])
    return -fcc_model.predict(X_new)[0]  # Minimize negative gasoline yield

# 📌 Homepage Route
@app.route("/")
def home():
    return render_template("index.html")

# 📌 Route for Reaction Yield Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        temperature = float(data["Temperature"])
        pressure = float(data["Pressure"])
        catalyst = float(data["Catalyst"])
        structure = float(data["Structure"])

        # Scale input
        input_data = np.array([[temperature, pressure, catalyst, structure]])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = rf_model.predict(input_scaled)[0]

        return jsonify({"Reaction Yield": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

# 📌 Route for FCC Optimization
@app.route('/optimize', methods=['GET'])
def optimize():
    res = gp_minimize(objective, space, n_calls=30, random_state=42)
    best_params = dict(zip(["Temperature", "Catalyst_to_Oil_Ratio", "Residence_Time"], res.x))
    best_yield = -res.fun  # Convert back to positive yield

    return jsonify({
        "Optimized Parameters": best_params,
        "Expected Gasoline Yield": round(best_yield, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
