import pandas as pd

# Load dataset (example CSV file from Materials Project)
df = pd.read_csv("solid_state_materials.csv")

# Selecting important features
selected_features = [
    "material_id", "formula", "crystal_system", "space_group_symbol", 
    "energy_above_hull", "formation_energy", "predicted_stable", 
    "volume", "density", "band_gap", "is_gap_direct", "is_metal",
    "bulk_modulus_voigt", "shear_modulus_voigt", "elastic_anisotropy",
    "weighted_surface_energy", "surface_anisotropy", "shape_factor", 
    "work_function", "piezoelectric_modulus", "total_dielectric_constant"
]

df_selected = df[selected_features]
df_selected.head()
df_selected.fillna(df_selected.mean(), inplace=True)  # Replace NaNs with mean values
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_selected.iloc[:, 4:]), columns=df_selected.columns[4:])
df_scaled.insert(0, "material_id", df_selected["material_id"])
df_scaled.insert(1, "formula", df_selected["formula"])
df_scaled.head()
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
le = LabelEncoder()
df_scaled["crystal_system"] = le.fit_transform(df_selected["crystal_system"])
df_scaled["space_group_symbol"] = le.fit_transform(df_selected["space_group_symbol"])
df_scaled.to_csv("processed_solid_state_data.csv", index=False)

