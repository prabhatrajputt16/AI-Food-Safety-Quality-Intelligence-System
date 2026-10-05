"""
AI Food Safety & Quality Intelligence System
Dataset Generator — 5000 synthetic food samples
"""

import os, numpy as np, pandas as pd, random
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE, "plots"),  exist_ok=True)

np.random.seed(42)
random.seed(42)
N = 5000

food_categories = ["Dairy", "Meat", "Vegetables", "Fruits", "Grains", "Seafood", "Bakery", "Beverages"]
storage_types   = ["Refrigerated", "Frozen", "Ambient", "Controlled_Atmosphere"]
packaging_types = ["Vacuum", "Modified_Atmosphere", "Standard", "Cryovac"]
seasons         = ["Summer", "Winter", "Spring", "Autumn"]
suppliers       = ["SupplierA", "SupplierB", "SupplierC", "SupplierD", "SupplierE"]

# ── Base features ────────────────────────────────────────────────────────────
food_cat    = np.random.choice(food_categories, N)
storage_t   = np.random.choice(storage_types,   N)
packaging_t = np.random.choice(packaging_types, N)
season      = np.random.choice(seasons,         N)
supplier    = np.random.choice(suppliers,        N)

temperature      = np.round(np.random.normal(5,  8,  N), 2)   # °C
humidity         = np.round(np.random.normal(65, 20, N), 2)   # %
storage_days     = np.random.randint(0, 60, N)
pH               = np.round(np.random.uniform(3.5, 8.5, N), 2)
moisture_percent = np.round(np.random.normal(50, 20, N), 2)
defect_count     = np.random.randint(0, 15, N)
packaging_damage = np.random.randint(0, 5,  N)
odor_score       = np.round(np.random.uniform(0, 10, N), 2)
color_uniformity = np.round(np.random.uniform(0, 1,  N), 3)
texture_score    = np.round(np.random.uniform(0, 10, N), 2)
microbial_load   = np.round(np.random.exponential(1000, N), 0)  # CFU/g
water_activity   = np.round(np.random.uniform(0.1, 0.99, N), 3)
fat_content      = np.round(np.random.uniform(0, 40, N), 2)
protein_content  = np.round(np.random.uniform(0, 35, N), 2)
sugar_content    = np.round(np.random.uniform(0, 50, N), 2)
salt_content     = np.round(np.random.uniform(0, 5,  N), 2)
atp_reading      = np.round(np.random.exponential(200, N), 0)   # RLU
ethylene_ppm     = np.round(np.random.exponential(5,  N), 3)
co2_percent      = np.round(np.random.uniform(0, 15, N), 2)
batch_size       = np.random.randint(50, 5000, N)
transport_hours  = np.round(np.random.exponential(12, N), 1)
cold_chain_break = np.random.randint(0, 3, N)

# ── Clamp to realistic ranges ────────────────────────────────────────────────
temperature      = np.clip(temperature,      -25, 40)
humidity         = np.clip(humidity,           0, 100)
moisture_percent = np.clip(moisture_percent,   0, 100)

# ── Derived / engineered features ───────────────────────────────────────────
freshness_index = np.round(
    10
    - 0.15 * storage_days
    - 0.05 * np.abs(temperature - 4)
    - 0.03 * defect_count
    - 0.2  * packaging_damage
    + 0.1  * color_uniformity * 10,
    2
)
freshness_index = np.clip(freshness_index, 0, 10)

risk_score_raw = (
    0.20 * (storage_days / 60)
    + 0.15 * (np.abs(temperature - 4) / 44)
    + 0.10 * (humidity / 100)
    + 0.10 * (microbial_load / microbial_load.max())
    + 0.10 * (defect_count / 14)
    + 0.08 * (packaging_damage / 4)
    + 0.08 * (cold_chain_break / 2)
    + 0.07 * (water_activity)
    + 0.07 * (atp_reading / atp_reading.max())
    + 0.05 * ((pH - 3.5) / 5)
)
risk_score = np.round(np.clip(risk_score_raw * 10, 0, 10), 2)

quality_score = np.round(np.clip(10 - risk_score + np.random.normal(0, 0.3, N), 0, 10), 2)

shelf_life_days = np.round(
    np.clip(
        30
        - 0.4  * storage_days
        - 0.2  * np.abs(temperature - 4)
        - 0.5  * defect_count
        - 1.0  * packaging_damage
        - 2.0  * cold_chain_break
        + np.random.normal(0, 2, N),
        0, 90
    ), 1
)

# ── Target labels ─────────────────────────────────────────────────────────────
#   Quality Grade: A / B / C / D
def assign_quality_grade(qs):
    if   qs >= 8.0: return "A"
    elif qs >= 6.0: return "B"
    elif qs >= 4.0: return "C"
    else:           return "D"

quality_grade = np.array([assign_quality_grade(q) for q in quality_score])

#   Safety Status: Safe / Monitor / Unsafe
def assign_safety(rs, ml, db):
    if   rs >= 7   or ml > 5000 or db >= 2: return "Unsafe"
    elif rs >= 4.5 or ml > 2000 or db == 1: return "Monitor"
    else:                                    return "Safe"

safety_status = np.array([
    assign_safety(r, m, d)
    for r, m, d in zip(risk_score, microbial_load, cold_chain_break)
])

#   Prediction Reason (top driver)
def top_reason(row):
    scores = {
        "High storage duration":     row["storage_days"] / 60,
        "Temperature deviation":     abs(row["temperature"] - 4) / 44,
        "High microbial load":       row["microbial_load"] / 10000,
        "Packaging damage":          row["packaging_damage"] / 4,
        "Defect detected":           row["defect_count"] / 14,
        "Cold-chain break":          row["cold_chain_break"] / 2,
        "High water activity":       row["water_activity"],
        "pH out of range":           abs(row["pH"] - 6) / 5,
    }
    return max(scores, key=scores.get)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "food_category":     food_cat,
    "storage_type":      storage_t,
    "packaging_type":    packaging_t,
    "season":            season,
    "supplier":          supplier,
    "temperature":       temperature,
    "humidity":          humidity,
    "storage_days":      storage_days,
    "pH":                pH,
    "moisture_percent":  moisture_percent,
    "defect_count":      defect_count,
    "packaging_damage":  packaging_damage,
    "odor_score":        odor_score,
    "color_uniformity":  color_uniformity,
    "texture_score":     texture_score,
    "microbial_load":    microbial_load,
    "water_activity":    water_activity,
    "fat_content":       fat_content,
    "protein_content":   protein_content,
    "sugar_content":     sugar_content,
    "salt_content":      salt_content,
    "atp_reading":       atp_reading,
    "ethylene_ppm":      ethylene_ppm,
    "co2_percent":       co2_percent,
    "batch_size":        batch_size,
    "transport_hours":   transport_hours,
    "cold_chain_break":  cold_chain_break,
    "freshness_index":   freshness_index,
    "risk_score":        risk_score,
    "quality_score":     quality_score,
    "shelf_life_days":   shelf_life_days,
    "quality_grade":     quality_grade,
    "safety_status":     safety_status,
})

df["prediction_reason"] = df.apply(top_reason, axis=1)

df.to_csv(os.path.join(BASE, "food_safety_dataset.csv"), index=False)
print(f"Dataset saved: {len(df)} rows × {len(df.columns)} columns")
print(df["quality_grade"].value_counts())
print(df["safety_status"].value_counts())
