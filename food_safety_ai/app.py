"""
╔══════════════════════════════════════════════════════════════════╗
║   AI Food Safety & Quality Intelligence System                   ║
║   Powered by XGBoost  |  Streamlit UI                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Food Safety Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    color: #f1f5f9;
}
[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-card .val { font-size: 2rem; font-weight: 700; }
.metric-card .lbl { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }

/* Result badges */
.badge-safe    { background:#16a34a; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.badge-monitor { background:#d97706; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.badge-unsafe  { background:#dc2626; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.badge-a { background:#16a34a; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.badge-b { background:#2563eb; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.badge-c { background:#d97706; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.badge-d { background:#dc2626; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; font-size:1.1rem; }

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e293b;
    border-left: 4px solid #2563eb;
    padding-left: 12px;
    margin: 20px 0 14px 0;
}

/* Top header banner */
.top-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 28px 40px;
    border-radius: 16px;
    margin-bottom: 28px;
    border: 1px solid #1e40af;
}
.top-banner h1 { color: #f8fafc; font-size: 2.2rem; margin: 0; font-weight: 800; }
.top-banner p  { color: #94a3b8; margin: 6px 0 0 0; font-size: 1rem; }

/* Gauge container */
.gauge-label { text-align:center; font-size:0.85rem; color:#64748b; margin-top:-12px; }

/* Info box */
.info-box {
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
    padding:14px 18px; margin:10px 0; color:#1e40af; font-size:0.9rem;
}
.warn-box {
    background:#fffbeb; border:1px solid #fde68a; border-radius:10px;
    padding:14px 18px; margin:10px 0; color:#92400e; font-size:0.9rem;
}
.danger-box {
    background:#fef2f2; border:1px solid #fecaca; border-radius:10px;
    padding:14px 18px; margin:10px 0; color:#991b1b; font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_artefacts():
    models_dir = os.path.join(BASE, "models")
    return {
        "model_q":  joblib.load(os.path.join(models_dir, "model_quality.pkl")),
        "model_s":  joblib.load(os.path.join(models_dir, "model_safety.pkl")),
        "scaler":   joblib.load(os.path.join(models_dir, "scaler.pkl")),
        "le_enc":   joblib.load(os.path.join(models_dir, "label_encoders.pkl")),
        "le_q":     joblib.load(os.path.join(models_dir, "le_quality.pkl")),
        "le_s":     joblib.load(os.path.join(models_dir, "le_safety.pkl")),
        "feat":     joblib.load(os.path.join(models_dir, "feature_cols.pkl")),
    }

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "food_safety_dataset.csv"))

@st.cache_data
def load_metrics():
    with open(os.path.join(BASE, "models", "metrics.json")) as f:
        return json.load(f)

try:
    art     = load_artefacts()
    df_full = load_data()
    metrics = load_metrics()
    LOADED  = True
except Exception as e:
    LOADED = False
    LOAD_ERR = str(e)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 FoodSafe AI")
    st.markdown("---")
    st.markdown("### Navigation")
    tab_choice = st.radio(
        "",
        ["🏠 Dashboard", "🔬 Predict", "📊 Model Metrics", "📈 Data Explorer", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### Quick Stats")
    if LOADED:
        st.metric("Dataset Size",  f"{len(df_full):,} samples")
        st.metric("Features Used", f"{len(art['feat'])}")
        st.metric("Model",         "XGBoost")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#475569;'>AI Food Safety & Quality Intelligence System v1.0</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if tab_choice == "🏠 Dashboard":
    st.markdown("""
    <div class="top-banner">
        <h1>🧬 AI Food Safety & Quality Intelligence System</h1>
        <p>XGBoost-powered platform for real-time food quality grading, safety risk detection, shelf-life estimation & prediction explanation</p>
    </div>
    """, unsafe_allow_html=True)

    if not LOADED:
        st.error(f"Model not found. Please run `train_model.py` first.\n\n`{LOAD_ERR}`")
        st.stop()

    m = metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (c1, "Quality Accuracy",  f"{m['quality']['accuracy']*100:.1f}%", "#22c55e"),
        (c2, "Quality F1 Score",  f"{m['quality']['f1']*100:.1f}%",       "#3b82f6"),
        (c3, "Safety Accuracy",   f"{m['safety']['accuracy']*100:.1f}%",  "#f59e0b"),
        (c4, "Safety F1 Score",   f"{m['safety']['f1']*100:.1f}%",        "#8b5cf6"),
        (c5, "ROC-AUC (Quality)", f"{m['quality']['auc']*100:.1f}%",      "#ef4444"),
        (c6, "ROC-AUC (Safety)",  f"{m['safety']['auc']*100:.1f}%",       "#06b6d4"),
    ]
    for col, label, val, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val" style="color:{color}">{val}</div>
                <div class="lbl">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Quality Grade Distribution</div>', unsafe_allow_html=True)
        grade_counts = df_full["quality_grade"].value_counts().reset_index()
        grade_counts.columns = ["Grade", "Count"]
        color_map = {"A": "#16a34a", "B": "#2563eb", "C": "#f59e0b", "D": "#dc2626"}
        fig = px.bar(grade_counts, x="Grade", y="Count",
                     color="Grade", color_discrete_map=color_map,
                     text="Count", template="plotly_white")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=300)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Safety Status Distribution</div>', unsafe_allow_html=True)
        safety_counts = df_full["safety_status"].value_counts().reset_index()
        safety_counts.columns = ["Status", "Count"]
        sc_map = {"Safe": "#16a34a", "Monitor": "#f59e0b", "Unsafe": "#dc2626"}
        fig2 = px.pie(safety_counts, names="Status", values="Count",
                      color="Status", color_discrete_map=sc_map,
                      hole=0.45, template="plotly_white")
        fig2.update_layout(margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown('<div class="section-header">Risk Score vs Quality Score</div>', unsafe_allow_html=True)
        sample = df_full.sample(500, random_state=1)
        fig3 = px.scatter(sample, x="risk_score", y="quality_score",
                          color="safety_status", size="storage_days",
                          color_discrete_map=sc_map,
                          template="plotly_white", opacity=0.7,
                          labels={"risk_score": "Risk Score", "quality_score": "Quality Score"})
        fig3.update_layout(margin=dict(t=10, b=10), height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-header">Shelf Life by Food Category</div>', unsafe_allow_html=True)
        fig4 = px.box(df_full, x="food_category", y="shelf_life_days",
                      color="food_category", template="plotly_white",
                      labels={"food_category": "Category", "shelf_life_days": "Shelf Life (days)"})
        fig4.update_layout(showlegend=False, margin=dict(t=10, b=10), height=320,
                           xaxis_tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)

    # Confusion matrix images
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    cm1_path = os.path.join(BASE, "plots", "cm_quality.png")
    cm2_path = os.path.join(BASE, "plots", "cm_safety.png")
    ci1, ci2 = st.columns(2)
    if os.path.exists(cm1_path) and os.path.exists(cm2_path):
        with ci1:
            st.image(cm1_path, caption="Quality Grade Confusion Matrix", use_container_width=True)
        with ci2:
            st.image(cm2_path, caption="Safety Status Confusion Matrix", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "🔬 Predict":
    st.markdown("""
    <div class="top-banner">
        <h1>🔬 Real-Time Food Quality Prediction</h1>
        <p>Enter food sample parameters below to get instant quality grade, safety status, risk score, shelf-life estimate & prediction reason</p>
    </div>
    """, unsafe_allow_html=True)

    if not LOADED:
        st.error("Model artefacts missing. Run `train_model.py` first.")
        st.stop()

    with st.form("predict_form"):
        st.markdown('<div class="section-header">🥩 Product Information</div>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        food_category  = r1c1.selectbox("Food Category",  ["Dairy","Meat","Vegetables","Fruits","Grains","Seafood","Bakery","Beverages"])
        storage_type   = r1c2.selectbox("Storage Type",   ["Refrigerated","Frozen","Ambient","Controlled_Atmosphere"])
        packaging_type = r1c3.selectbox("Packaging Type", ["Vacuum","Modified_Atmosphere","Standard","Cryovac"])
        supplier       = r1c4.selectbox("Supplier",       ["SupplierA","SupplierB","SupplierC","SupplierD","SupplierE"])

        st.markdown('<div class="section-header">🌡️ Environmental & Storage Conditions</div>', unsafe_allow_html=True)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        temperature     = r2c1.slider("Temperature (°C)",    -25.0, 40.0, 4.0, 0.5)
        humidity        = r2c2.slider("Humidity (%)",          0.0,100.0,65.0, 1.0)
        storage_days    = r2c3.slider("Storage Days",          0,    60,   7)
        cold_chain_break= r2c4.slider("Cold Chain Breaks",     0,    3,    0)

        st.markdown('<div class="section-header">🧪 Chemical & Physical Parameters</div>', unsafe_allow_html=True)
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        pH              = r3c1.slider("pH Level",              3.5,  8.5,  6.5, 0.1)
        moisture_pct    = r3c2.slider("Moisture (%)",          0.0, 100.0, 50.0, 0.5)
        water_activity  = r3c3.slider("Water Activity",        0.1,  0.99, 0.85, 0.01)
        microbial_load  = r3c4.number_input("Microbial Load (CFU/g)", 0, 100000, 500, 100)

        r4c1, r4c2, r4c3, r4c4 = st.columns(4)
        fat_content     = r4c1.slider("Fat Content (%)",       0.0,  40.0, 5.0,  0.5)
        protein_content = r4c2.slider("Protein Content (%)",   0.0,  35.0, 10.0, 0.5)
        sugar_content   = r4c3.slider("Sugar Content (%)",     0.0,  50.0, 8.0,  0.5)
        salt_content    = r4c4.slider("Salt Content (%)",      0.0,   5.0, 1.0,  0.1)

        st.markdown('<div class="section-header">🔍 Quality Inspection</div>', unsafe_allow_html=True)
        r5c1, r5c2, r5c3, r5c4 = st.columns(4)
        defect_count    = r5c1.slider("Defect Count",          0,   15,  0)
        packaging_damage= r5c2.slider("Packaging Damage",      0,    5,  0)
        odor_score      = r5c3.slider("Odor Score (0-10)",     0.0,10.0, 8.0, 0.1)
        color_uniformity= r5c4.slider("Color Uniformity (0-1)",0.0, 1.0, 0.9, 0.01)

        r6c1, r6c2, r6c3, r6c4 = st.columns(4)
        texture_score   = r6c1.slider("Texture Score (0-10)",  0.0,10.0, 7.5, 0.1)
        atp_reading     = r6c2.number_input("ATP Reading (RLU)", 0, 10000, 100, 10)
        ethylene_ppm    = r6c3.slider("Ethylene (ppm)",        0.0, 50.0,  1.0, 0.1)
        co2_percent     = r6c4.slider("CO₂ (%)",               0.0, 15.0,  2.0, 0.1)

        r7c1, r7c2, r7c3 = st.columns(3)
        batch_size      = r7c1.number_input("Batch Size",      50, 5000, 500, 50)
        transport_hours = r7c2.slider("Transport Hours",       0.0, 120.0, 6.0, 0.5)
        season          = r7c3.selectbox("Season",             ["Summer","Winter","Spring","Autumn"])

        submitted = st.form_submit_button("🚀 Run AI Prediction", use_container_width=True)

    if submitted:
        # ── Build feature vector ──────────────────────────────────────────────
        CAT_COLS = ["food_category","storage_type","packaging_type","season","supplier","prediction_reason"]
        le_enc   = art["le_enc"]

        # Derived features
        freshness_index = max(0, min(10,
            10 - 0.15*storage_days - 0.05*abs(temperature-4)
            - 0.03*defect_count - 0.2*packaging_damage + 0.1*color_uniformity*10
        ))
        risk_score_raw = (
            0.20*(storage_days/60) + 0.15*(abs(temperature-4)/44)
            + 0.10*(humidity/100)  + 0.10*(microbial_load/100000)
            + 0.10*(defect_count/14)+ 0.08*(packaging_damage/4)
            + 0.08*(cold_chain_break/2)+ 0.07*(water_activity)
            + 0.07*(atp_reading/10000)+ 0.05*((pH-3.5)/5)
        )
        risk_score    = round(min(10, max(0, risk_score_raw*10)), 2)
        quality_score = round(min(10, max(0, 10 - risk_score)), 2)
        shelf_life    = round(max(0, 30 - 0.4*storage_days - 0.2*abs(temperature-4)
                                  - 0.5*defect_count - 1.0*packaging_damage
                                  - 2.0*cold_chain_break), 1)

        # Top reason
        reasons = {
            "High storage duration":  storage_days/60,
            "Temperature deviation":  abs(temperature-4)/44,
            "High microbial load":    microbial_load/100000,
            "Packaging damage":       packaging_damage/4,
            "Defect detected":        defect_count/14,
            "Cold-chain break":       cold_chain_break/2,
            "High water activity":    water_activity,
            "pH out of range":        abs(pH-6)/5,
        }
        pred_reason = max(reasons, key=reasons.get)

        # Encode categoricals
        def safe_encode(le, val):
            try:    return int(le.transform([val])[0])
            except: return 0

        enc_vals = {
            "food_category_enc":    safe_encode(le_enc["food_category"],    food_category),
            "storage_type_enc":     safe_encode(le_enc["storage_type"],     storage_type),
            "packaging_type_enc":   safe_encode(le_enc["packaging_type"],   packaging_type),
            "season_enc":           safe_encode(le_enc["season"],           season),
            "supplier_enc":         safe_encode(le_enc["supplier"],         supplier),
            "prediction_reason_enc":safe_encode(le_enc["prediction_reason"],pred_reason),
        }

        NUM_VALS = {
            "temperature": temperature, "humidity": humidity,
            "storage_days": storage_days, "pH": pH,
            "moisture_percent": moisture_pct, "defect_count": defect_count,
            "packaging_damage": packaging_damage, "odor_score": odor_score,
            "color_uniformity": color_uniformity, "texture_score": texture_score,
            "microbial_load": microbial_load, "water_activity": water_activity,
            "fat_content": fat_content, "protein_content": protein_content,
            "sugar_content": sugar_content, "salt_content": salt_content,
            "atp_reading": atp_reading, "ethylene_ppm": ethylene_ppm,
            "co2_percent": co2_percent, "batch_size": batch_size,
            "transport_hours": transport_hours, "cold_chain_break": cold_chain_break,
            "freshness_index": freshness_index, "risk_score": risk_score,
            "quality_score": quality_score, "shelf_life_days": shelf_life,
        }

        feat_dict = {**NUM_VALS, **enc_vals}
        X_row = np.array([[feat_dict[c] for c in art["feat"]]])
        X_sc  = art["scaler"].transform(X_row)

        grade_idx  = art["model_q"].predict(X_sc)[0]
        safety_idx = art["model_s"].predict(X_sc)[0]
        grade_prob = art["model_q"].predict_proba(X_sc)[0]
        safety_prob= art["model_s"].predict_proba(X_sc)[0]

        grade  = art["le_q"].inverse_transform([grade_idx])[0]
        safety = art["le_s"].inverse_transform([safety_idx])[0]

        # ── Results ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">📋 Prediction Results</div>', unsafe_allow_html=True)

        rb1, rb2, rb3, rb4 = st.columns(4)
        badge_grade  = f"badge-{grade.lower()}"
        badge_safety = f"badge-{safety.lower().replace(' ','')}"
        rb1.markdown(f"**Quality Grade**<br><span class='{badge_grade}'>{grade}</span>",   unsafe_allow_html=True)
        rb2.markdown(f"**Safety Status**<br><span class='{badge_safety}'>{safety}</span>", unsafe_allow_html=True)
        rb3.markdown(f"**Risk Score**<br><span style='font-size:1.8rem;font-weight:700;color:#ef4444'>{risk_score}/10</span>",   unsafe_allow_html=True)
        rb4.markdown(f"**Est. Shelf Life**<br><span style='font-size:1.8rem;font-weight:700;color:#2563eb'>{shelf_life}d</span>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauges
        gc1, gc2, gc3 = st.columns(3)
        def make_gauge(val, max_val, title, color, suffix=""):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val,
                title={"text": title, "font": {"size": 14}},
                number={"suffix": suffix, "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, max_val], "tickwidth": 1},
                    "bar": {"color": color},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, max_val*0.33], "color": "#dcfce7"},
                        {"range": [max_val*0.33, max_val*0.66], "color": "#fef9c3"},
                        {"range": [max_val*0.66, max_val], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.75, "value": val},
                }
            ))
            fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
            return fig

        gc1.plotly_chart(make_gauge(risk_score,       10,  "Risk Score",       "#ef4444"), use_container_width=True)
        gc2.plotly_chart(make_gauge(quality_score,    10,  "Quality Score",    "#22c55e"), use_container_width=True)
        gc3.plotly_chart(make_gauge(freshness_index,  10,  "Freshness Index",  "#3b82f6"), use_container_width=True)

        # Confidence bars
        st.markdown('<div class="section-header">📊 Prediction Confidence</div>', unsafe_allow_html=True)
        pb1, pb2 = st.columns(2)
        with pb1:
            st.markdown("**Quality Grade Confidence**")
            grade_labels = art["le_q"].classes_
            for lbl, prob in zip(grade_labels, grade_prob):
                col_map = {"A":"#16a34a","B":"#2563eb","C":"#f59e0b","D":"#dc2626"}
                bar_color = col_map.get(lbl, "#64748b")
                st.markdown(f"""
                <div style="display:flex;align-items:center;margin:4px 0">
                  <span style="width:30px;font-weight:700;color:{bar_color}">{lbl}</span>
                  <div style="flex:1;background:#e2e8f0;border-radius:6px;height:22px;margin:0 10px">
                    <div style="width:{prob*100:.1f}%;background:{bar_color};height:100%;border-radius:6px;
                                display:flex;align-items:center;padding-left:8px;color:white;font-size:0.8rem">
                      {prob*100:.1f}%
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        with pb2:
            st.markdown("**Safety Status Confidence**")
            safety_labels = art["le_s"].classes_
            for lbl, prob in zip(safety_labels, safety_prob):
                sc_map2 = {"Safe":"#16a34a","Monitor":"#f59e0b","Unsafe":"#dc2626"}
                bar_color = sc_map2.get(lbl, "#64748b")
                st.markdown(f"""
                <div style="display:flex;align-items:center;margin:4px 0">
                  <span style="width:80px;font-weight:700;color:{bar_color}">{lbl}</span>
                  <div style="flex:1;background:#e2e8f0;border-radius:6px;height:22px;margin:0 10px">
                    <div style="width:{prob*100:.1f}%;background:{bar_color};height:100%;border-radius:6px;
                                display:flex;align-items:center;padding-left:8px;color:white;font-size:0.8rem">
                      {prob*100:.1f}%
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # Reason & Advisory
        st.markdown('<div class="section-header">🧠 AI Prediction Reason & Advisory</div>', unsafe_allow_html=True)
        reason_col, advisory_col = st.columns(2)
        with reason_col:
            st.markdown(f"""
            <div class="info-box">
                <b>🔍 Primary Prediction Driver:</b><br>
                <span style="font-size:1.1rem">{pred_reason}</span><br><br>
                <b>Top Risk Factors:</b><br>
                {"<br>".join([f"• {k}: {v*100:.1f}%" for k,v in sorted(reasons.items(), key=lambda x:-x[1])[:4]])}
            </div>""", unsafe_allow_html=True)

        with advisory_col:
            if safety == "Unsafe":
                st.markdown(f"""
                <div class="danger-box">
                    <b>⛔ UNSAFE — DO NOT DISTRIBUTE</b><br>
                    This product poses a significant food safety risk.<br>
                    Estimated shelf life remaining: <b>{shelf_life} days</b><br>
                    Recommended action: <b>Quarantine and dispose immediately.</b>
                </div>""", unsafe_allow_html=True)
            elif safety == "Monitor":
                st.markdown(f"""
                <div class="warn-box">
                    <b>⚠️ MONITOR CLOSELY</b><br>
                    Product shows early signs of quality degradation.<br>
                    Estimated shelf life remaining: <b>{shelf_life} days</b><br>
                    Recommended action: <b>Increase inspection frequency. Use soon.</b>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                    <b>✅ SAFE FOR DISTRIBUTION</b><br>
                    Product meets safety and quality standards.<br>
                    Estimated shelf life remaining: <b>{shelf_life} days</b><br>
                    Recommended action: <b>Approved for sale/distribution.</b>
                </div>""", unsafe_allow_html=True)

        # Risk breakdown radar
        st.markdown('<div class="section-header">📡 Risk Factor Radar</div>', unsafe_allow_html=True)
        radar_cats  = list(reasons.keys())
        radar_vals  = [v*10 for v in reasons.values()]
        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself", fillcolor="rgba(239,68,68,0.15)",
            line=dict(color="#ef4444", width=2),
            name="Risk Factors"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=False, height=400,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📊 Model Metrics":
    st.markdown("""
    <div class="top-banner">
        <h1>📊 Model Performance Metrics</h1>
        <p>XGBoost model evaluation — accuracy, F1, recall, precision, AUC and cross-validation results</p>
    </div>
    """, unsafe_allow_html=True)

    if not LOADED:
        st.error("Model artefacts missing.")
        st.stop()

    m = metrics
    st.markdown('<div class="section-header">🏆 Quality Grade Model (A/B/C/D)</div>', unsafe_allow_html=True)
    qc1,qc2,qc3,qc4,qc5 = st.columns(5)
    for col, lbl, val, color in [
        (qc1,"Accuracy",  f"{m['quality']['accuracy']*100:.2f}%",  "#22c55e"),
        (qc2,"F1 Score",  f"{m['quality']['f1']*100:.2f}%",        "#3b82f6"),
        (qc3,"Recall",    f"{m['quality']['recall']*100:.2f}%",     "#f59e0b"),
        (qc4,"Precision", f"{m['quality']['precision']*100:.2f}%",  "#8b5cf6"),
        (qc5,"ROC-AUC",   f"{m['quality']['auc']*100:.2f}%",       "#ef4444"),
    ]:
        col.markdown(f"""<div class="metric-card"><div class="val" style="color:{color}">{val}</div><div class="lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <br><div class="info-box">
    📊 <b>5-Fold Cross-Validation F1 (Quality):</b>
    {m['quality']['cv_f1_mean']*100:.2f}% ± {m['quality']['cv_f1_std']*100:.2f}%
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🛡️ Safety Status Model (Safe/Monitor/Unsafe)</div>', unsafe_allow_html=True)
    sc1,sc2,sc3,sc4,sc5 = st.columns(5)
    for col, lbl, val, color in [
        (sc1,"Accuracy",  f"{m['safety']['accuracy']*100:.2f}%",   "#22c55e"),
        (sc2,"F1 Score",  f"{m['safety']['f1']*100:.2f}%",         "#3b82f6"),
        (sc3,"Recall",    f"{m['safety']['recall']*100:.2f}%",      "#f59e0b"),
        (sc4,"Precision", f"{m['safety']['precision']*100:.2f}%",   "#8b5cf6"),
        (sc5,"ROC-AUC",   f"{m['safety']['auc']*100:.2f}%",        "#ef4444"),
    ]:
        col.markdown(f"""<div class="metric-card"><div class="val" style="color:{color}">{val}</div><div class="lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <br><div class="info-box">
    📊 <b>5-Fold Cross-Validation F1 (Safety):</b>
    {m['safety']['cv_f1_mean']*100:.2f}% ± {m['safety']['cv_f1_std']*100:.2f}%
    </div>""", unsafe_allow_html=True)

    # Metric comparison bar chart
    st.markdown('<div class="section-header">📈 Side-by-Side Metric Comparison</div>', unsafe_allow_html=True)
    metric_names = ["Accuracy","F1 Score","Recall","Precision","ROC-AUC"]
    q_vals = [m["quality"]["accuracy"], m["quality"]["f1"], m["quality"]["recall"],
              m["quality"]["precision"], m["quality"]["auc"]]
    s_vals = [m["safety"]["accuracy"],  m["safety"]["f1"],  m["safety"]["recall"],
              m["safety"]["precision"],  m["safety"]["auc"]]
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="Quality Grade", x=metric_names, y=[v*100 for v in q_vals],
                             marker_color="#3b82f6", text=[f"{v*100:.1f}%" for v in q_vals],
                             textposition="outside"))
    fig_cmp.add_trace(go.Bar(name="Safety Status", x=metric_names, y=[v*100 for v in s_vals],
                             marker_color="#22c55e", text=[f"{v*100:.1f}%" for v in s_vals],
                             textposition="outside"))
    fig_cmp.update_layout(barmode="group", yaxis=dict(range=[0,110], title="Score (%)"),
                          template="plotly_white", height=380,
                          legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Feature importance plots
    st.markdown('<div class="section-header">🔑 Feature Importance</div>', unsafe_allow_html=True)
    fi1, fi2 = st.columns(2)
    fi1_path = os.path.join(BASE, "plots", "fi_quality.png")
    fi2_path = os.path.join(BASE, "plots", "fi_safety.png")
    if os.path.exists(fi1_path):
        fi1.image(fi1_path, caption="Quality Grade — Top Features", use_container_width=True)
    if os.path.exists(fi2_path):
        fi2.image(fi2_path, caption="Safety Status — Top Features", use_container_width=True)

    # Confusion matrices
    st.markdown('<div class="section-header">🔢 Confusion Matrices</div>', unsafe_allow_html=True)
    cm1, cm2 = st.columns(2)
    cm1_path = os.path.join(BASE, "plots", "cm_quality.png")
    cm2_path = os.path.join(BASE, "plots", "cm_safety.png")
    if os.path.exists(cm1_path):
        cm1.image(cm1_path, use_container_width=True)
    if os.path.exists(cm2_path):
        cm2.image(cm2_path, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📈 Data Explorer":
    st.markdown("""
    <div class="top-banner">
        <h1>📈 Dataset Explorer</h1>
        <p>Explore the 5,000-sample food safety training dataset — distributions, correlations, filters & statistics</p>
    </div>
    """, unsafe_allow_html=True)

    if not LOADED:
        st.error("Dataset missing.")
        st.stop()

    # Filters
    st.markdown('<div class="section-header">🔎 Filters</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    sel_cat    = f1.multiselect("Food Category", df_full["food_category"].unique().tolist(),
                                 default=df_full["food_category"].unique().tolist())
    sel_grade  = f2.multiselect("Quality Grade", ["A","B","C","D"], default=["A","B","C","D"])
    sel_safety = f3.multiselect("Safety Status", ["Safe","Monitor","Unsafe"],
                                 default=["Safe","Monitor","Unsafe"])
    df_filt = df_full[
        df_full["food_category"].isin(sel_cat) &
        df_full["quality_grade"].isin(sel_grade) &
        df_full["safety_status"].isin(sel_safety)
    ]
    st.caption(f"Showing **{len(df_filt):,}** of {len(df_full):,} records")

    # Summary stats
    st.markdown('<div class="section-header">📋 Summary Statistics</div>', unsafe_allow_html=True)
    num_cols = ["temperature","humidity","storage_days","pH","moisture_percent",
                "defect_count","risk_score","quality_score","shelf_life_days","microbial_load"]
    st.dataframe(df_filt[num_cols].describe().round(2), use_container_width=True)

    # Scatter
    st.markdown('<div class="section-header">🔵 Interactive Scatter Plot</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    x_ax  = sc1.selectbox("X-Axis", num_cols, index=2)
    y_ax  = sc2.selectbox("Y-Axis", num_cols, index=8)
    color = sc3.selectbox("Color by", ["quality_grade","safety_status","food_category","storage_type"])
    cmap  = {"A":"#16a34a","B":"#2563eb","C":"#f59e0b","D":"#dc2626",
             "Safe":"#16a34a","Monitor":"#f59e0b","Unsafe":"#dc2626"}
    samp = df_filt.sample(min(1000, len(df_filt)), random_state=42)
    fig_sc = px.scatter(samp, x=x_ax, y=y_ax, color=color,
                        color_discrete_map=cmap,
                        template="plotly_white", opacity=0.65)
    fig_sc.update_layout(height=380, margin=dict(t=10,b=10))
    st.plotly_chart(fig_sc, use_container_width=True)

    # Histogram
    st.markdown('<div class="section-header">📊 Distribution</div>', unsafe_allow_html=True)
    hc1, hc2 = st.columns(2)
    hist_col = hc1.selectbox("Feature", num_cols, index=6)
    fig_hist = px.histogram(df_filt, x=hist_col, color="quality_grade",
                             color_discrete_map=cmap, nbins=40,
                             template="plotly_white", barmode="overlay", opacity=0.7)
    fig_hist.update_layout(height=320, margin=dict(t=10,b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation heatmap
    with hc2:
        st.markdown("**Correlation Matrix (top numeric features)**")
        corr = df_filt[num_cols].corr().round(2)
        fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, template="plotly_white",
                              aspect="auto")
        fig_heat.update_layout(height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    # Raw data
    st.markdown('<div class="section-header">🗄️ Raw Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(df_filt.sample(min(100, len(df_filt)), random_state=1).reset_index(drop=True),
                 use_container_width=True, height=320)
    csv = df_filt.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered CSV", csv, "food_safety_filtered.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "ℹ️ About":
    st.markdown("""
    <div class="top-banner">
        <h1>ℹ️ About This System</h1>
        <p>AI Food Safety & Quality Intelligence System — architecture, features, and dataset overview</p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<div class="section-header">🧠 System Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>ML Framework:</b> XGBoost (eXtreme Gradient Boosting)<br>
        <b>Task 1:</b> Multi-class classification — Quality Grade (A/B/C/D)<br>
        <b>Task 2:</b> Multi-class classification — Safety Status (Safe/Monitor/Unsafe)<br>
        <b>Dataset:</b> 5,000 synthetic food samples<br>
        <b>Features:</b> 32 input features (numeric + categorical)<br>
        <b>Validation:</b> 80/20 train-test split + 5-fold cross-validation<br>
        <b>Outputs:</b> Grade, Safety Status, Risk Score (0–10), Quality Score (0–10), Shelf Life (days), Prediction Reason
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">📦 Dataset Features</div>', unsafe_allow_html=True)
        features_info = {
            "temperature":       "Storage temperature (°C)",
            "humidity":          "Relative humidity (%)",
            "storage_days":      "Days in storage",
            "pH":                "pH level of the food",
            "moisture_percent":  "Moisture content (%)",
            "defect_count":      "Number of visible defects",
            "packaging_damage":  "Level of packaging damage (0–4)",
            "odor_score":        "Odor quality score (0–10)",
            "color_uniformity":  "Color uniformity index (0–1)",
            "texture_score":     "Texture quality score (0–10)",
            "microbial_load":    "Microbial count (CFU/g)",
            "water_activity":    "Water activity (Aw)",
            "fat_content":       "Fat content (%)",
            "protein_content":   "Protein content (%)",
            "sugar_content":     "Sugar content (%)",
            "salt_content":      "Salt / sodium content (%)",
            "atp_reading":       "ATP bioluminescence (RLU)",
            "ethylene_ppm":      "Ethylene gas concentration (ppm)",
            "co2_percent":       "CO₂ concentration (%)",
            "cold_chain_break":  "Number of cold chain interruptions",
            "transport_hours":   "Hours spent in transport",
            "freshness_index":   "Derived freshness score (0–10)",
            "risk_score":        "Derived overall risk score (0–10)",
            "quality_score":     "Derived quality score (0–10)",
            "shelf_life_days":   "Estimated remaining shelf life (days)",
        }
        feat_df = pd.DataFrame(list(features_info.items()), columns=["Feature", "Description"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=420)

    with a2:
        st.markdown('<div class="section-header">⚙️ XGBoost Configuration</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>n_estimators:</b> 600<br>
        <b>max_depth:</b> 7<br>
        <b>learning_rate:</b> 0.05<br>
        <b>subsample:</b> 0.85<br>
        <b>colsample_bytree:</b> 0.85<br>
        <b>reg_alpha:</b> 0.1 (L1 regularization)<br>
        <b>reg_lambda:</b> 1.5 (L2 regularization)<br>
        <b>min_child_weight:</b> 3<br>
        <b>gamma:</b> 0.1<br>
        <b>objective:</b> multi:softmax<br>
        <b>eval_metric:</b> mlogloss
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">🎯 Prediction Outputs</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Quality Grade:</b><br>
        &nbsp;&nbsp;• <b style="color:#16a34a">A</b> — Excellent (Quality Score ≥ 8.0)<br>
        &nbsp;&nbsp;• <b style="color:#2563eb">B</b> — Good (Score 6.0–7.9)<br>
        &nbsp;&nbsp;• <b style="color:#f59e0b">C</b> — Acceptable (Score 4.0–5.9)<br>
        &nbsp;&nbsp;• <b style="color:#dc2626">D</b> — Poor (Score < 4.0)<br><br>
        <b>Safety Status:</b><br>
        &nbsp;&nbsp;• <b style="color:#16a34a">Safe</b> — Approved for distribution<br>
        &nbsp;&nbsp;• <b style="color:#f59e0b">Monitor</b> — Requires closer inspection<br>
        &nbsp;&nbsp;• <b style="color:#dc2626">Unsafe</b> — Quarantine immediately
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">🏗️ Tech Stack</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        🐍 <b>Python 3.13</b><br>
        🤖 <b>XGBoost 3.x</b> — Gradient Boosting ML<br>
        🔬 <b>Scikit-learn</b> — Preprocessing & Evaluation<br>
        📊 <b>Streamlit</b> — Interactive Web UI<br>
        📈 <b>Plotly</b> — Interactive Charts<br>
        🐼 <b>Pandas / NumPy</b> — Data Engineering<br>
        💾 <b>Joblib</b> — Model Serialization
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin-top:40px;border-color:#e2e8f0">
<div style="text-align:center;color:#94a3b8;font-size:0.8rem;padding:10px 0">
    🧬 AI Food Safety & Quality Intelligence System &nbsp;|&nbsp; Powered by XGBoost + Streamlit
</div>
""", unsafe_allow_html=True)
