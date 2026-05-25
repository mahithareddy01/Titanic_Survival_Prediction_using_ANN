import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
# --- PAGE CONFIG (must be first Streamlit call) ---
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
 
# --- GLOBAL STYLES ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;600&display=swap%27);
 
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}
 
/* Page background */
.stApp {
    background: linear-gradient(160deg, #0a1628 0%, #0d2244 50%, #0a1e3d 100%);
    color: #e8edf5;
}
 
/* Hero title */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    text-align: center;
    color: #f0c060;
    letter-spacing: 0.02em;
    margin-bottom: 0.2rem;
    text-shadow: 0 2px 12px rgba(240,192,96,0.25);
}
.hero-subtitle {
    font-size: 1.1rem;
    text-align: center;
    color: #8ab4d8;
    margin-bottom: 1.5rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
 
/* About card */
.about-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(138,180,216,0.2);
    border-left: 4px solid #f0c060;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 24px;
    color: #b8cfe8;
    line-height: 1.7;
}
.about-card h3 {
    color: #f0c060;
    font-family: 'Playfair Display', serif;
    margin-bottom: 8px;
}
 
/* Section headers */
h2, h3 {
    color: #f0c060 !important;
    font-family: 'Playfair Display', serif !important;
}
 
/* Input widgets */
.stSelectbox label, .stSlider label,
.stNumberInput label, .stRadio label {
    color: #8ab4d8 !important;
    font-weight: 600;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
 
/* Container / form border */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
}
 
/* Predict button */
.stButton > button {
    background: linear-gradient(90deg, #c8922a, #f0c060);
    color: #0a1628 !important;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: none;
    border-radius: 8px;
    padding: 14px 0;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s ease, transform 0.1s ease;
    box-shadow: 0 4px 20px rgba(240,192,96,0.3);
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}
 
/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(138,180,216,0.2);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
[data-testid="stMetricLabel"] {
    color: #8ab4d8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stMetricValue"] {
    color: #f0c060 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem !important;
}
 
/* Result box */
.result-box {
    border-radius: 10px;
    padding: 16px 22px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 700;
    font-family: 'Playfair Display', serif;
    margin: 12px 0;
    letter-spacing: 0.04em;
}
.result-survived {
    background: rgba(40,167,69,0.15);
    border: 2px solid #28a745;
    color: #4cdb7a;
}
.result-not-survived {
    background: rgba(220,53,69,0.15);
    border: 2px solid #dc3545;
    color: #ff6b7a;
}
 
/* Divider */
hr {
    border-color: rgba(138,180,216,0.15) !important;
}
</style>
""", unsafe_allow_html=True)
 
# --- SECTION 1: Hero Header ---
st.markdown("""
<div class='about-card'>
<h3>Titanic Survival Prediction</h3>
<p>Deep Learning · ANN Model · Real-Time Inference</p>
</div>
""", unsafe_allow_html=True)

 
# FIX: use_container_width replaces deprecated use_column_width

# --- SECTION 2: About ---
st.markdown("""
<div class='about-card'>
<h3>About This Application</h3>
<p>This application predicts the survival probability of Titanic passengers using a
    pre-trained Artificial Neural Network (ANN) built with TensorFlow. By adjusting passenger
    attributes below, you can see how the model assesses the likelihood of survival based on
    factors such as passenger class, age, fare, and demographic details.</p>
</div>
""", unsafe_allow_html=True)
 
# --- SECTION 3: Load Model ---
model_path = './titanic_model.h5'
 
@st.cache_resource
def load_model():
    try:
        # .h5 (HDF5) files load normally with load_model() in all Keras versions.
        # compile=False skips deserializing the saved optimizer/metrics,
        # which fixes Keras 3 compatibility errors like 'keras.metrics.mse'
        # not being a KerasSaveable. Inference is unaffected.
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(
            f"Error loading model from '{model_path}'. "
            f"Ensure the file exists in the same directory as app.py. Details: {e}"
        )
        return None
 
model = load_model()
 
if model is None:
    st.stop()
 
# --- SECTION 4: Passenger Input Form ---
st.header("Passenger Information")
 
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        pclass   = st.selectbox("Passenger Class", [1, 2, 3], index=2,
                                help="1st = Upper · 2nd = Middle · 3rd = Lower")
        age      = st.slider("Age", 0, 80, 22)
        sibsp    = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=8,  value=0)
        parch    = st.number_input("Parents / Children Aboard", min_value=0, max_value=6,  value=0)
    with col2:
        fare     = st.number_input("Fare (£)", min_value=0.0, max_value=512.0, value=7.25, step=0.01)
        sex      = st.radio("Sex", ["male", "female"], index=0)
        embarked = st.radio("Port of Embarkation", ["C", "Q", "S"], index=2,
                            help="C = Cherbourg · Q = Queenstown · S = Southampton")
 
# --- SECTION 5: Detect model's expected input size ---
def get_expected_features(model):
    """Read the input dimension from the first layer's weight matrix."""
    try:
        return model.layers[0].get_weights()[0].shape[0]
    except Exception:
        pass
    try:
        return model.input_shape[-1]
    except Exception:
        return 8  # fallback to full feature set
 
N_FEATURES = get_expected_features(model)
 
# Feature sets ordered from smallest to largest, matching common Titanic ANN tutorials.
# The model's actual input size selects the right subset automatically.
FEATURE_SETS = {
    3: ['Pclass', 'Sex_male', 'Age'],
    4: ['Pclass', 'Sex_male', 'Age', 'Fare'],
    5: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
    6: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male'],
    7: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_S'],
    8: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S'],
}
 
def preprocess_input(pclass, age, sibsp, parch, fare, sex, embarked):
    """Build the full 8-column frame then slice to however many features the model needs."""
    full = {
        'Pclass':      pclass,
        'Age':         age,
        'SibSp':       sibsp,
        'Parch':       parch,
        'Fare':        fare,
        'Sex_male':    1 if sex == 'male' else 0,
        'Embarked_Q':  1 if embarked == 'Q' else 0,
        'Embarked_S':  1 if embarked == 'S' else 0,
    }
    cols = FEATURE_SETS.get(N_FEATURES, list(full.keys())[:N_FEATURES])
    import numpy as np
    return np.array([[full[c] for c in cols]], dtype='float32')
 
# --- SECTION 6: Prediction ---
st.header("Get Prediction")
 
if st.button("Predict Survival", type="primary", use_container_width=True):
    processed_input = preprocess_input(pclass, age, sibsp, parch, fare, sex, embarked)
 
    prediction_proba = float(model.predict(processed_input)[0][0])
 
    survival_pct     = prediction_proba * 100
    non_survival_pct = (1 - prediction_proba) * 100
    survived         = prediction_proba > 0.5
    confidence       = abs(prediction_proba - 0.5) * 200  # 0–100 % relative to 0.5
 
    # Result banner
    st.markdown(
        f"<div class='result-box {'result-survived' if survived else 'result-not-survived'}'>"
        f"{'✦ Survived' if survived else '✕ Did Not Survive'}"
        f"</div>",
        unsafe_allow_html=True,
    )
 
    # Metric cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Outcome",              "Survived" if survived else "Not Survived")
    with c2:
        st.metric("Survival Probability", f"{survival_pct:.1f}%")
    with c3:
        st.metric("Confidence Score",     f"{confidence:.1f}%")
 
    # --- Chart ---
    st.markdown("<h4 style='text-align:center; margin-top:20px;'>Probability Breakdown</h4>",
                unsafe_allow_html=True)
 
    data = pd.DataFrame({
        'Category':    ['Survived', 'Not Survived'],
        'Probability': [survival_pct, non_survival_pct],
    })
 
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#0d2244')
    ax.set_facecolor('#0a1628')
 
    bars = ax.bar(
        data['Category'], data['Probability'],
        color=['#28a745', '#dc3545'],
        width=0.45,
        edgecolor='none',
    )
    ax.bar_label(bars, fmt='%.1f%%', padding=4,
                 color='#e8edf5', fontsize=11, fontweight='bold')
 
    ax.set_ylim(0, 115)
    ax.set_ylabel("Probability (%)", color='#8ab4d8', fontsize=10)
    ax.set_title("Survival vs. Non-Survival", color='#f0c060',
                 fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#8ab4d8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
 
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)