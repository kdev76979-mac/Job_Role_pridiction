import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin

# Define ColumnSelector as expected by the pickled pipeline
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        self.column = column
    def __setstate__(self, state):
        self.__dict__.update(state)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.column]

# Must set page_config FIRST
st.set_page_config(page_title="Job Role Prediction", page_icon="🔮", layout="centered")

# Initialize session state for view transition
if 'prediction_state' not in st.session_state:
    st.session_state.prediction_state = False
if 'top_roles' not in st.session_state:
    st.session_state.top_roles = []
if 'top_probs' not in st.session_state:
    st.session_state.top_probs = []

# CSS Styling for High-Contrast Dark Mode Neon & Glassmorphism
st.markdown("""
<style>
/* Base Dark Theme and Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App Background: High Contrast deep space charcoal black */
.stApp {
    background: #020305;
    background-image: 
        radial-gradient(circle at 10% 20%, rgba(0, 255, 255, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(255, 0, 255, 0.08) 0%, transparent 40%);
    background-attachment: fixed;
    color: #FFFFFF;
}

/* Centralized Layout Max Width */
.block-container {
    max-width: 850px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* Frosted glass containers (High visibility) mapped to Streamlit Native Border to avoid empty box bug */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(10, 15, 25, 0.6);
    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);
    border-radius: 20px;
    border: 1px solid rgba(0, 255, 255, 0.3);
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.8), inset 0 0 25px rgba(0, 255, 255, 0.05);
    padding: 3rem !important;
    margin-bottom: 2.5rem !important;
    position: relative;
    z-index: 1;
    transition: transform 0.4s ease, box-shadow 0.4s ease, border-color 0.3s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 20px 60px rgba(0, 255, 255, 0.2), inset 0 0 30px rgba(0, 255, 255, 0.1);
    border-color: rgba(0, 255, 255, 0.7);
}

/* Titles */
h1, h2, h3, h4 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
}

.main-title {
    font-size: 3.5rem;
    background: -webkit-linear-gradient(45deg, #00FFFF, #FF00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
    font-weight: 800;
    text-shadow: 0 0 50px rgba(0, 255, 255, 0.4);
}

.subtitle {
    text-align: center;
    color: #00FFFF;
    font-size: 1.2rem;
    margin-bottom: 3rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
}

/* Input Fields Overrides: Subtle glowing neon borders when active */
div[data-baseweb="input"] > div, 
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {
    background-color: rgba(0, 0, 0, 0.6) !important;
    border: 1px solid rgba(0, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
    transition: all 0.3s ease !important;
}

div[data-baseweb="input"] > div:focus-within, 
div[data-baseweb="textarea"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #00FFFF !important;
    box-shadow: 0 0 25px rgba(0, 255, 255, 0.5) !important;
    background-color: rgba(0, 0, 0, 0.8) !important;
}

/* Custom Button */
.stButton > button {
    background: transparent !important;
    color: #00FFFF !important;
    font-weight: 800 !important;
    font-size: 1.3rem !important;
    border: 2px solid #00FFFF !important;
    border-radius: 12px !important;
    padding: 1.2rem 2rem !important;
    width: 100%;
    margin-top: 1rem;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.3), inset 0 0 15px rgba(0, 255, 255, 0.2) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    backdrop-filter: blur(10px);
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    background: rgba(0, 255, 255, 0.15) !important;
    box-shadow: 0 0 40px rgba(0, 255, 255, 0.9), inset 0 0 30px rgba(0, 255, 255, 0.5) !important;
    text-shadow: 0 0 15px rgba(0, 255, 255, 1);
    color: #FFFFFF !important;
    border-color: #FFFFFF !important;
}

/* Secondary Button / Recalibrate */
.recalibrate-btn > button {
    background: transparent !important;
    border: 2px solid #FF00FF !important;
    color: #FF00FF !important;
    box-shadow: 0 0 20px rgba(255, 0, 255, 0.3) !important;
}
.recalibrate-btn > button:hover {
    background: rgba(255, 0, 255, 0.15) !important;
    border: 2px solid #FFFFFF !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 40px rgba(255, 0, 255, 0.9) !important;
    text-shadow: 0 0 15px rgba(255, 0, 255, 1);
}

/* Match Bars */
.role-container { margin: 2.5rem 0; }
.role-header {
    display: flex; justify-content: space-between; margin-bottom: 1rem; font-weight: 600;
}
.role-name { font-size: 1.6rem; letter-spacing: 1px; color: #FFFFFF;}
.match-pct { color: #FFFFFF; font-weight: 800; font-size: 1.4rem; text-shadow: 0 0 15px rgba(255,255,255,0.7); }

/* Neon Progress Track */
.bar-track {
    width: 100%; background-color: rgba(0, 0, 0, 0.8); border-radius: 20px; height: 18px; overflow: hidden; position: relative;
    border: 1px solid rgba(255,255,255,0.1);
}

@keyframes fillAnim { from { width: 0; } }

.bar-fill-1 {
    height: 100%; background: linear-gradient(90deg, #00FFFF, #FF00FF); box-shadow: 0 0 25px rgba(0, 255, 255, 0.9); border-radius: 20px; animation: fillAnim 1.2s cubic-bezier(0.1, 0.8, 0.2, 1) forwards;
}
.bar-fill-2 {
    height: 100%; background: linear-gradient(90deg, #0088FF, #AA00FF); box-shadow: 0 0 25px rgba(170, 0, 255, 0.9); border-radius: 20px; animation: fillAnim 1.4s cubic-bezier(0.1, 0.8, 0.2, 1) forwards;
}
.bar-fill-3 {
    height: 100%; background: linear-gradient(90deg, #00FFCC, #0088FF); box-shadow: 0 0 25px rgba(0, 255, 204, 0.9); border-radius: 20px; animation: fillAnim 1.6s cubic-bezier(0.1, 0.8, 0.2, 1) forwards;
}

/* Text styles */
label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* Custom Skill Item Row layout */
.skill-row {
    font-size: 1.2rem;
    color: #FFFFFF;
    padding: 10px 0;
    font-weight: 500;
    border-bottom: 1px solid rgba(0, 255, 255, 0.1);
}

</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    try:
        pipeline = joblib.load('career_pipeline_v3.pkl')
        encoder = joblib.load('label_encoder_v3.pkl')
        return pipeline, encoder
    except Exception as e:
        return None, None

pipeline, encoder = load_models()

# Main Header
st.markdown("<h1 class='main-title'>Job Role Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A.I. CAREER INTELLIGENCE PLATFORM</p>", unsafe_allow_html=True)

if not pipeline or not encoder:
    st.error("Model Error: Ensure `career_pipeline_v3.pkl` and `label_encoder_v3.pkl` exist in the root folder.")
    st.stop()


# --- VIEW ROUTING ---
if not st.session_state.prediction_state:
    # --- INPUT SECTION ---
    
    # 1. Academic & Aptitude Profile Container
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:-1rem;'>Academic & Aptitude Profile</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        cgpa = st.slider("Academic CPI / CGPA", min_value=1.0, max_value=10.0, value=7.8, step=0.1)
        aptitude = st.slider("Logical / Aptitude Score", min_value=1, max_value=100, value=85, step=1)
        degree = st.selectbox("Degree / Branch", ["B.Tech CSE", "BCA", "BSc IT", "Data Science", "M.Tech", "MCA", "Other"])

    # 2. Experience & Industry Container
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:-1rem;'>Experience & Industry</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        experience = st.selectbox("Experience Level", ["Fresher", "1-3 Years", "3+ Years"])
        industry = st.selectbox("Target Industry", ["Tech", "Finance", "Healthcare", "E-Commerce", "Other"])

    # 3. IT / Technical Skills Container
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:-1rem;'>IT / Technical Skills</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #00FFFF; margin-bottom: 2rem; font-weight: 500;'>Select your technical proficiencies.</p>", unsafe_allow_html=True)
        
        skill_list = ["Python", "Java", "SQL", "React", "AWS", "Docker", "Machine Learning", "TensorFlow", "C++", "Data Visualization"]
        selected_skills = []
        
        for skill in skill_list:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"<div class='skill-row'>{skill}</div>", unsafe_allow_html=True)
            with c2:
                is_checked = st.toggle("", key=f"tg_{skill}", label_visibility="collapsed")
                if is_checked:
                    selected_skills.append(skill)

    # 4. Soft Skills Profile Container
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:-1rem;'>Soft Skills Profile</h3>", unsafe_allow_html=True)
        soft_skills = st.text_area("Identify key interpersonal strengths", placeholder="e.g., Communication, Leadership, Problem Solving", height=120)

    # 5. Action Button
    if st.button("INITIALIZE PREDICTION SEQUENCE"):
        it_skills_text = " ".join(selected_skills)
        
        if len(selected_skills) == 0 and len(soft_skills.strip()) == 0:
            st.warning("Prediction constraint: Please define at least one skill to initiate the simulation.")
        else:
            with st.spinner("Analyzing parameters and computing trajectory vectors..."):
                time.sleep(1.5)
                
                # Currently only IT Skills and Soft Skills are used by the ML model. 
                # (The other inputs just enrich the high-tech UI form)
                input_df = pd.DataFrame({
                    'IT Skills': [it_skills_text],
                    'Soft Skills': [soft_skills]
                })

                try:
                    probas = pipeline.predict_proba(input_df)[0]
                    top3_indices = np.argsort(probas)[::-1][:3]
                    
                    st.session_state.top_roles = encoder.inverse_transform(top3_indices)
                    st.session_state.top_probs = probas[top3_indices] * 100
                    st.session_state.prediction_state = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Computation Error: {e}")

else:
    # --- RESULTS VIEW ---
    st.markdown('''
    <div class="glass-container" style="text-align: center; padding: 3rem; margin-bottom: 3rem;">
        <h1 style="font-size: 3.5rem; font-weight: 800; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.5rem; background: -webkit-linear-gradient(45deg, #00FFFF, #FF00FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 30px rgba(0, 255, 255, 0.4);">
            Congratulations!
        </h1>
        <p style="color: #00FFFF; font-size: 1.2rem; letter-spacing: 2px; text-shadow: 0 0 10px rgba(0, 255, 255, 0.4); margin-bottom: 0;">
            YOUR OPTIMAL TRAJECTORIES HAVE BEEN COMPUTED
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; margin-bottom: 3rem;'>Top Career Trajectories</h2>", unsafe_allow_html=True)

    bar_classes = ['bar-fill-1', 'bar-fill-2', 'bar-fill-3']
    colors = ['#00FFFF', '#0088FF', '#00FFCC']

    for i in range(3):
        role = st.session_state.top_roles[i]
        prob = st.session_state.top_probs[i]
        bar_class = bar_classes[i]
        color = colors[i]
        
        html_str = f"""
        <div class="role-container">
            <div class="role-header">
                <span class="role-name">{role.upper()}</span>
                <span class="match-pct" style="color: {color}">{prob:.1f}% Match</span>
            </div>
            <div class="bar-track">
                <div class="{bar_class}" style="width: {prob}%;"></div>
            </div>
        </div>
        """
        st.markdown(html_str, unsafe_allow_html=True)
        
    st.markdown('<br><br>', unsafe_allow_html=True)
    
    # Recalibrate Button
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.markdown('<div class="recalibrate-btn">', unsafe_allow_html=True)
        if st.button("RECALIBRATE ROUTE"):
            st.session_state.prediction_state = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
