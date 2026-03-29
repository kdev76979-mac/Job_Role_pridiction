import streamlit as st
import pandas as pd
import joblib
import time
import os

# Set page config
st.set_page_config(
    page_title="Career Domain Predictor | AI",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Glassmorphism, Dark Mode, and premium feel
st.markdown("""
<style>
    /* Main Dark Theme and Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit components */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 3rem 0;
        animation: fadeIn 1.5s ease-in-out;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Glassmorphism Card Container for inputs */
    /* Note: Streamlit containers don't naturally wrap like simple HTML divs do in Markdown,
       so we use visual styling on the app background and specific elements to simulate it,
       along with markdown injections. */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.6);
    }

    /* Glowing Result Card */
    .glowing-card {
        background: rgba(17, 24, 39, 0.8);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-top: 2rem;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.3), inset 0 0 20px rgba(192, 132, 252, 0.2);
        border: 1px solid rgba(192, 132, 252, 0.3);
        animation: glow 3s infinite alternate;
    }

    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(96, 165, 250, 0.3), inset 0 0 20px rgba(192, 132, 252, 0.2); }
        to { box-shadow: 0 0 40px rgba(96, 165, 250, 0.6), inset 0 0 40px rgba(192, 132, 252, 0.4); }
    }
    
    .result-title {
        color: #94a3b8;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }
    
    .result-role {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #4ade80, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    /* Target specific streamlit elements for custom styling */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        font-weight: 600;
        font-size: 1.2rem;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #4f46e5 0%, #a855f7 100%);
        color: white;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Customizing Toggle/Radio Options */
    div.row-widget.stRadio > div {
        flex-direction: row;
        gap: 1rem;
    }
    
    div.row-widget.stRadio > div > label {
        background: rgba(255,255,255,0.05);
        padding: 10px 20px;
        border-radius: 30px;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid transparent;
    }
    
    div.row-widget.stRadio > div > label:hover {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
    }

    /* Styling Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def load_model():
    model_path = 'Job_role.pkl'
    try:
        if not os.path.exists(model_path):
            st.toast("Model file 'Job_role.pkl' not found. Please ensure it is in the same directory.", icon="❌")
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.toast(f"Error loading model: {str(e)}", icon="⚠️")
        return None

# --- Application Header ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Discover Your Path</div>
    <div class="hero-subtitle">Leverage our AI-driven engine to predict your optimal career domain in the tech landscape based on your current skill set.</div>
</div>
""", unsafe_allow_html=True)

st.write("---")

st.subheader("🎓 Academic Profile")
cgpa = st.slider("CGPA (Cumulative Grade Point Average)", min_value=0.0, max_value=10.0, value=7.5, step=0.1, format="%.1f")

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🛠️ Technical Competency")
st.write("Indicate your proficiency in the following areas:")

col1, col2 = st.columns(2)

with col1:
    python_skill = st.radio("Python Programming", options=["No", "Yes"], index=1, horizontal=True)
    sql_skill = st.radio("SQL / Databases", options=["No", "Yes"], index=0, horizontal=True)
    ml_skill = st.radio("Machine Learning", options=["No", "Yes"], index=0, horizontal=True)
    web_skill = st.radio("Web Development", options=["No", "Yes"], index=0, horizontal=True)
    
with col2:
    cloud_skill = st.radio("Cloud Computing (AWS/GCP/Azure)", options=["No", "Yes"], index=0, horizontal=True)
    kali_linux_skill = st.radio("Kali Linux / Cybersecurity", options=["No", "Yes"], index=0, horizontal=True)
    communication_skill = st.radio("Communication Skills", options=["No", "Yes"], index=1, horizontal=True)

# Conversion to binary logic
skill_map = {"No": 0, "Yes": 1}

# Predict button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Analyze Profile & Predict Role ✨", use_container_width=True):
    model = load_model()
    
    if model is None:
        st.error("Prediction cannot proceed. Missing trained model `Job_role.pkl`.")
    else:
        # Simulated Processing State
        with st.status("Initializing AI analysis...", expanded=True) as status:
            st.write("Extracting feature parameters...")
            time.sleep(0.6)
            st.write("Applying inference via Machine Learning model...")
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.015)
                progress_bar.progress(percent_complete + 1)
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
        try:
            # Prepare Data mapping to expected features: ['cgpa', 'python', 'sql', 'ml', 'web', 'cloud', 'kali_linux', 'communication']
            input_data = pd.DataFrame([{
                'cgpa': cgpa,
                'python': skill_map[python_skill],
                'sql': skill_map[sql_skill],
                'ml': skill_map[ml_skill],
                'web': skill_map[web_skill],
                'cloud': skill_map[cloud_skill],
                'kali_linux': skill_map[kali_linux_skill],
                'communication': skill_map[communication_skill]
            }])
            
            # Prediction
            prediction = model.predict(input_data)[0]
            
            # Success Animation
            st.balloons()
            
            # Display Glowing Card
            st.markdown(f"""
            <div class="glowing-card">
                <div class="result-title">Optimal Career Domain</div>
                <div class="result-role">{prediction}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Resources Section based on prediction
            st.subheader(f"📚 Recommended Next Steps for: {prediction}")
            with st.expander(f"View Curated Resources for {prediction}", expanded=True):
                st.markdown(f"""
                ### Accelerate your journey into **{prediction}**:
                
                * **Optimize Your Portfolio**: Ensure your GitHub and portfolio showcase high-impact projects matching **{prediction}** requirements.
                * **Certifications**: Consider obtaining domain-specific certifications to validate your authority in this space.
                * **Networking Strategy**: Engage with specialized communities and connect with professionals working as {prediction} on LinkedIn.
                * **Targeted Upskilling**: Focus your continued learning on advanced frameworks commonly utilized by tier-1 tech companies for this role.
                """)
                
                if skill_map[communication_skill] == 0:
                    st.info("💡 **Pro Tip**: Excellent communication is highly valued for this role. Consider investing time in technical writing, presentation skills, and collaborative soft-skills.")
                    
        except Exception as e:
            st.toast(f"An error occurred during prediction: {str(e)}", icon="🚨")
            st.error("Prediction failed. Please ensure your input array matches the exact schema on which the model was trained.")
