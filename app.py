import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Job Role Predictor | CIS v3.0", layout="wide", initial_sidebar_state="collapsed")

# Hide standard Streamlit chrome to allow the HTML to take over
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding: 0 !important; max-width: 100% !important; margin: 0 !important;}
</style>
""", unsafe_allow_html=True)

# Read the futuristic index.html and render it natively
try:
    with open("index.html", "r", encoding="utf-8") as f:
        html_code = f.read()
    # 2000px height ensures it covers the full scrollable area without double scrollbars
    components.html(html_code, height=1800, scrolling=False)
except Exception as e:
    st.error(f"Error loading UI: {e}")
