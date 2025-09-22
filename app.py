import re
import subprocess
import streamlit as st
import joblib
import numpy as np
import time
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="AI-Counselor CGPA Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with animations and red/blue theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global text color enforcement */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Make input labels and help text white against dark background */
    .stNumberInput label, .stSlider label {
        color: #ffffff !important;
    }

    .stNumberInput .stNumberInput-label, .stSlider .stSlider-label {
        color: #ffffff !important;
    }

    /* Streamlit input labels */
    label {
        color: #ffffff !important;
    }

    /* Help text should also be white */
    .stHelp, .stTooltipIcon {
        color: #ffffff !important;
    }

    /* Ensure content inside white containers is black */
    .main-container, .main-container * {
        color: #000000 !important;
    }

    .metric-card, .metric-card * {
        color: #000000 !important;
    }

    .recommendation-item, .recommendation-item * {
        color: #000000 !important;
    }

    /* But keep the main input area labels white */
    .input-section label {
        color: #ffffff !important;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Background with animated gradient */
    .main > div {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 25%, #dc2626 75%, #b91c1c 100%);
        background-size: 400% 400%;
        animation: gradientShift 8s ease infinite;
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Floating particles */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    }

    .particle {
        position: absolute;
        width: 8px;
        height: 8px;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { 
            transform: translateY(0px) translateX(0px) rotate(0deg); 
            opacity: 0.6; 
        }
        25% { 
            transform: translateY(-20px) translateX(10px) rotate(90deg); 
            opacity: 1; 
        }
        50% { 
            transform: translateY(-10px) translateX(-10px) rotate(180deg); 
            opacity: 0.8; 
        }
        75% { 
            transform: translateY(-25px) translateX(15px) rotate(270deg); 
            opacity: 0.9; 
        }
    }

    /* Main container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 40px;
        margin: 20px;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        z-index: 2;
        animation: slideInUp 1s ease-out;
        color: #000000 !important;
    }

    .main-container * {
        color: #000000 !important;
    }

    @keyframes slideInUp {
        from { 
            transform: translateY(50px); 
            opacity: 0; 
        }
        to { 
            transform: translateY(0); 
            opacity: 1; 
        }
    }

    /* Header styles */
    .header-container {
        text-align: center;
        margin-bottom: 40px;
        animation: fadeInDown 1.2s ease-out;
    }

    @keyframes fadeInDown {
        from { 
            transform: translateY(-30px); 
            opacity: 0; 
        }
        to { 
            transform: translateY(0); 
            opacity: 1; 
        }
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 10px;
        animation: titlePulse 3s ease-in-out infinite;
    }

    @keyframes titlePulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    .subtitle {
        font-size: 1.3rem;
        color: #000000;
        font-weight: 400;
        margin-bottom: 30px;
    }

    /* Input section styling */
    .input-section {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(220, 38, 38, 0.05));
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 2px solid rgba(59, 130, 246, 0.1);
        transition: all 0.3s ease;
        animation: slideInLeft 1s ease-out;
    }

    .input-section label {
        color: #ffffff !important;
    }

    .input-section .stNumberInput label,
    .input-section .stSlider label {
        color: #ffffff !important;
    }

    .input-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.2);
    }

    @keyframes slideInLeft {
        from { 
            transform: translateX(-50px); 
            opacity: 0; 
        }
        to { 
            transform: translateX(0); 
            opacity: 1; 
        }
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Streamlit widget customization */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }

    .stSelectbox > div > div:hover {
        border-color: #3b82f6;
        transform: scale(1.02);
    }

    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        padding: 12px 16px;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        transform: scale(1.02);
    }

    .stSlider > div > div > div {
        background: linear-gradient(90deg, #dc2626, #f59e0b, #10b981);
    }

    .stSlider > div > div > div > div {
        background: #3b82f6;
        width: 24px !important;
        height: 24px !important;
        border-radius: 50%;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }

    .stSlider > div > div > div > div:hover {
        transform: scale(1.2);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #dc2626, #3b82f6);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 16px 32px;
        font-size: 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
    }

    .stButton > button:active {
        transform: translateY(-1px);
    }

    /* Results section */
    .results-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(59, 130, 246, 0.05));
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 2px solid rgba(16, 185, 129, 0.1);
        animation: slideInRight 1s ease-out;
    }

    @keyframes slideInRight {
        from { 
            transform: translateX(50px); 
            opacity: 0; 
        }
        to { 
            transform: translateX(0); 
            opacity: 1; 
        }
    }

    .cgpa-display {
        text-align: center;
        padding: 30px;
        margin: 20px 0;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        animation: cgpaAnimation 2s ease-in-out infinite;
    }

    @keyframes cgpaAnimation {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    .cgpa-value {
        font-size: 4rem;
        font-weight: 700;
        margin: 10px 0;
        animation: numberCount 2s ease-out;
    }

    @keyframes numberCount {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .cgpa-category {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 10px;
    }

    /* Progress bar */
    .progress-container {
        background: #e5e7eb;
        border-radius: 10px;
        height: 20px;
        margin: 15px 0;
        overflow: hidden;
        position: relative;
    }

    .progress-bar {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #dc2626, #f59e0b, #10b981);
        transition: width 2s ease-out;
        position: relative;
        animation: progressFill 2s ease-out;
    }

    @keyframes progressFill {
        from { width: 0% !important; }
    }

    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background: linear-gradient(45deg, 
            rgba(255,255,255,0.3) 25%, 
            transparent 25%, 
            transparent 50%, 
            rgba(255,255,255,0.3) 50%, 
            rgba(255,255,255,0.3) 75%, 
            transparent 75%);
        background-size: 30px 30px;
        animation: progressShine 2s linear infinite;
    }

    @keyframes progressShine {
        0% { background-position: -30px 0; }
        100% { background-position: 30px 0; }
    }

    /* Alert styling */
    .success-alert {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        animation: alertSlideIn 0.5s ease-out;
    }

    .warning-alert {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        animation: alertSlideIn 0.5s ease-out;
    }

    .error-alert {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        animation: alertSlideIn 0.5s ease-out;
    }

    .info-alert {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        animation: alertSlideIn 0.5s ease-out;
    }

    @keyframes alertSlideIn {
        from { 
            transform: translateX(-20px); 
            opacity: 0; 
        }
        to { 
            transform: translateX(0); 
            opacity: 1; 
        }
    }

    /* Recommendations styling */
    .recommendation-item {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px 20px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out;
        color: #000000 !important;
    }

    .recommendation-item * {
        color: #000000 !important;
    }

    .recommendation-item:hover {
        transform: translateX(10px);
        background: rgba(255, 255, 255, 1);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }

    @keyframes fadeInUp {
        from { 
            transform: translateY(20px); 
            opacity: 0; 
        }
        to { 
            transform: translateY(0); 
            opacity: 1; 
        }
    }

    /* Loading spinner */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px;
    }

    .spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(59, 130, 246, 0.3);
        border-top: 4px solid #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 1px solid rgba(59, 130, 246, 0.1);
        transition: all 0.3s ease;
        animation: cardFadeIn 1s ease-out;
        color: #000000 !important;
    }

    .metric-card * {
        color: #000000 !important;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
    }

    @keyframes cardFadeIn {
        from { 
            transform: scale(0.9); 
            opacity: 0; 
        }
        to { 
            transform: scale(1); 
            opacity: 1; 
        }
    }

    /* Floating elements */
    .floating-icon {
        animation: floatingIcon 3s ease-in-out infinite;
    }

    @keyframes floatingIcon {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
</style>

<div class="particles">
    <!-- Particles will be generated by JavaScript -->
    <div class="particle" style="left: 10%; top: 20%; animation-delay: 0s;"></div>
    <div class="particle" style="left: 80%; top: 10%; animation-delay: 1s;"></div>
    <div class="particle" style="left: 60%; top: 70%; animation-delay: 2s;"></div>
    <div class="particle" style="left: 20%; top: 80%; animation-delay: 3s;"></div>
    <div class="particle" style="left: 90%; top: 50%; animation-delay: 4s;"></div>
    <div class="particle" style="left: 40%; top: 30%; animation-delay: 5s;"></div>
</div>

<!-- Additional CSS to fix input labels specifically -->
<style>
    /* Force input labels to be white on the dark background */
    .stApp label {
        color: #ffffff !important;
    }

    .stNumberInput > div > label {
        color: #ffffff !important;
    }

    .stSlider > div > div > div {
        color: #ffffff !important;
    }

    /* Help text visibility */
    .stHelp {
        color: #ffffff !important;
    }

    /* Make sure section headers are white on dark background */
    .section-header {
        color: #ffffff !important;
    }

    .input-section .section-header {
        color: #ffffff !important;
    }

    .results-container .section-header {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


def call_ollama(prompt: str, model: str = "llama3") -> str:
    """
    Runs `ollama run <model>` feeding `prompt` on stdin.
    Requires the Ollama CLI and the model installed locally.
    """
    try:
        cp = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        return cp.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        return f"<Ollama error> {e.stderr.decode('utf-8')}"


@st.cache_data
def load_ridge(path="Ridge.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None


def get_performance_category(cgpa):
    """Categorize CGPA performance on 0-10 scale"""
    if cgpa >= 8.5:
        return "excellent", "#10b981"
    elif cgpa >= 7.0:
        return "good", "#3b82f6"
    elif cgpa >= 6.0:
        return "average", "#f59e0b"
    elif cgpa >= 4.0:
        return "below average", "#f97316"
    else:
        return "poor", "#dc2626"


def create_progress_bar(value, max_value=10):
    """Create an animated progress bar"""
    percentage = (value / max_value) * 100
    return f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {percentage}%"></div>
    </div>
    """


def create_cgpa_display(cgpa, category, color):
    """Create an animated CGPA display"""
    return f"""
    <div class="cgpa-display">
        <div class="floating-icon" style="font-size: 2rem;">🎓</div>
        <div class="cgpa-value" style="color: {color};">{cgpa:.2f}</div>
        <div class="cgpa-category" style="color: {color};">{category.title()} Performance</div>
        {create_progress_bar(cgpa)}
    </div>
    """


# Load the Ridge regression model
ridge_model = load_ridge()

# Header with animations
st.markdown("""
<div class="main-container">
    <div class="header-container">
        <h1 class="main-title">🎓 AI-Counselor CGPA Predictor</h1>
        <p class="subtitle">Advanced Academic Performance Analysis with AI-Powered Insights</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Create main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="input-section">
        <h2 class="section-header">
            <span class="floating-icon">📊</span>
            Student Profile Input
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Input fields with enhanced styling
    study_hours = st.number_input("📚 Study Hours per Day",
                                  min_value=0.0, max_value=12.0,
                                  value=4.0, step=0.5,
                                  help="How many hours do you study daily?")

    sleep_hours = st.number_input("😴 Sleep Hours per Day",
                                  min_value=0.0, max_value=12.0,
                                  value=7.0, step=0.5,
                                  help="How many hours do you sleep daily?")

    attendance = st.slider("🎯 Attendance Percentage",
                           min_value=0, max_value=100,
                           value=75, step=1,
                           help="Your overall class attendance percentage")

    stress = st.slider("😰 Stress Level",
                       min_value=1, max_value=5,
                       value=3, step=1,
                       help="Rate your stress level (1=Low, 5=High)")

    anxiety = st.slider("😨 Anxiety Level",
                        min_value=1, max_value=5,
                        value=3, step=1,
                        help="Rate your anxiety level (1=Low, 5=High)")

    depression = st.slider("😔 Depression Level",
                           min_value=1, max_value=5,
                           value=2, step=1,
                           help="Rate your depression level (1=Low, 5=High)")

    social = st.slider("👥 Social Activity Level",
                       min_value=1, max_value=5,
                       value=3, step=1,
                       help="Rate your social activity level (1=Low, 5=High)")

with col2:
    st.markdown("""
    <div class="results-container">
        <h2 class="section-header">
            <span class="floating-icon">🎯</span>
            Prediction Results
        </h2>
    </div>
    """, unsafe_allow_html=True)

# Prediction button with enhanced styling
if st.button("🔮 Predict My CGPA", type="primary"):
    if ridge_model is None:
        st.markdown('<div class="error-alert">❌ Regression model not loaded.</div>',
                    unsafe_allow_html=True)
        st.stop()

    # Show loading animation
    with st.spinner("🤖 AI is analyzing your profile..."):
        # Simulate processing time for better UX
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        # Input validation with enhanced alerts
        input_issues = []
        if study_hours > 12:
            input_issues.append("⚠️ Study hours > 12 seems unrealistic")
        if sleep_hours > 12 or sleep_hours < 3:
            input_issues.append("⚠️ Sleep hours outside normal range (3-12)")
        if any(x < 1 or x > 5 for x in [stress, anxiety, depression, social]):
            input_issues.append("⚠️ Rating scales should be 1-5")

        if input_issues:
            st.markdown('<div class="warning-alert"><strong>Input Validation Issues:</strong>',
                        unsafe_allow_html=True)
            for issue in input_issues:
                st.markdown(f'<div class="recommendation-item">{issue}</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Make prediction
        features = np.array([[study_hours, sleep_hours, attendance,
                              stress, anxiety, depression, social]])
        raw_baseline = float(ridge_model.predict(features)[0])
        baseline = max(0, min(10, raw_baseline))

        # Get performance category
        category, color = get_performance_category(baseline)

        # Display results with animations
        if raw_baseline != baseline:
            st.markdown(
                f'<div class="warning-alert">⚠️ Model predicted {raw_baseline:.2f}, clamped to valid range: {baseline:.2f}<br>This suggests the model may need retraining with better data.</div>',
                unsafe_allow_html=True)

        # Display CGPA with animation
        st.markdown(create_cgpa_display(baseline, category, "#000000"),
                    unsafe_allow_html=True)

        # AI Counselor Analysis
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #dc2626; margin-bottom: 15px;">🤖 AI Counselor Analysis</h3>
        """, unsafe_allow_html=True)

        # Create AI prompt
        prompt = (
            "You are an academic counselor analyzing student performance on a 10-point CGPA scale where:\n"
            "- 8.5-10.0: Excellent performance\n"
            "- 7.0-8.4: Good performance\n"
            "- 6.0-6.9: Average performance\n"
            "- 4.0-5.9: Below average performance\n"
            "- 0.0-3.9: Poor performance\n\n"
            "A student has these attributes:\n"
            f"- Study Hours per day: {study_hours}\n"
            f"- Sleep Hours per day: {sleep_hours}\n"
            f"- Attendance: {attendance}%\n"
            f"- Stress Level: {stress}/5\n"
            f"- Anxiety Level: {anxiety}/5\n"
            f"- Depression Level: {depression}/5\n"
            f"- Social Activity: {social}/5\n\n"
            f"The regression model predicted a CGPA of {baseline:.2f} out of 10.\n"
            "Based on the student's profile, provide a refined CGPA prediction (0-10 scale) that considers:\n"
            "1. How stress/anxiety/depression might affect academic performance\n"
            "2. Whether study hours and sleep balance is optimal\n"
            "3. Impact of attendance on grades\n"
            "4. Social activity balance\n\n"
            "Respond in exactly this format:\n"
            "<refined_cgpa_number>. <brief explanation considering the factors above and correct performance category>\n"
            "Example: 6.2. Average performance expected due to balanced study habits but high stress levels may limit peak achievement."
        )

        # Get AI response
        with st.spinner("🧠 AI Counselor is thinking..."):
            raw = call_ollama(prompt, model="llama3").strip()

        # Parse AI response
        float_matches = re.findall(r"[0-9]+(?:\.[0-9]+)?", raw)
        if float_matches:
            num_str = float_matches[0]
            ai_cgpa = float(num_str)
            idx = raw.find(num_str) + len(num_str)
            explanation = raw[idx:].lstrip(" *:.-\n ").strip()

            ai_category, ai_color = get_performance_category(ai_cgpa)

            # Display AI prediction
            st.markdown(create_cgpa_display(ai_cgpa, ai_category, "#000000"),
                        unsafe_allow_html=True)

            if explanation:
                st.markdown(f'<div class="info-alert"><strong>AI Analysis:</strong> {explanation}</div>',
                            unsafe_allow_html=True)

            # Compare predictions
            cgpa_diff = ai_cgpa - baseline
            if abs(cgpa_diff) > 0.1:
                if cgpa_diff > 0:
                    st.markdown(
                        f'<div class="success-alert">📈 AI predicts {cgpa_diff:.2f} points higher than baseline, suggesting positive factors in your profile</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="warning-alert">📉 AI predicts {abs(cgpa_diff):.2f} points lower than baseline, indicating potential challenges</div>',
                        unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-alert">✅ AI prediction aligns closely with baseline model</div>',
                            unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Performance Insights
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #000000; margin-bottom: 15px;">📈 Performance Insights</h3>
        """, unsafe_allow_html=True)

        baseline_category, baseline_color = get_performance_category(baseline)
        st.markdown(
            f'<strong>Baseline Performance Category:</strong> <span style="color: #000000;">{baseline_category.title()}</span>',
            unsafe_allow_html=True)

        if float_matches and explanation:
            st.markdown(
                f'<strong>AI-Refined Performance Category:</strong> <span style="color: #000000;">{ai_category.title()}</span>',
                unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Recommendations
        recommendations = []
        if study_hours < 2:
            recommendations.append("📚 Increase daily study hours for better academic performance")
        if sleep_hours < 6:
            recommendations.append("😴 Ensure adequate sleep (7-8 hours) for optimal cognitive function")
        elif sleep_hours > 10:
            recommendations.append("⏰ Consider balancing sleep with more productive activities")
        if attendance < 80:
            recommendations.append("🎯 Improve attendance to enhance learning outcomes")
        if stress >= 4 or anxiety >= 4:
            recommendations.append("🧘 Consider stress management techniques or counseling support")
        if depression >= 3:
            recommendations.append("💚 Seek mental health support if experiencing persistent low mood")
        if social < 2:
            recommendations.append("👥 Engage in more social activities for better mental health and networking")
        elif social >= 4:
            recommendations.append("⚖️ Balance social activities with academic focus")

        # Advanced recommendations based on AI prediction
        if float_matches:
            ai_cgpa = float(float_matches[0])
            if ai_cgpa < 5.0 and (stress >= 3 or anxiety >= 3 or depression >= 2):
                recommendations.append(
                    "💚 Mental health appears to be significantly impacting academic performance - prioritize counseling")
            if ai_cgpa < 6.0 and study_hours < 3:
                recommendations.append("📖 Low predicted CGPA suggests need for significantly more study time")
            if ai_cgpa >= 7.0 and baseline < 6.0:
                recommendations.append(
                    "⭐ Your profile shows potential for good performance - maintain current positive habits")

        # Display recommendations with animations
        if recommendations:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #000000; margin-bottom: 20px;">💡 Personalized Recommendations</h3>
            """, unsafe_allow_html=True)

            for i, rec in enumerate(recommendations):
                st.markdown(f'<div class="recommendation-item" style="animation-delay: {i * 0.1}s;">{rec}</div>',
                            unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="success-alert">🌟 Your academic profile looks well-balanced! Keep up the good work.</div>',
                unsafe_allow_html=True)

        # Performance Metrics Dashboard
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #000000; margin-bottom: 20px;">📊 Performance Metrics Dashboard</h3>
        """, unsafe_allow_html=True)

        # Create metrics columns
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            study_impact = min(100, (study_hours / 8) * 100)
            st.metric(
                label="📚 Study Impact",
                value=f"{study_impact:.0f}%",
                delta=f"{study_impact - 75:.0f}%" if study_impact != 75 else None
            )

        with metric_col2:
            sleep_quality = 100 if 7 <= sleep_hours <= 8 else max(0, 100 - abs(sleep_hours - 7.5) * 20)
            st.metric(
                label="😴 Sleep Quality",
                value=f"{sleep_quality:.0f}%",
                delta=f"{sleep_quality - 75:.0f}%" if sleep_quality != 75 else None
            )

        with metric_col3:
            mental_health = max(0, 100 - ((stress + anxiety + depression - 6) * 10))
            st.metric(
                label="🧠 Mental Health",
                value=f"{mental_health:.0f}%",
                delta=f"{mental_health - 75:.0f}%" if mental_health != 75 else None
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Study Schedule Optimizer
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #000000; margin-bottom: 20px;">⏰ Optimized Study Schedule</h3>
        """, unsafe_allow_html=True)

        # Calculate optimal schedule based on inputs
        optimal_study = max(3, min(6, 10 - baseline))  # Inverse relationship with current performance
        optimal_sleep = 7.5  # Optimal sleep hours
        optimal_break_ratio = 0.2  # 20% break time

        effective_study = optimal_study * (1 - optimal_break_ratio)

        schedule_data = {
            "Activity": ["📚 Focused Study", "☕ Breaks", "😴 Sleep", "👥 Social/Other"],
            "Hours": [effective_study, optimal_study * optimal_break_ratio, optimal_sleep,
                      24 - optimal_study - optimal_sleep],
            "Recommendation": [
                "High-intensity focused learning",
                "Short breaks every 45-60 minutes",
                "Consistent sleep schedule",
                "Balanced social activities"
            ]
        }

        for i in range(len(schedule_data["Activity"])):
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>{schedule_data["Activity"][i]}</strong>: {schedule_data["Hours"][i]:.1f} hours/day
                <br><small style="color: #64748b;">{schedule_data["Recommendation"][i]}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Goal Setting Section
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #000000; margin-bottom: 20px;">🎯 Academic Goal Setting</h3>
        """, unsafe_allow_html=True)

        current_cgpa = ai_cgpa if float_matches else baseline
        target_cgpa = min(10, current_cgpa + 1.0)  # Set target 1 point higher

        # Calculate required improvements
        improvements_needed = []

        if current_cgpa < target_cgpa:
            if study_hours < 4:
                improvements_needed.append(f"📈 Increase study hours to {study_hours + 1:.1f} hours/day")
            if attendance < 90:
                improvements_needed.append(f"🎯 Improve attendance to {min(100, attendance + 10)}%")
            if stress > 3:
                improvements_needed.append("🧘 Implement stress reduction techniques")
            if sleep_hours < 7 or sleep_hours > 8.5:
                improvements_needed.append("😴 Optimize sleep schedule (7-8 hours)")

        st.markdown(f"""
        <div class="info-alert">
            <strong style="color: #000000 !important;">Current CGPA:</strong> <span style="color: #000000 !important;">{current_cgpa:.2f}</span><br>
            <strong style="color: #000000 !important;">Target CGPA:</strong> <span style="color: #000000 !important;">{target_cgpa:.2f}</span><br>
            <strong style="color: #000000 !important;">Improvement Needed:</strong> <span style="color: #000000 !important;">+{target_cgpa - current_cgpa:.2f} points</span>
        </div>
        """, unsafe_allow_html=True)

        if improvements_needed:
            st.markdown("<strong style='color: #000000 !important;'>Action Items to Reach Your Goal:</strong>",
                        unsafe_allow_html=True)
            for improvement in improvements_needed:
                st.markdown(f'<div class="recommendation-item" style="color: #000000 !important;">{improvement}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="success-alert">🌟 You\'re on track! Maintain your current habits to sustain excellent performance.</div>',
                unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Progress Tracking Section
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #000000; margin-bottom: 20px;">📈 Progress Tracking</h3>
        """, unsafe_allow_html=True)

        # Simulate historical data for demonstration
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        historical_cgpa = [max(0, current_cgpa - 1.5 + i * 0.3) for i in range(6)]

        st.markdown("""
        <div class="info-alert">
            📊 Track your progress monthly and adjust strategies based on performance trends.
            Set up regular check-ins with academic advisors for optimal results.
        </div>
        """, unsafe_allow_html=True)

        # Weekly schedule suggestion
        weekly_schedule = {
            "Monday": "📚 Heavy study day - Focus on difficult subjects",
            "Tuesday": "📝 Review and practice problems",
            "Wednesday": "👥 Group study sessions",
            "Thursday": "📖 Catch up on readings",
            "Friday": "🔄 Week review and preparation",
            "Saturday": "⚖️ Balanced study and social activities",
            "Sunday": "🧘 Light review and relaxation"
        }

        st.markdown("<strong style='color: #000000 !important;'>Suggested Weekly Schedule:</strong>",
                    unsafe_allow_html=True)
        for day, activity in weekly_schedule.items():
            st.markdown(
                f'<div class="recommendation-item" style="color: #000000 !important;"><strong style="color: #000000 !important;">{day}:</strong> <span style="color: #000000 !important;">{activity}</span></div>',
                unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# Footer with additional information
st.markdown("""
<div class="main-container" style="margin-top: 40px;">
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(220, 38, 38, 0.1)); border-radius: 15px;">
        <h4 style="color: #000000; margin-bottom: 15px;">🎓 Academic Success Tips</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 20px;">
            <div class="metric-card">
                <strong style="color: #000000 !important;">🎯 Set SMART Goals</strong><br>
                <small style="color: #000000 !important;">Specific, Measurable, Achievable, Relevant, Time-bound objectives</small>
            </div>
            <div class="metric-card">
                <strong style="color: #000000 !important;">🔄 Regular Review</strong><br>
                <small style="color: #000000 !important;">Weekly assessment of progress and strategy adjustments</small>
            </div>
            <div class="metric-card">
                <strong style="color: #000000 !important;">💡 Active Learning</strong><br>
                <small style="color: #000000 !important;">Engage with material through practice and application</small>
            </div>
            <div class="metric-card">
                <strong style="color: #000000 !important;">🤝 Seek Support</strong><br>
                <small style="color: #000000 !important;">Utilize academic resources, tutors, and peer study groups</small>
            </div>
        </div>
        <p style="margin-top: 20px; color: #000000; font-style: italic;">
            Remember: Academic success is a journey, not a destination. Stay consistent and be patient with yourself! 🌟
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Add some JavaScript for enhanced interactivity
st.markdown("""
<script>
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Add particle movement
    function moveParticles() {
        const particles = document.querySelectorAll('.particle');
        particles.forEach(particle => {
            const currentX = parseFloat(particle.style.left) || 0;
            const currentY = parseFloat(particle.style.top) || 0;

            // Random movement
            const newX = currentX + (Math.random() - 0.5) * 2;
            const newY = currentY + (Math.random() - 0.5) * 2;

            // Keep particles within bounds
            particle.style.left = Math.max(0, Math.min(100, newX)) + '%';
            particle.style.top = Math.max(0, Math.min(100, newY)) + '%';
        });
    }

    // Move particles every 5 seconds
    setInterval(moveParticles, 5000);

    // Add hover effects to buttons
    document.addEventListener('DOMContentLoaded', function() {
        const buttons = document.querySelectorAll('button');
        buttons.forEach(button => {
            button.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-3px) scale(1.02)';
            });
            button.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
            });
        });
    });
</script>
""", unsafe_allow_html=True)