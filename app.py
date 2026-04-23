import streamlit as st
import numpy as np
import pickle
import torch
import xgboost as xgb
import yfinance as yf
import datetime
import time
import textwrap
import os
from groq import Groq 

# --- IMPORTS FOR RAG/FINBERT ---
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# --- CUSTOM MODULES ---
from src.models_arch import AttentionLSTM
from src.utils import generate_investment_thought
from src.data_loader import get_market_data_for_date
from src.finbert_rag import get_finbert_analysis, get_hybrid_verdict 

# --- 1. PAGE CONFIGURATION & SAFE CSS ---
st.set_page_config(
    page_title="Institutional Quant Desk", 
    layout="wide", 
    page_icon="🏦",
    initial_sidebar_state="expanded" 
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
    
    /* Force Inter font safely WITHOUT breaking Streamlit icons */
    html, body, p, h1, h2, h3, h4, h5, h6, label, .pro-box {
        font-family: 'Inter', sans-serif !important;
    }

    /* =========================================================
       1. CORE BACKGROUND FIXES (Eliminating the White Gaps)
       ========================================================= */
    .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    [data-testid="stBottomBlockContainer"], 
    [data-testid="stBottom"] {
        background-color: #0A0E17 !important;
    }

    /* Remove the red/colorful decoration line at the very top of the header */
    header[data-testid="stHeader"]::before {
        background-color: transparent !important;
    }

    /* =========================================================
       2. SIDEBAR STYLING
       ========================================================= */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1e293b !important;
    }
    
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }

    /* =========================================================
       3. BUTTON STYLING (Making white buttons dark & sleek)
       ========================================================= */
    [data-testid="baseButton-secondary"] {
        background-color: #111827 !important;
        color: #cbd5e1 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    
    [data-testid="baseButton-secondary"]:hover {
        background-color: #1e293b !important;
        border-color: #3b82f6 !important; 
        color: #ffffff !important;
    }

    /* =========================================================
       4. TYPOGRAPHY & METRICS CARDS
       ========================================================= */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff; 
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748b; 
        margin-bottom: 25px;
        font-weight: 500;
    }

    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #1e293b;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
    }

    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* =========================================================
       5. GROQ CHAT VISIBILITY FIXES
       ========================================================= */
    [data-testid="stChatMessageContent"], 
    [data-testid="stChatMessageContent"] p, 
    [data-testid="stChatMessageContent"] li, 
    [data-testid="stChatMessageContent"] strong {
        color: #f8fafc !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background-color: transparent !important;
        padding: 10px;
        margin-bottom: 15px;
    }

    /* =========================================================
       6. CHAT INPUT BOX FIX (White Box, Black Text)
       ========================================================= */
    /* Target the container wrapping the chat box to keep it dark */
    [data-testid="stChatInput"] {
        background-color: #0A0E17 !important; 
    }
    /* Target the actual input box - Make it White */
    [data-testid="stChatInput"] > div {
        background-color: #ffffff !important; 
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }
    /* Target the text you type inside the box - Make it Black */
    [data-testid="stChatInput"] textarea {
        color: #000000 !important; 
        -webkit-text-fill-color: #000000 !important; 
        caret-color: #000000 !important; /* Black blinking cursor */
    }
    /* Target the placeholder text - Make it dark grey */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #475569 !important; 
        -webkit-text-fill-color: #475569 !important;
    }
    /* Target the send button arrow - Make it Black */
    [data-testid="stChatInput"] svg {
        fill: #000000 !important; 
    }

    /* =========================================================
       7. EXPANDER FIX (View Analyzed Headlines)
       ========================================================= */
    [data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }
    [data-testid="stExpander"] summary:hover {
        background-color: #334155 !important;
    }
    [data-testid="stExpander"] summary p {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpanderDetails"] {
        background-color: #0A0E17 !important;
        padding: 15px !important;
    }
    [data-testid="stExpanderDetails"] p, 
    [data-testid="stExpanderDetails"] li, 
    [data-testid="stExpanderDetails"] span {
        color: #cbd5e1 !important; /* Crisp, light slate text */
        line-height: 1.6 !important;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 2. MODEL LOADING ---
@st.cache_resource(show_spinner="Loading Deep Learning Models...")
def load_all_models():
    MODEL_DIR = "saved_models"
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(f"{MODEL_DIR}/xgb_model.json")
    lstm_model = AttentionLSTM(input_dim=47, hidden_dim=64, num_layers=2, output_dim=2)
    lstm_model.load_state_dict(torch.load(f"{MODEL_DIR}/lstm_weights.pth"))
    lstm_model.eval()
    with open(f"{MODEL_DIR}/meta_model.pkl", "rb") as f:
        meta_model = pickle.load(f)
    return xgb_model, lstm_model, meta_model

@st.cache_resource(show_spinner="Waking up FinBERT AI...")
def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# --- 3. LIVE MARKET FETCHER ---
@st.cache_data(ttl=60)
def get_live_nifty_price():
    try:
        nifty = yf.Ticker("^NSEI")
        todays_data = nifty.history(period="1d")
        if not todays_data.empty:
            current_price = todays_data['Close'].iloc[-1]
            open_price = todays_data['Open'].iloc[0]
            change = current_price - open_price
            pct_change = (change / open_price) * 100
            return current_price, change, pct_change
        return None, None, None
    except Exception:
        return None, None, None

# Load all models
try:
    xgb_model, lstm_model, meta_model = load_all_models()
    finbert_pipeline = load_finbert()
except Exception as e:
    st.error(f"Failed to load models. System Error: {e}")
    st.stop()

# --- 4. TOP NAVIGATION BAR ---
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("<div class='main-title'>🏦 NIFTY 50 Quant Desk</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Regime-Conditioned Meta-Learner Architecture | Multi-Modal Sentiment Integration</div>", unsafe_allow_html=True)

with header_col2:
    live_price, change, pct_change = get_live_nifty_price()
    if live_price is not None:
        st.markdown("<div style='color: #64748b; font-size: 0.85rem; font-weight: 600; margin-bottom: -15px;'>🔴 LIVE NSE PRINT</div>", unsafe_allow_html=True)
        st.metric(label="", value=f"₹{live_price:,.2f}", delta=f"{change:,.2f} ({pct_change:.2f}%)")
st.write("") 

# Initialize session state variables
if 'meta_prob' not in st.session_state:
    st.session_state.meta_prob = None
if 'messages' not in st.session_state:
    st.session_state.messages = [] # For Groq Chat history

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("<h3 style='color: #ffffff; text-align: center; margin-top: 0px;'>Control Center</h3>", unsafe_allow_html=True)
    st.divider()
    
    if st.button("🔄 Sync Live Market Data", use_container_width=True):
        with st.spinner("Connecting to NSE / Yahoo Finance..."):
            try:
                from daily_updater import update_dataset
                csv_path = "datasets/processed/nifty_engineered_features.csv"
                update_dataset(csv_path)
                st.toast("✅ Database Successfully Synced!", icon="📈")
            except Exception as e:
                st.error(f"Sync Failed: {e}")

    selected_date = st.date_input("Select Trading Horizon", value=datetime.date.today(), key="unique_target_date")
    st.divider()
    run_forecast = st.button("🚀 Execute Quant Forecast", type="primary", use_container_width=True)
    
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 6. CORE LOGIC EXECUTION ---
if run_forecast:
    my_bar = st.progress(0, text="Parsing Microstructure & Sequence Data...")
    try:
        live_data = get_market_data_for_date(
            csv_path="datasets/processed/nifty_engineered_features.csv", 
            scaler_path="saved_models/scaler.pkl",
            target_date=str(selected_date)
        )
        my_bar.progress(40, text="Executing XGBoost Tabular Engine...")
        xgb_p = xgb_model.predict_proba(live_data['X_2d'])[0][1]
        
        my_bar.progress(70, text="Processing Deep Learning Tensors...")
        with torch.no_grad():
            lstm_p = torch.softmax(lstm_model(live_data['X_3d']), dim=1)[0][1].item()
        
        my_bar.progress(90, text="Synthesizing Meta-Learner Consensus...")
        meta_features = np.array([[xgb_p, lstm_p, live_data['regime_200'], live_data['vol_quartile']]])
        m_prob = meta_model.predict_proba(meta_features)[0][1]
        
        st.session_state.meta_prob = m_prob
        st.session_state.live_data = live_data
        st.session_state.xgb_p = xgb_p
        st.session_state.lstm_p = lstm_p
        
        my_bar.progress(100, text="Forecast Complete.")
        time.sleep(0.5)
        my_bar.empty()

    except Exception as e:
        my_bar.empty()
        st.error(f"Execution Error: {e}")

# --- 7. DASHBOARD DISPLAY & CLEAN COLOR ENGINE ---
if st.session_state.meta_prob is not None:
    m_prob = st.session_state.meta_prob
    live_data = st.session_state.live_data
    
    # Sharp, Professional Accent Colors
    if m_prob >= 0.55:
        accent_hex = "#10b981"  # Sharp Emerald
        direction = "UP 🟢"
    elif m_prob <= 0.45:
        accent_hex = "#ef4444"  # Sharp Red
        direction = "DOWN 🔴"
    else:
        accent_hex = "#f59e0b"  # Sharp Amber
        direction = "SIDEWAYS 🟡"

    # Injecting clean borders
    st.markdown(f"""
    <style>
        header[data-testid="stHeader"] {{ border-top: 3px solid {accent_hex}; }}
        div[data-testid="metric-container"] {{ border-top: 3px solid {accent_hex} !important; }}
        div[data-testid="stMetricValue"] > div {{ color: {accent_hex} !important; }}
        
        /* Clean custom box for the thesis */
        .pro-box {{
            background-color: #111827;
            border: 1px solid #1e293b;
            border-left: 4px solid {accent_hex};
            padding: 20px;
            border-radius: 6px;
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.7;
        }}
        .pro-box code {{
            background: transparent !important;
            color: inherit !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    regime_text = "Bullish 🐂" if live_data['regime_200'] == 1 else "Bearish 🐻"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Meta-Learner Signal", value=direction, delta=f"{m_prob*100:.1f}% Confidence")
    col2.metric(label="XGBoost Edge", value=f"{st.session_state.xgb_p*100:.1f}%", delta="Tabular", delta_color="off")
    col3.metric(label="LSTM Edge", value=f"{st.session_state.lstm_p*100:.1f}%", delta="Sequential", delta_color="off")
    col4.metric(label="Macro Regime", value=regime_text, delta=f"Vol Quartile: {int(live_data['vol_quartile'])}", delta_color="off")
    
    st.write("") 
    st.write("") 
    
    main_col1, main_col2 = st.columns([1, 1], gap="large")
    
    with main_col1:
        st.markdown("<div class='section-header'>🧠 Quantitative Thesis</div>", unsafe_allow_html=True)
        raw_thought = generate_investment_thought(m_prob, st.session_state.xgb_p, st.session_state.lstm_p, live_data['regime_200'], live_data['vol_quartile'])
        clean_thought = textwrap.dedent(raw_thought).strip() 
        st.markdown(f"<div class='pro-box'>\n\n{clean_thought}\n\n</div>", unsafe_allow_html=True)
            
    with main_col2:
        st.markdown("<div class='section-header'>📰 FinBERT AI Overlay</div>", unsafe_allow_html=True)
        if st.button("⚡ Query Global News Sentiment", use_container_width=True):
            with st.spinner("Scraping NLP Intelligence..."):
                sentiment_score, news_summary = get_finbert_analysis("^NSEI")
                verdict, rationale = get_hybrid_verdict(m_prob, sentiment_score)
                
                v_col1, v_col2 = st.columns(2)
                v_col1.metric("FinBERT Score", f"{sentiment_score:.2f}")
                v_col2.metric("Final CIO Action", verdict)
                
                st.markdown(f"<div class='pro-box'><strong>Rationale:</strong> {rationale}</div>", unsafe_allow_html=True)
                with st.expander("View Analyzed Headlines"):
                    st.markdown(news_summary)

    st.divider()

   # --- 8. GROQ CHATBOT INTEGRATION & DYNAMIC FOLLOW-UPS ---
    st.markdown("<div class='section-header'>💬 Ask the Quant AI (Powered by Groq)</div>", unsafe_allow_html=True)
    
    # Try to initialize Groq Client
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    
    if not groq_api_key:
        st.warning("⚠️ Groq API key not found. Please add it to your `.streamlit/secrets.toml` file to enable the chat assistant.")
    else:
        client = Groq(api_key=groq_api_key)

        # Dynamic System Prompt injecting current dashboard metrics + Follow-up Instructions
        system_prompt = f"""You are a highly sophisticated Quantitative Analyst AI designed to explain an institutional trading dashboard. 
        The user has just run a forecast on the NIFTY 50 index. 
        Here is the current dashboard situation:
        - Meta-Learner Confidence (Final Signal): {m_prob*100:.1f}% (Above 55% is Bullish, Below 45% is Bearish)
        - XGBoost Confidence (Tabular Data): {st.session_state.xgb_p*100:.1f}%
        - LSTM Confidence (Sequential Data): {st.session_state.lstm_p*100:.1f}%
        - Macro Regime: {'Bullish (Above 200 SMA)' if live_data['regime_200'] == 1 else 'Bearish (Below 200 SMA)'}
        - Volatility Quartile: {int(live_data['vol_quartile'])} (1 is lowest volatility, 4 is highest volatility)
        
        Answer the user's questions clearly, concisely, and professionally. Explain *why* the models might be outputting these specific numbers based on the regime and volatility.
        
        CRITICAL INSTRUCTION: At the end of EVERY response, you MUST provide 2-3 relevant follow-up questions the user could ask to explore the current data deeper. Format them exactly like this:
        
        **Suggested Follow-Ups:**
        * [Question 1]
        * [Question 2]
        """

        # Display chat history
        for message in st.session_state.messages:
            if message["role"] != "system": 
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # --- DYNAMIC FOLLOW-UP BUTTONS BASED ON MARKET CONDITIONS ---
        # Only show these if the chat is empty
        if len(st.session_state.messages) == 0:
            st.info("👋 The forecast is complete. Click a relevant question below to dive into the current market conditions:")
            
            # Generate condition-specific questions
            regime_str = "Bullish" if live_data['regime_200'] == 1 else "Bearish"
            vol_str = "High Volatility" if live_data['vol_quartile'] >= 3 else "Low Volatility"
            
            col_q1, col_q2, col_q3 = st.columns(3)
            
            # Question 1: Ties into the Meta-Learner Consensus
            if col_q1.button(f"🧠 Why is the signal {direction}?", use_container_width=True):
                st.session_state.triggered_prompt = f"Break down why the Meta-Learner is giving a {direction} signal at {m_prob*100:.1f}%. How much is it weighing the LSTM vs XGBoost right now?"
            
            # Question 2: Ties into the Regime and Volatility
            if col_q2.button(f"📊 Explain the {regime_str} + {vol_str} impact", use_container_width=True):
                st.session_state.triggered_prompt = f"We are currently in a {regime_str} regime with {vol_str} (Quartile {int(live_data['vol_quartile'])}). How does this specific environment historically affect the models?"
                
            # Question 3: Ties into model divergence (if any)
            if col_q3.button("⚖️ Are the two base models agreeing?", use_container_width=True):
                st.session_state.triggered_prompt = f"The XGBoost says {st.session_state.xgb_p*100:.1f}% and the LSTM says {st.session_state.lstm_p*100:.1f}%. Are they confirming each other or diverging, and what does that mean?"

        # Accept user input from either the chat bar OR the dynamic buttons
        prompt = st.chat_input("Ask me to explain the current metrics or the models...")
        
        # Override prompt if a dynamic button was clicked
        if 'triggered_prompt' in st.session_state:
            prompt = st.session_state.triggered_prompt
            del st.session_state.triggered_prompt # Delete so it doesn't loop endlessly

        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build messages list for the API
            api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing market behavior..."):
                    try:
                        chat_completion = client.chat.completions.create(
                            messages=api_messages,
                            model="llama-3.1-8b-instant",
                            temperature=0.4,
                            max_tokens=1024,
                        )
                        response = chat_completion.choices[0].message.content
                        st.markdown(response)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        # Force a rerun to hide the initial starter buttons once chat starts
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error communicating with Groq API: {e}")

else:
    st.info("👈 System Ready. Please select a date and click 'Execute Quant Forecast' in the sidebar to generate the institutional report.")
