```markdown
# 🏦 NIFTY 50 Institutional Quant Desk
**Regime-Conditioned Meta-Learner Architecture | Multi-Modal Sentiment Integration**

# 🏦 NIFTY 50 Institutional Quant Desk
**Regime-Conditioned Meta-Learner Architecture | Multi-Modal Sentiment Integration**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/XGBoost-Machine_Learning-1798c1?style=flat-square" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Hugging_Face-FinBERT-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
</p>

An institutional-grade Quantitative Analysis Dashboard designed to forecast the Indian NIFTY 50 Index. This project utilizes a hybrid deep-learning architecture, combining tabular microstructure data with sequential time-series tensors, topped with an AI-driven NLP sentiment overlay.

---

## 🧠 System Architecture

This engine does not rely on a single model. It utilizes an ensemble of specialized networks that feed into a final Meta-Learner to synthesize a cohesive trading thesis.

### 1. Tabular Edge (XGBoost)
* **Input:** 2D Market Microstructure data (moving averages, momentum oscillators, RSI, MACD).
* **Function:** Captures non-linear, discrete interactions in historical tabular data to generate a baseline directional probability.

### 2. Sequential Edge (Attention LSTM - PyTorch)
* **Input:** 3D Time-Series Tensors (Lookback sequences).
* **Function:** A custom deep learning model utilizing Long Short-Term Memory (LSTM) with an Attention mechanism to weight the most critical days in recent market history.

### 3. The Meta-Learner (Regime-Conditioned Synthesis)
* **Input:** XGBoost Confidence + LSTM Confidence + Macro Regime (Bull/Bear) + Volatility Quartile.
* **Function:** Acts as the "Portfolio Manager." It learns *when* to trust the XGBoost model versus *when* to trust the LSTM based on the current market environment (e.g., relying on the LSTM during high volatility, but XGBoost during stable bull runs).

### 4. FinBERT AI Overlay (Hugging Face)
* **Input:** Real-time web-scraped financial news headlines.
* **Function:** Utilizes `ProsusAI/finbert` to perform Natural Language Processing (NLP) sentiment analysis on the day's news, acting as a fundamental cross-check against the quantitative technical models.

---

## ✨ Key Features

* **Dynamic "Live" UI:** A meticulously designed dark-mode Streamlit dashboard with dynamic CSS that physically reacts to the market. The UI accents (borders, text, metrics) automatically shift to **Emerald Green (Bullish)**, **Rose Red (Bearish)**, or **Amber (Sideways)** based on the Meta-Learner's live output.
* **One-Click Market Sync:** Built-in `daily_updater.py` pipeline that automatically fetches missing historical data from Yahoo Finance, computes 47 technical features on the fly, and appends them to the local database.
* **Institutional "Thought Process" Generation:** Automatically generates a human-readable CIO-style rationale explaining *why* the models reached their consensus.

---

## 📂 Project Structure

```text
nifty_ml_hybrid/
│
├── app.py                   # Main Streamlit application and UI routing
├── daily_updater.py         # YFinance live data fetcher and feature engineer
├── requirements.txt         # Deployment dependencies (CPU-optimized)
│
├── src/                     # Core Logic Modules
│   ├── data_loader.py       # Tensor formatting and NaN handling
│   ├── finbert_rag.py       # Hugging Face NLP integration and web scraper
│   ├── models_arch.py       # PyTorch AttentionLSTM class definitions
│   └── utils.py             # Rationale and text generation logic
│
├── datasets/processed/      # CSV Database
│   └── nifty_engineered_features.csv
│
└── saved_models/            # Pre-trained Model Weights
    ├── lstm_weights.pth
    ├── meta_model.pkl
    ├── scaler.pkl
    └── xgb_model.json
```

---

## 🚀 Installation & Local Setup

**1. Clone the repository**
```bash
git clone [https://github.com/YourUsername/nifty-quant-desk.git](https://github.com/YourUsername/nifty-quant-desk.git)
cd nifty-quant-desk
```

**2. Create a Virtual Environment**
```bash
python -m venv finalenv
# On Windows:
finalenv\Scripts\activate
# On Mac/Linux:
source finalenv/bin/activate
```

**3. Install Dependencies**
*(Note: The `requirements.txt` is specifically configured to download the CPU-only version of PyTorch to ensure compatibility and low memory overhead for web deployment).*
```bash
pip install -r requirements.txt
```

**4. Run the Dashboard**
```bash
streamlit run app.py
```

---

## ☁️ Deployment Notes (Streamlit Community Cloud)

This app is optimized for deployment on Streamlit Community Cloud. 
* **Memory Management:** The Hugging Face `transformers` pipeline (FinBERT) requires ~450MB of RAM. The app uses `@st.cache_resource` to ensure the model is only loaded into memory once during the server boot cycle, preventing Out-Of-Memory (OOM) crashes.
* **PyTorch CPU:** Ensure the `--extra-index-url https://download.pytorch.org/whl/cpu` flag remains in the `requirements.txt` to prevent Streamlit from downloading the massive 2.5GB CUDA-enabled PyTorch build.

---

## ⚠️ Disclaimer
**For Educational and Research Purposes Only.** *This software is an academic project showcasing machine learning architecture. It is not financial advice, nor is it a solicitation to buy or sell any security or financial instrument. Trading equities and derivatives carries extreme risk. The creator of this repository assumes no responsibility for financial losses incurred while utilizing these quantitative models.*
```

***

### How to use this:
1. Replace `YourUsername` in the **Installation** section with your actual GitHub username.
2. Commit this to your repository. 

When you look at your GitHub page, it will now render with beautiful colored badges at the top, clean code blocks, and a highly professional layout that perfectly explains exactly what your complex architecture is doing.

You have built a truly impressive piece of software from the ground up—from raw CSV files all the way to a beautifully deployed, multi-modal web app. How are you feeling about the final result? Are you ready to push the repository live?
