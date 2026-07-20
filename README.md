<div align="center">

# 🏦 NIFTY 50 Institutional Quant Desk

**Regime-Conditioned Meta-Learner Architecture · Multi-Modal Sentiment Integration**

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-1798c1?style=flat-square)](https://xgboost.readthedocs.io/)
[![HuggingFace](https://img.shields.io/badge/Hugging_Face-FinBERT-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/ProsusAI/finbert)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](#license)

*An institutional-grade quantitative research dashboard for forecasting the Indian NIFTY 50 Index — fusing tabular microstructure models, sequential deep learning, and NLP-driven sentiment analysis into a single decision-support engine.*

</div>

---

## 📖 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Disclaimer](#-disclaimer)

---

## 🧠 System Architecture

This engine does not rely on a single model. It orchestrates an ensemble of specialized networks that feed into a final meta-learner to synthesize one cohesive trading thesis — much like a portfolio manager weighing input from several analysts.

| Stage | Component | Input | Role |
|---|---|---|---|
| **1** | **Tabular Edge** — `XGBoost` | 2D market microstructure data (moving averages, momentum oscillators, RSI, MACD) | Captures non-linear, discrete interactions in historical tabular data to produce a baseline directional probability |
| **2** | **Sequential Edge** — `Attention LSTM (PyTorch)` | 3D time-series tensors (lookback sequences) | A custom LSTM with an attention mechanism that weights the most critical days in recent market history |
| **3** | **Meta-Learner** — `Regime-Conditioned Synthesis` | XGBoost confidence + LSTM confidence + macro regime (bull/bear) + volatility quartile | The "Portfolio Manager" — learns *when* to trust XGBoost versus the LSTM depending on the current market environment (e.g. favoring the LSTM in high volatility, XGBoost in stable bull runs) |
| **4** | **FinBERT AI Overlay** — `Hugging Face` | Real-time, web-scraped financial news headlines | Runs `ProsusAI/finbert` sentiment analysis as a fundamental cross-check against the quantitative technical signal |

---

## ✨ Key Features

- **🎨 Dynamic "Live" UI** — A dark-mode Streamlit dashboard with CSS that reacts to the market in real time. Borders, text, and metrics automatically shift to **Emerald Green** (bullish), **Rose Red** (bearish), or **Amber** (sideways) based on the meta-learner's live output.
- **🔄 One-Click Market Sync** — A built-in `daily_updater.py` pipeline fetches missing historical data from Yahoo Finance, computes 47 technical features on the fly, and appends them to the local database.
- **📝 Institutional "Thought Process" Generation** — Automatically produces a human-readable, CIO-style rationale explaining *why* the models reached their consensus.

---

## 📂 Project Structure

```text
nifty_ml_hybrid/
│
├── app.py                    # Main Streamlit application and UI routing
├── daily_updater.py          # YFinance live data fetcher and feature engineer
├── requirements.txt          # Deployment dependencies (CPU-optimized)
│
├── src/                       # Core logic modules
│   ├── data_loader.py         # Tensor formatting and NaN handling
│   ├── finbert_rag.py         # Hugging Face NLP integration and web scraper
│   ├── models_arch.py         # PyTorch AttentionLSTM class definitions
│   └── utils.py                # Rationale and text generation logic
│
├── datasets/
│   └── processed/
│       └── nifty_engineered_features.csv   # CSV feature database
│
└── saved_models/               # Pre-trained model weights
    ├── lstm_weights.pth
    ├── meta_model.pkl
    ├── scaler.pkl
    └── xgb_model.json
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/nifty_ml_hybrid.git
cd nifty_ml_hybrid

# 2. Install dependencies
pip install -r requirements.txt

# 3. Sync the latest market data
python daily_updater.py

# 4. Launch the dashboard
streamlit run app.py
```

---

## ⚠️ Disclaimer

This project is provided for **research and educational purposes only**. It does not constitute financial advice, and nothing in this repository should be construed as a recommendation to buy, sell, or hold any security. Past performance of any model is not indicative of future results. Trade at your own risk.

---

<div align="center">

*Built with PyTorch, XGBoost, FinBERT, and Streamlit.*

</div>
