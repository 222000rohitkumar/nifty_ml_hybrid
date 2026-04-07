from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import torch
import numpy as np

# Use the industry-standard FinBERT model
FINBERT_MODEL = "ProsusAI/finbert"

def fetch_real_news(query="Nifty 50 Indian Stock Market", limit=5):
    """
    Bulletproof news fetcher using Google News RSS. 
    Bypasses the buggy yfinance library completely.
    """
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    headlines = []
    try:
        # Masking as a standard browser so Google doesn't block the request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            root = ET.fromstring(response.read())
            # Parse the top 5 news titles
            for item in root.findall('.//item')[:limit]:
                headlines.append(item.find('title').text)
    except Exception as e:
        print(f"RSS Fetch Error: {e}")
        pass
    return headlines

def get_finbert_analysis(ticker="^NSEI"): 
    """
    Fetches latest news and performs sentiment analysis using FinBERT.
    Returns a sentiment score and a summary string.
    """
    try:
        # 1. Initialize the Hugging Face pipeline
        tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
        model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        # 2. Fetch News using our bulletproof Google News method
        headlines = fetch_real_news()
        
        if not headlines:
            return 0.0, "Network block: Could not fetch live news. Please check firewall settings."

        # 3. Run Sentiment Analysis
        results = nlp(headlines)
        
        sentiment_values = []
        summary_text = ""
        
        for i, res in enumerate(results):
            label = res['label']
            score = res['score']
            summary_text += f"- {headlines[i]} (**{label.upper()}**)\n"
            
            # Convert labels to numerical scores
            if label == 'positive':
                sentiment_values.append(score)
            elif label == 'negative':
                sentiment_values.append(-score)
            else:
                sentiment_values.append(0)

        avg_sentiment = np.mean(sentiment_values)
        return avg_sentiment, summary_text

    except Exception as e:
        return 0.0, f"FinBERT Analysis Error: {str(e)}"

def get_hybrid_verdict(meta_prob, sentiment_score):
    """
    Combines the Quant Meta-Learner probability with FinBERT Sentiment.
    """
    # Normalize and weight the two signals (70% Quant / 30% Sentiment)
    combined_score = (meta_prob * 0.7) + (((sentiment_score + 1) / 2) * 0.3)
    
    if combined_score > 0.65:
        return "INVEST (Strong Buy)", "Technical models and market news are highly aligned for an upward move."
    elif combined_score > 0.52:
        return "INVEST (Cautious)", "Technicals are positive, but news sentiment is mixed/neutral."
    elif combined_score < 0.40:
        return "DO NOT INVEST", "Both mathematical and sentiment indicators suggest significant downside risk."
    else:
        return "WAIT / HOLD CASH", "Conflicting signals between technical indicators and news sentiment."