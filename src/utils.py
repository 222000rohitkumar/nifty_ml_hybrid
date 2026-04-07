def generate_investment_thought(meta_prob, xgb_prob, lstm_prob, regime_200, vol_quartile):
    """
    Synthesizes model probabilities into an actionable trading thesis.
    """
    # 1. Determine Market State
    trend = "Bullish" if regime_200 == 1 else "Bearish"
    vol_state = ["Low", "Moderate", "High", "Extreme"][int(vol_quartile)]
    
    # 2. Determine Action
    if meta_prob > 0.65:
        action = "STRONG LONG"
        confidence = "High"
    elif meta_prob > 0.50:
        action = "LONG"
        confidence = "Moderate"
    elif meta_prob < 0.35:
        action = "STRONG SHORT / CASH"
        confidence = "High"
    else:
        action = "SHORT / CASH"
        confidence = "Moderate"

    # 3. Generate The Narrative
    thought = f"""
    **Market Context:** The NIFTY is currently in a {trend} trend with {vol_state} volatility.
    
    **Model Consensus:** Our XGBoost Microstructure engine gives a {xgb_prob*100:.1f}% probability of an upward move. 
    Our Sequence-based LSTM gives a {lstm_prob*100:.1f}% probability. 
    
    **Meta-Learner Synthesis:** Factoring in the current regime, the Hybrid Meta-Learner assigns a final **{meta_prob*100:.1f}%** probability of a positive return tomorrow. 
    
    **Recommended Action:** **{action}** ({confidence} Confidence). 
    """
    
    # 4. Add Risk Warnings
    if vol_quartile >= 2 and action == "LONG":
        thought += "\n*Risk Warning: Initiating long positions in high-volatility regimes carries elevated drawdown risk. Reduce position size.*"
        
    return thought