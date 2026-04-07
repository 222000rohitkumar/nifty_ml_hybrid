# File: src/__init__.py

# This allows app.py to import these functions directly from 'src' 
# instead of typing 'from src.models_arch import AttentionLSTM'

from .models_arch import AttentionLSTM
from .utils import generate_investment_thought

__all__ = ["AttentionLSTM", "generate_investment_thought"]