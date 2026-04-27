import torch
import torch.nn as nn

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(AttentionLSTM, self).__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention Layer
        self.attention_linear = nn.Linear(hidden_dim * 2, 1)
        
        # Fully Connected Head
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        
        # Apply Attention
        attention_weights = self.attention_linear(lstm_out) 
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Context Vector
        context_vector = torch.sum(attention_weights * lstm_out, dim=1) 
        
        # Pass through Fully Connected layers
        out = self.fc1(context_vector)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) 
        
        return out
