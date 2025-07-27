# src/models/multimodal_model.py

import torch
import torch.nn as nn

class MultimodalPolymerPredictor(nn.Module):
    def __init__(self, lstm_dim, rdkit_dim, scikit_dim, hidden_dim=128, output_dim=5):
        super(MultimodalPolymerPredictor, self).__init__()

        self.lstm_branch = nn.Sequential(
            nn.Linear(lstm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.rdkit_branch = nn.Sequential(
            nn.Linear(rdkit_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.scikit_branch = nn.Sequential(
            nn.Linear(scikit_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, lstm_vec, rdkit_vec, scikit_vec):
        x1 = self.lstm_branch(lstm_vec)
        x2 = self.rdkit_branch(rdkit_vec)
        x3 = self.scikit_branch(scikit_vec)

        x = torch.cat([x1, x2, x3], dim=1)
        return self.fusion_layer(x)
