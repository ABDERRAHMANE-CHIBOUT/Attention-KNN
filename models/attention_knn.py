import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionKNN(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dim=64):
        super(AttentionKNN, self).__init__()

        self.num_classes = num_classes

        # Attention network
        self.attn = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query, neighbors, neighbor_labels):
        """
        query: (batch_size, dim)
        neighbors: (batch_size, K, dim)
        neighbor_labels: (batch_size, K)
        """

        batch_size, K, dim = neighbors.shape

        # Expand query to match neighbors
        query_expanded = query.unsqueeze(1).repeat(1, K, 1)

        # Feature interaction
        diff = torch.abs(query_expanded - neighbors)

        # Concatenate [query, neighbor, |difference|]
        attn_input = torch.cat([query_expanded, neighbors, diff], dim=-1)

        # Compute attention scores
        scores = self.attn(attn_input).squeeze(-1)  # (batch_size, K)

        # Normalize with softmax
        weights = F.softmax(scores, dim=1)  # (batch_size, K)

        # One-hot encode labels — use fixed num_classes
        one_hot = F.one_hot(neighbor_labels, num_classes=self.num_classes).float()

        # Weighted sum of neighbor labels
        weighted_votes = weights.unsqueeze(-1) * one_hot
        output = weighted_votes.sum(dim=1)  # (batch_size, num_classes)

        return output, weights
