import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.constants import MODEL_VOCAB_SIZES
from einops import repeat

class ATP_R_MLP(nn.Module):
    def __init__(self, args, actual_sequence_length):
        super(ATP_R_MLP, self).__init__()        
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout # Use higher value if args.dropout is low (< 0.3)
        self.actual_sequence_length = actual_sequence_length
        self.num_layers = args.num_layers
        
        # Learnable parameters for feature weighting
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_delta_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_variance = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_range = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM], self.hidden_dim)

        # --- NEW REGULARIZED BOTTLENECK ---
        # Forces the model to find cross-feature correlations instead of memorizing raw values
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 6, self.hidden_dim // 2), # Compress
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.4), # High dropout here is key to prevent memorization
            nn.Linear(self.hidden_dim // 2, self.hidden_dim), # Expand
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(p=self.dropout)
        )

        self.lin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_dim if i > 0 else self.hidden_dim * self.actual_sequence_length
            out_dim = self.hidden_dim if (i+1) < self.num_layers else 1
            self.lin_layers.append(nn.Linear(in_dim, out_dim))
            if (i+1) < self.num_layers:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        self.sigmoid = nn.Sigmoid()

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        sorted_TDS_normalized = torch.nan_to_num(sorted_TDS_normalized, nan=0.0)
        normalized_ATP = torch.nan_to_num(normalized_ATP, nan=0.0)

        tds_safe = sorted_TDS_normalized.to(torch.float32)
        probs = F.softmax(tds_safe, dim=-1)
        safe_probs = torch.clamp(probs, min=1e-7)
        
        entropy = -(probs * torch.log(safe_probs)).sum(dim=-1, keepdim=True)
        delta_entropy = torch.zeros_like(entropy)
        delta_entropy[:, 1:, :] = entropy[:, 1:, :] - entropy[:, :-1, :]
        
        variance = torch.var(probs, dim=-1, keepdim=True)
        dist_range = torch.max(probs, dim=-1, keepdim=True).values - torch.min(probs, dim=-1, keepdim=True).values
        
        # Scaling features with learnable params
        encoded_entropy = entropy.to(sorted_TDS_normalized.dtype) * self.param_for_entropy
        encoded_delta_entropy = delta_entropy.to(sorted_TDS_normalized.dtype) * self.param_for_delta_entropy
        encoded_variance = variance.to(sorted_TDS_normalized.dtype) * self.param_for_variance
        encoded_range = dist_range.to(sorted_TDS_normalized.dtype) * self.param_for_range

        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = (normalized_ATP * (2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))).unsqueeze(-1) * self.param_for_ATP_R)
        else:
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
                    
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        
        # Fusion
        raw_features = torch.cat([encoded_ATP_R, encoded_normalized_ATP, encoded_entropy, 
                                  encoded_delta_entropy, encoded_variance, encoded_range], dim=-1)
        x = self.feature_fusion(raw_features)
        
        x = x.flatten(start_dim=1)
        for i in range(self.num_layers):
            x = self.lin_layers[i](x)
            if (i+1) < self.num_layers:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)

        return self.sigmoid(torch.nan_to_num(x, nan=0.0)).squeeze(-1)
