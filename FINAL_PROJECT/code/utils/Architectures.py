import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.constants import MODEL_VOCAB_SIZES
from einops import repeat
from vit_pytorch import ViT
from utils.Architectures_utils import *

def get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape):
    model_mapping = {
        # LOS-based
        'LOS-Net': LOS_Net,
        'ATP_R_MLP': ATP_R_MLP,
        'ATP_R_Transf': ATP_R_Transf,
    }
    
    if args.probe_model in {'LOS-Net', 'ATP_R_Transf'}:
        return model_mapping[args.probe_model](args=args, max_sequence_length=max_sequence_length, input_dim=input_dim)
    elif args.probe_model in {'ATP_R_MLP'}:
        return model_mapping[args.probe_model](args=args, actual_sequence_length=actual_sequence_length)
    else:
        raise ValueError(f"Unknown model: {args.probe_model}")
    

######################## LOS ########################
class ATP_R_MLP(nn.Module):

    def __init__(self, args, actual_sequence_length):
        super(ATP_R_MLP, self).__init__()        
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.actual_sequence_length = actual_sequence_length
        
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_margin = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Temporal Dynamics Parameters
        self.param_for_delta_margin = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_delta_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM], self.hidden_dim)
        else:
            raise ValueError("Invalid encoding type.")

        # Non-linear Feature Fusion MLP
        # Concatenating 6 features of size hidden_dim -> 6 * hidden_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 6, self.hidden_dim * 3),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        )
        
        # Linear layers
        self.lin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_dim if i > 0 else self.hidden_dim * self.actual_sequence_length
            out_dim = self.hidden_dim if (i+1) < self.num_layers else 1
            self.lin_layers.append(nn.Linear(in_dim, out_dim))
            if (i+1) < self.num_layers:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        sorted_TDS_normalized = torch.nan_to_num(sorted_TDS_normalized, nan=0.0)
        normalized_ATP = torch.nan_to_num(normalized_ATP, nan=0.0)

        tds_safe = sorted_TDS_normalized.to(torch.float32)
        margin = tds_safe[:, :, 0:1] - normalized_ATP.to(torch.float32)
        
        probs = F.softmax(tds_safe, dim=-1)
        safe_probs = torch.clamp(probs, min=1e-7)
        entropy = -(probs * torch.log(safe_probs)).sum(dim=-1, keepdim=True)
        
        # Temporal Dynamics: Delta Features (Current step - Previous step)
        delta_margin = torch.zeros_like(margin)
        delta_margin[:, 1:, :] = margin[:, 1:, :] - margin[:, :-1, :]
        
        delta_entropy = torch.zeros_like(entropy)
        delta_entropy[:, 1:, :] = entropy[:, 1:, :] - entropy[:, :-1, :]
        
        # Encoding
        encoded_margin = margin.to(sorted_TDS_normalized.dtype) * self.param_for_margin
        encoded_entropy = entropy.to(sorted_TDS_normalized.dtype) * self.param_for_entropy
        encoded_delta_margin = delta_margin.to(sorted_TDS_normalized.dtype) * self.param_for_delta_margin
        encoded_delta_entropy = delta_entropy.to(sorted_TDS_normalized.dtype) * self.param_for_delta_entropy

        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
            
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        
        # Feature Fusion: Concatenate 6 features
        raw_features = torch.cat([encoded_ATP_R, encoded_normalized_ATP, encoded_margin, encoded_entropy, encoded_delta_margin, encoded_delta_entropy], dim=-1)
        x = self.feature_fusion(raw_features)
        
        x = x.flatten(start_dim=1)
        
        for i in range(self.num_layers):
            x = self.lin_layers[i](x)
            if (i+1) < self.num_layers:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)

        x = torch.nan_to_num(x, nan=0.0)
        return self.sigmoid(x).squeeze(-1)


class ATP_R_Transf(nn.Module):
    def __init__(self, args, max_sequence_length, input_dim=1):
        super(ATP_R_Transf, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = args.hidden_dim
        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = args.pool
        
        assert self.pool in {'cls', 'mean', 'attention'}, "Pool type must be 'cls', 'mean', or 'attention'"

        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_margin = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Temporal Dynamics Parameters
        self.param_for_delta_margin = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_delta_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM], self.hidden_dim)
        else:
            raise ValueError("Invalid encoding type.")

        # Non-linear Feature Fusion MLP
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 6, self.hidden_dim * 3),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        if self.pool == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_dim // 2, 1)
            )

        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
    
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        sorted_TDS_normalized = torch.nan_to_num(sorted_TDS_normalized, nan=0.0)
        normalized_ATP = torch.nan_to_num(normalized_ATP, nan=0.0)

        tds_safe = sorted_TDS_normalized.to(torch.float32)
        margin = tds_safe[:, :, 0:1] - normalized_ATP.to(torch.float32)
        
        probs = F.softmax(tds_safe, dim=-1)
        safe_probs = torch.clamp(probs, min=1e-7)
        entropy = -(probs * torch.log(safe_probs)).sum(dim=-1, keepdim=True)
        
        # Temporal Dynamics
        delta_margin = torch.zeros_like(margin)
        delta_margin[:, 1:, :] = margin[:, 1:, :] - margin[:, :-1, :]
        
        delta_entropy = torch.zeros_like(entropy)
        delta_entropy[:, 1:, :] = entropy[:, 1:, :] - entropy[:, :-1, :]
        
        encoded_margin = margin.to(sorted_TDS_normalized.dtype) * self.param_for_margin
        encoded_entropy = entropy.to(sorted_TDS_normalized.dtype) * self.param_for_entropy
        encoded_delta_margin = delta_margin.to(sorted_TDS_normalized.dtype) * self.param_for_delta_margin
        encoded_delta_entropy = delta_entropy.to(sorted_TDS_normalized.dtype) * self.param_for_delta_entropy
            
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
                    
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        
        # Feature Fusion
        raw_features = torch.cat([encoded_ATP_R, encoded_normalized_ATP, encoded_margin, encoded_entropy, encoded_delta_margin, encoded_delta_entropy], dim=-1)
        x = self.feature_fusion(raw_features)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x += self.pos_embedding(pos_indices)

        for layer in self.attention_layers:
            x = layer(x)

        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'attention':
            attn_weights = F.softmax(self.attention_pool(x), dim=1) 
            x = (x * attn_weights).sum(dim=1) 
        else: 
            x = x[:, 0]

        x = self.mlp_head(x)
        x = torch.nan_to_num(x, nan=0.0) 
        
        return self.sigmoid(x).squeeze(-1)
    

class LOS_Net(nn.Module):
    def __init__(self, args, max_sequence_length, input_dim=1):
        super().__init__()
        self.args = args
        self.max_sequence_length = max_sequence_length
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.heads = args.heads
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = args.pool
        
        assert self.pool in {'cls', 'mean', 'attention'}, "Pool type must be 'cls', 'mean', or 'attention'"
        
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))
        self.param_for_margin = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))
        self.param_for_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))
        
        # Temporal Dynamics Parameters
        self.param_for_delta_margin = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))
        self.param_for_delta_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))

        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM], self.hidden_dim // 2)
        else:
            raise ValueError("Invalid encoding type.")
        
        # Non-linear Feature Fusion MLP
        # Concatenating 6 features of size (hidden_dim // 2) -> 3 * hidden_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear((self.hidden_dim // 2) * 6, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )

        self.input_proj = nn.Linear(input_dim, self.hidden_dim // 2)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)
        
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        if self.pool == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_dim // 2, 1)
            )
        
        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
    
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        sorted_TDS_normalized = torch.nan_to_num(sorted_TDS_normalized, nan=0.0)
        normalized_ATP = torch.nan_to_num(normalized_ATP, nan=0.0)

        tds_safe = sorted_TDS_normalized.to(torch.float32)
        margin = tds_safe[:, :, 0:1] - normalized_ATP.to(torch.float32)
        
        probs = F.softmax(tds_safe, dim=-1)
        safe_probs = torch.clamp(probs, min=1e-7)
        entropy = -(probs * torch.log(safe_probs)).sum(dim=-1, keepdim=True)
        
        # Temporal Dynamics
        delta_margin = torch.zeros_like(margin)
        delta_margin[:, 1:, :] = margin[:, 1:, :] - margin[:, :-1, :]
        
        delta_entropy = torch.zeros_like(entropy)
        delta_entropy[:, 1:, :] = entropy[:, 1:, :] - entropy[:, :-1, :]
        
        encoded_margin = margin.to(sorted_TDS_normalized.dtype) * self.param_for_margin
        encoded_entropy = entropy.to(sorted_TDS_normalized.dtype) * self.param_for_entropy
        encoded_delta_margin = delta_margin.to(sorted_TDS_normalized.dtype) * self.param_for_delta_margin
        encoded_delta_entropy = delta_entropy.to(sorted_TDS_normalized.dtype) * self.param_for_delta_entropy

        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
            
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        
        # Mapping normalized vocab
        encoded_sorted_TDS_normalized = self.input_proj(sorted_TDS_normalized.to(torch.float32))
        
        # Feature Fusion
        raw_features = torch.cat([encoded_ATP_R, encoded_normalized_ATP, encoded_margin, encoded_entropy, encoded_delta_margin, encoded_delta_entropy], dim=-1)
        x_scalars = self.feature_fusion(raw_features)
        
        # Combine contextual features with the scalars
        x = torch.cat((encoded_sorted_TDS_normalized, x_scalars), dim=-1)
        
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x += self.pos_embedding(pos_indices)
        
        for layer in self.attention_layers:
            x = layer(x)
        
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'attention':
            attn_weights = F.softmax(self.attention_pool(x), dim=1) 
            x = (x * attn_weights).sum(dim=1) 
        else: 
            x = x[:, 0]
        
        x = self.mlp_head(x)
        x = torch.nan_to_num(x, nan=0.0) 
        
        return self.sigmoid(x).squeeze(-1)
######################## LOS ########################