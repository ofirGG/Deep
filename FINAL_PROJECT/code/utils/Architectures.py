import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.constants import MODEL_VOCAB_SIZES
from einops import repeat

def get_model(args, max_sequence_length, actual_sequence_length, input_dim, input_shape):
    model_mapping = {
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
        
        # --- Early Fusion Projector ---
        self.stats_projector = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, self.hidden_dim)
        )
        
        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM], self.hidden_dim)
        else:
            raise ValueError("Invalid encoding type.")

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

    # --- NOW TAKES STATS AS AN ARGUMENT ---
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R, stats):
        encoded_stats = self.stats_projector(stats.to(torch.float32))

        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
                    
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        
        x = encoded_ATP_R + encoded_normalized_ATP + encoded_stats
        x = x.flatten(start_dim=1)
        
        for i in range(self.num_layers):
            x = self.lin_layers[i](x)
            if (i+1) < self.num_layers:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)
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

        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        self.stats_projector = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, self.hidden_dim)
        )
        
        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM], self.hidden_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, nhead=self.heads, dropout=self.dropout, dim_feedforward=self.hidden_dim, batch_first=True
            ) for _ in range(self.num_layers)
        ])
        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
    
    # --- NOW TAKES STATS AS AN ARGUMENT ---
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R, stats):
        encoded_stats = self.stats_projector(stats.to(torch.float32))
            
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
                    
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        
        x = encoded_ATP_R + encoded_normalized_ATP + encoded_stats

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  
        x = torch.cat((cls_tokens, x), dim=1)  
        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)  
        x += self.pos_embedding(pos_indices)  

        for layer in self.attention_layers:
            x = layer(x)  

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)  
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
        
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))

        self.stats_projector = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, self.hidden_dim // 2) # Important: Projects to half dim for LOS_Net
        )

        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM], self.hidden_dim // 2)
        
        self.input_proj = nn.Linear(input_dim, self.hidden_dim // 2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)
        
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, nhead=self.heads, dropout=self.dropout, dim_feedforward=self.hidden_dim, batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
    
    # --- NOW TAKES STATS AS AN ARGUMENT ---
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R, stats):
        encoded_stats = self.stats_projector(stats.to(torch.float32))

        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
            
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        encoded_sorted_TDS_normalized = self.input_proj(sorted_TDS_normalized.to(torch.float32))
        
        x = torch.cat((encoded_sorted_TDS_normalized, encoded_ATP_R + encoded_normalized_ATP + encoded_stats), dim=-1)
        
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x += self.pos_embedding(pos_indices)
        
        for layer in self.attention_layers:
            x = layer(x)
        
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)
        return self.sigmoid(x).squeeze(-1)