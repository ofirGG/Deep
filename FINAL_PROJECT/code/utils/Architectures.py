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
        
        # Feature Engineering Parameters
        self.param_for_margin = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM],
            self.hidden_dim,
            # sparse=True
            )
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")

        
        # Linear layers
        self.lin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_dim if i > 0 else self.hidden_dim * self.actual_sequence_length
            out_dim = self.hidden_dim if (i+1) < self.num_layers else 1
            self.lin_layers.append(nn.Linear(in_dim, out_dim))
            if (i+1) < self.num_layers:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Output act
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        """
        Computes encoded_ATP_R based on normalized_ATP and ATP_R.
        """
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R

    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        # Defensive check against NaNs from preprocessing
        sorted_TDS_normalized = torch.nan_to_num(sorted_TDS_normalized, nan=0.0)
        normalized_ATP = torch.nan_to_num(normalized_ATP, nan=0.0)

        # Feature Engineering: Calculate Margin and Local Entropy safely in float32
        tds_safe = sorted_TDS_normalized.to(torch.float32)
        margin = tds_safe[:, :, 0:1] - normalized_ATP.to(torch.float32)
        
        probs = F.softmax(tds_safe, dim=-1)
        safe_probs = torch.clamp(probs, min=1e-7) # Safe from float16 underflow
        entropy = -(probs * torch.log(safe_probs)).sum(dim=-1, keepdim=True)
        
        encoded_margin = margin.to(sorted_TDS_normalized.dtype) * self.param_for_margin
        encoded_entropy = entropy.to(sorted_TDS_normalized.dtype) * self.param_for_entropy

        # Encoding one-hot rank
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
                    
        # Encoding normalized mark
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        
        # Combine all scalar embeddings
        x = encoded_ATP_R + encoded_normalized_ATP + encoded_margin + encoded_entropy
        x = x.flatten(start_dim=1)
        
        for i in range(self.num_layers):
            x = self.lin_layers[i](x)
            if (i+1) < self.num_layers:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout)

        x = torch.nan_to_num(x, nan=0.0) # Final safeguard
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
        
        # Updated pool choices
        assert self.pool in {'cls', 'mean', 'attention'}, "Pool type must be 'cls', 'mean', or 'attention'"

        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Feature Engineering Parameters
        self.param_for_margin = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.param_for_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM],
            self.hidden_dim,
            )
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        # Positional embeddings with a predefined max sequence length
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)

        # Transformer encoder layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Attention Pooling Layer
        if self.pool == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_dim // 2, 1)
            )

        # Classification head
        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        """
        Computes encoded_ATP_R based on normalized_ATP and ATP_R.
        """
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
    
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        # Defensive check against NaNs
        sorted_TDS_normalized = torch.nan_to_num(sorted_TDS_normalized, nan=0.0)
        normalized_ATP = torch.nan_to_num(normalized_ATP, nan=0.0)

        # Feature Engineering safely
        tds_safe = sorted_TDS_normalized.to(torch.float32)
        margin = tds_safe[:, :, 0:1] - normalized_ATP.to(torch.float32)
        
        probs = F.softmax(tds_safe, dim=-1)
        safe_probs = torch.clamp(probs, min=1e-7)
        entropy = -(probs * torch.log(safe_probs)).sum(dim=-1, keepdim=True)
        
        encoded_margin = margin.to(sorted_TDS_normalized.dtype) * self.param_for_margin
        encoded_entropy = entropy.to(sorted_TDS_normalized.dtype) * self.param_for_entropy

        # Encoding one-hot rank
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
                    
        # Encoding normalized mark
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        
        x = encoded_ATP_R + encoded_normalized_ATP + encoded_margin + encoded_entropy

        # Add [CLS] token
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # Generate positional indices and add embeddings
        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)
        pos_embeddings = self.pos_embedding(pos_indices)
        x += pos_embeddings

        # Pass through Transformer layers
        for layer in self.attention_layers:
            x = layer(x)

        # Pooling logic
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'attention':
            attn_weights = F.softmax(self.attention_pool(x), dim=1) # Shape: [B, N+1, 1]
            x = (x * attn_weights).sum(dim=1) # Weighted sum over sequence
        else: # 'cls'
            x = x[:, 0]

        # Final classification head
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
        
        # Updated pool choices
        assert self.pool in {'cls', 'mean', 'attention'}, "Pool type must be 'cls', 'mean', or 'attention'"
        
        self.param_for_normalized_ATP = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))

        # Feature Engineering Parameters
        self.param_for_margin = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))
        self.param_for_entropy = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))

        if self.args.rank_encoding == 'scale_encoding':
            self.param_for_ATP_R = nn.Parameter(torch.randn(1, 1, self.hidden_dim // 2))        
        elif self.args.rank_encoding == 'one_hot_encoding':
            self.one_hot_embedding = nn.Embedding(MODEL_VOCAB_SIZES[self.args.LLM],
            self.hidden_dim // 2,
            )
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
        
        # Input embedding layer
        self.input_proj = nn.Linear(input_dim, self.hidden_dim // 2)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(self.max_sequence_length + 1, self.hidden_dim)
        
        # Transformer encoder layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.heads,
                dropout=self.dropout,
                dim_feedforward=self.hidden_dim,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Attention Pooling Layer
        if self.pool == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_dim // 2, 1)
            )
        
        # Classification head
        self.mlp_head = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def compute_encoded_ATP_R(self, normalized_ATP, ATP_R):
        """
        Computes encoded_ATP_R based on normalized_ATP and ATP_R.
        """
        encoded_ATP_R = 2 * (0.5 - (ATP_R / MODEL_VOCAB_SIZES[self.args.LLM]))
        return normalized_ATP * encoded_ATP_R.unsqueeze(-1) * self.param_for_ATP_R
    
    def forward(self, sorted_TDS_normalized, normalized_ATP, ATP_R):
        # Defensive check against NaNs
        sorted_TDS_normalized = torch.nan_to_num(sorted_TDS_normalized, nan=0.0)
        normalized_ATP = torch.nan_to_num(normalized_ATP, nan=0.0)

        # Feature Engineering safely
        tds_safe = sorted_TDS_normalized.to(torch.float32)
        margin = tds_safe[:, :, 0:1] - normalized_ATP.to(torch.float32)
        
        probs = F.softmax(tds_safe, dim=-1)
        safe_probs = torch.clamp(probs, min=1e-7)
        entropy = -(probs * torch.log(safe_probs)).sum(dim=-1, keepdim=True)
        
        encoded_margin = margin.to(sorted_TDS_normalized.dtype) * self.param_for_margin
        encoded_entropy = entropy.to(sorted_TDS_normalized.dtype) * self.param_for_entropy

        # Encoding one-hot rank
        if self.args.rank_encoding == 'scale_encoding':
            encoded_ATP_R = self.compute_encoded_ATP_R(normalized_ATP=normalized_ATP, ATP_R=ATP_R)
        elif self.args.rank_encoding == 'one_hot_encoding':
            encoded_ATP_R = normalized_ATP * self.one_hot_embedding(ATP_R)
        else:
            raise ValueError("Invalid encoding type. Please choose either 'scale_encoding' or 'one_hot_encoding'.")
            
        # Encoding normalized mark
        encoded_normalized_ATP = normalized_ATP * self.param_for_normalized_ATP
        
        # Encoding normalized vocab
        encoded_sorted_TDS_normalized = self.input_proj(sorted_TDS_normalized.to(torch.float32))
        
        x_scalars = encoded_ATP_R + encoded_normalized_ATP + encoded_margin + encoded_entropy
        x = torch.cat((encoded_sorted_TDS_normalized, x_scalars), dim=-1)
        
        # Adding CLS token
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Positional embeddings
        pos_indices = torch.arange(n + 1, device=x.device).unsqueeze(0)
        x += self.pos_embedding(pos_indices)
        
        # Transformer layers
        for layer in self.attention_layers:
            x = layer(x)
        
        # Pooling logic
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'attention':
            attn_weights = F.softmax(self.attention_pool(x), dim=1) # Shape: [B, N+1, 1]
            x = (x * attn_weights).sum(dim=1) # Weighted sum over sequence
        else: # 'cls'
            x = x[:, 0]
        
        # Classification head
        x = self.mlp_head(x)
        x = torch.nan_to_num(x, nan=0.0) # Final safeguard
        return self.sigmoid(x).squeeze(-1)
######################## LOS ########################