import torch.nn.functional as F
import os
from tqdm import tqdm
import torch
from utils.constants import MAXIMAL_VOCAB_SIZE
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import multiprocessing

def stratified_split(dataset, percentage=1/5, random_state=42):
    if not (0 < percentage <= 1):
        raise ValueError("Percentage must be a value between 0 and 1.")
    num_splits = int(1 / percentage)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Cannot perform stratified split.")
    labels = [dataset[i][-1] for i in tqdm(range(len(dataset)), desc="Extracting labels")]
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    splits = [indices for _, indices in skf.split(np.zeros(len(labels)), labels)]
    return splits
    
def get_train_val_test_indices(splits):
    num_chunks = len(splits)
    train_indices, val_indices, test_indices = [], [], []
    for i in tqdm(range(num_chunks), desc="Processing folds"):
        train_chunks = [splits[j] for j in range(num_chunks) if j != i and j != (i + 1) % num_chunks]
        val_chunk = splits[(i + 1) % num_chunks]
        test_chunk = splits[i]
        train_indices.append(np.concatenate(train_chunks))
        val_indices.append(val_chunk)
        test_indices.append(test_chunk)
    return train_indices, val_indices, test_indices

def pad_tensor(tensor, N_max, pad_value):
    N, d = tensor.shape
    if N_max <= N:
        return tensor[:N_max, :]
    padded_tensor = torch.full((N_max, d), pad_value, dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:N, :] = tensor
    return padded_tensor

def patch_down_sample(tensor: torch.Tensor, L_new: int, N_new: int, method: str = "interpolation"):
    L, N, d = tensor.shape
    tensor = tensor.permute(2, 0, 1)  
    if method in ["sum_pool", "mean_pool", "max_pool"]:
        L_pad = (L_new - (L % L_new)) % L_new + L
        N_pad = (N_new - (N % N_new)) % N_new + N
        pad_L = L_pad - L
        pad_N = N_pad - N
        tensor = F.pad(tensor, (0, pad_N, 0, pad_L), mode="replicate")  
        factor_L = L_pad // L_new
        factor_N = N_pad // N_new
        tensor = tensor.unfold(1, factor_L, factor_L).unfold(2, factor_N, factor_N)
        if method == "sum_pool":
            tensor = tensor.sum(dim=(-1, -2))  
        elif method == "mean_pool":
            tensor = tensor.mean(dim=(-1, -2))  
        elif method == "max_pool":
            tensor = tensor.max(dim=-1)[0].max(dim=-1)[0]  
    elif method == "interpolation":
        tensor = tensor.unsqueeze(0)  
        tensor = F.interpolate(tensor, size=(L_new, N_new), mode='bilinear', align_corners=False)
        tensor = tensor.squeeze(0)  
    else:
        raise ValueError("Unsupported method. Choose from 'sum_pool', 'mean_pool', or 'interpolation'.")
    return tensor.permute(1, 2, 0)  

def process_file(params):
    file_idx, data_dir, output_dir, N_max, topk_preprocess, input_output_flag, input_type = params
    
    label_path = os.path.join(data_dir, f'label_{file_idx}.pt')
    if os.path.exists(label_path):
        label = torch.load(label_path)

    output_paths = [
        os.path.join(output_dir, f'TDS_topk_{input_output_flag}_{topk_preprocess}_{file_idx}.pt'),
        os.path.join(output_dir, f'ATP_{input_output_flag}_{file_idx}.pt'),
        os.path.join(output_dir, f'ATP_R_{input_output_flag}_{file_idx}.pt'),            
        os.path.join(output_dir, f'stats_{input_output_flag}_{file_idx}.pt'), # <--- NEW STATS FILE
        os.path.join(output_dir, f'label_{file_idx}.pt'),
    ]

    if (not all(os.path.exists(path) for path in output_paths[:4])) and input_type == "LOS":
        probs_path = os.path.join(data_dir, f'probs_{input_output_flag}_{file_idx}.pt')
        TDS = torch.load(probs_path)
        
        def get_ranks(TDS: torch.Tensor) -> torch.Tensor:
            vocab_size = TDS[:, :-2].shape[-1]
            one_hot_rank = pad_tensor(
                F.one_hot(TDS[:, -1].to(torch.int64), num_classes=vocab_size),
                N_max=N_max, pad_value=-1
            )
            mask = (one_hot_rank == -1).all(dim=1)  
            one_hot_rank[mask] = torch.zeros(vocab_size, dtype=torch.long)  
            one_hot_rank[mask, -1] = MAXIMAL_VOCAB_SIZE  
            ATP_R = torch.argmax(one_hot_rank, dim=-1)
            return ATP_R

        def normalize_sorted_probs(raw_probs: torch.Tensor, fixed_k: int = 1000) -> torch.Tensor:
            sorted_raw_probs = torch.sort(raw_probs[:, :-2], descending=True)[0]
            stable_probs_for_stats = sorted_raw_probs[:, :fixed_k]
            mu = torch.mean(stable_probs_for_stats, dim=-1, keepdim=True)
            std = torch.std(stable_probs_for_stats, dim=-1, keepdim=True) + 1e-8
            sorted_raw_probs_normalized = (sorted_raw_probs - mu) / std
            return sorted_raw_probs_normalized, mu, std
        
        # --- NEW: Extract Raw Stats Before Normalization ---
        sorted_raw_probs = torch.sort(TDS[:, :-2], descending=True)[0]
        topk_raw = sorted_raw_probs[:, :1000].to(torch.float32)
        probs_dist = F.softmax(topk_raw, dim=-1)
        safe_probs = torch.clamp(probs_dist, min=1e-7)
        
        entropy = -(probs_dist * torch.log(safe_probs)).sum(dim=-1, keepdim=True)
        delta_entropy = torch.zeros_like(entropy)
        delta_entropy[1:, :] = entropy[1:, :] - entropy[:-1, :]
        variance = torch.var(probs_dist, dim=-1, keepdim=True) + 1e-8
        dist_range = (torch.max(probs_dist, dim=-1, keepdim=True).values - torch.min(probs_dist, dim=-1, keepdim=True).values) + 1e-8
        
        stats = torch.cat([entropy, delta_entropy, variance, dist_range], dim=-1)
        stats_padded = pad_tensor(stats, N_max=N_max, pad_value=0.0)
        # ---------------------------------------------------

        sorted_TDS_normalized, mu, std = normalize_sorted_probs(raw_probs=TDS, fixed_k=1000)        
        sorted_TDS_normalized = pad_tensor(sorted_TDS_normalized, N_max=N_max, pad_value=0)
        normalized_ATP = pad_tensor((TDS[:, -2].reshape(-1, 1) - mu) / std, N_max=N_max, pad_value=0)
        ATP_R = get_ranks(TDS)
        
        torch.save(sorted_TDS_normalized[:, :topk_preprocess].to(torch.float16), output_paths[0])
        torch.save(normalized_ATP.to(torch.float16), output_paths[1])
        torch.save(ATP_R.to(torch.int32), output_paths[2])
        torch.save(stats_padded.to(torch.float32), output_paths[3]) # Save stats!
        torch.save(label, output_paths[4])

    else:
        print(f'Sample {file_idx} already preprocessed. Skipping...')

def preprocess_data(args, raw_data_dir, LLM, dataset_name, output_dir, N_max=100, topk_preprocess=1000000, input_output_flag = 'input', input_type = "LOS"):
    assert input_output_flag in ['input', 'output'], "input_output_flag must be 'input' or 'output'"
    data_dir = os.path.join(raw_data_dir, LLM, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    files = os.listdir(data_dir)
    file_indices = sorted(set(int(f.split('_')[-1].split('.')[0]) for f in files if f.startswith('probs')))
    params_list = [(idx, data_dir, output_dir, N_max, topk_preprocess, input_output_flag, input_type) for idx in file_indices]

    num_workers = min(int(0.15*multiprocessing.cpu_count()), len(file_indices))
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_file, params_list), total=len(file_indices), desc="Preprocessing files"))

class CustomSavedDataset(Dataset):
    def __init__(self, preprocessed_dir, topk_preprocess=1000000, topk_dim=1000, input_output_flag='input', input_type = "LOS"):
        self.preprocessed_dir = preprocessed_dir
        self.file_indices = self._get_indices()
        self.topk_dim = topk_dim
        self.input_type = input_type
    
        self.file_paths = {
            idx: {
                'sorted_TDS_normalized': os.path.join(self.preprocessed_dir, f'TDS_topk_{input_output_flag}_{topk_preprocess}_{idx}.pt'),
                'normalized_ATP': os.path.join(self.preprocessed_dir, f'ATP_{input_output_flag}_{idx}.pt'),
                'ATP_R': os.path.join(self.preprocessed_dir, f'ATP_R_{input_output_flag}_{idx}.pt'),
                'stats': os.path.join(self.preprocessed_dir, f'stats_{input_output_flag}_{idx}.pt'), # <--- Loading stats
                'label': os.path.join(self.preprocessed_dir, f'label_{idx}.pt')
            }
            for idx in self.file_indices
        }
      
    def _get_indices(self):
        files = os.listdir(self.preprocessed_dir)
        indices = sorted(set(int(f.split('_')[-1].split('.')[0]) for f in files if f.startswith('') and 'label' in f))
        return indices

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        paths = self.file_paths[file_idx]

        label = torch.load(paths['label'])
        if self.input_type == "LOS":
            sorted_TDS_normalized = torch.load(paths['sorted_TDS_normalized'])[:, :self.topk_dim]
            normalized_ATP = torch.load(paths['normalized_ATP'])
            ATP_R = torch.load(paths['ATP_R'])
            stats = torch.load(paths['stats']) # Return stats!
            return (sorted_TDS_normalized, normalized_ATP, ATP_R, stats, label)
        else:
            raise ValueError("Invalid input type.")