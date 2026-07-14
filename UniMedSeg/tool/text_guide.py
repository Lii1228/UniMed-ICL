import torch
import torch.nn as nn
import os
import math  

# ==============================================================================
# 1. Label Mapping Table 
# ==============================================================================
FREESURFER_LUT = {
    0: "Background",
    2: "Cerebral White Matter",       
    3: "Cerebral Cortex",             
    4: "Lateral Ventricle",           
    5: "Inferior Lateral Ventricle",
    7: "Cerebellum White Matter",
    8: "Cerebellum Cortex",
    10: "Thalamus",                   
    11: "Caudate",
    12: "Putamen",
    13: "Pallidum",
    14: "Third Ventricle",            
    15: "Fourth Ventricle",
    16: "Brain Stem",
    17: "Hippocampus",                
    18: "Amygdala",
    24: "Cerebrospinal Fluid",
    26: "Accumbens Area",
    28: "Ventral DC",
    30: "Vessel",
    31: "Choroid Plexus",
    # --- Map right hemisphere IDs to the same generic names ---
    41: "Cerebral White Matter",      
    42: "Cerebral Cortex",
    43: "Lateral Ventricle",
    44: "Inferior Lateral Ventricle",
    46: "Cerebellum White Matter",
    47: "Cerebellum Cortex",
    49: "Thalamus",
    50: "Caudate",
    51: "Putamen",
    52: "Pallidum",
    53: "Hippocampus",                
    54: "Amygdala",
    58: "Accumbens Area",
    60: "Ventral DC",
    62: "Vessel",
    63: "Choroid Plexus",
    # --- Miscellaneous ---
    72: "Fifth Ventricle",
    77: "White Matter Hypointensities",
    80: "Non White Matter Hypointensities",
    85: "Optic Chiasm",
    251: "Posterior Corpus Callosum",
    252: "Mid Posterior Corpus Callosum",
    253: "Central Corpus Callosum",
}

# ==============================================================================
# 2. Text Embedding Adapter (Supports dynamic token computation)
# ==============================================================================
class TextEmbeddingAdapter(nn.Module):
    """
    Manages the offline BiomedBERT cache and trainable projection layers.
    Automatically calculates the required number of tokens based on target_embed_dim 
    to prevent information loss.
    """
    def __init__(self, cache_path="biomedbert_embeddings_cache.pt", target_embed_dim=432):
        super().__init__()
        
        # ================= [Added] Logic for automatically locating sibling files =================
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sibling_path = os.path.join(current_dir, os.path.basename(cache_path))
        
        final_path = None

        if os.path.exists(sibling_path):
            final_path = sibling_path
            print(f"[TextAdapter] Found cache in module directory: {final_path}")
        elif os.path.exists(cache_path):
            final_path = cache_path
            print(f"[TextAdapter] Found cache in working directory: {final_path}")
        else:
            print(f"[TextAdapter] WARNING: Cache file NOT found.")
            print(f"              Checked: {sibling_path}")
            print(f"              Checked: {cache_path}")
        # ============================================================================================

        # 3. Load Cache
        if final_path and os.path.exists(final_path):
            self.embedding_cache = torch.load(final_path, map_location='cpu')
            self.cache_available = True
        else:
            self.embedding_cache = {}
            self.cache_available = False
            
        self.bert_dim = 768
        self.target_dim = target_embed_dim
        
        # 2. Automatically compute tokens_per_word
        self.tokens_per_word = math.ceil(self.bert_dim / self.target_dim)
        
        print(f"[TextAdapter] Auto-calculated tokens per word: {self.tokens_per_word}")
        print(f"              (BERT {self.bert_dim} -> Model {self.target_dim} x {self.tokens_per_word} = {self.target_dim * self.tokens_per_word})")
        
        # 3. [Modified] Define the projection layer as a two-layer MLP
        # Structure: Linear(768 -> 768) -> GELU -> Linear(768 -> target * k)
        output_dim = self.target_dim * self.tokens_per_word
        hidden_dim = self.bert_dim  # The hidden layer dimension is kept consistent with the BERT output, but can be adjusted as needed
        
        self.projection = nn.Sequential(
            nn.Linear(self.bert_dim, hidden_dim),
            nn.GELU(),  # GELU typically performs better than ReLU in BERT/Transformer-based tasks
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialization (iterate through all layers in Sequential)
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, id_list_batch, device):
        """
        Input: id_list_batch (List[List[int]]), e.g., [[17, 53], [0]]
        Output: tensor [B, L * tokens_per_word, target_dim]
        """
        if not self.cache_available:
            return None

        batch_embeddings = []
        max_len = 0
        
        # --- A. Table lookup to extract 768-dimensional BERT features ---
        for ids in id_list_batch:
            def flatten(item):
                if isinstance(item, (list, tuple)):
                    return [x for sub in item for x in flatten(sub)]
                return [item]
                
            flat_ids = flatten(ids)

            if not flat_ids or (len(flat_ids) == 1 and flat_ids[0] == 0):
                # Padding for empty features
                feat = torch.zeros(1, 1, self.bert_dim).to(device)
            else:
                key = tuple(sorted(flat_ids)) # Generate a tuple using the flattened list
                if key in self.embedding_cache:
                    feat = self.embedding_cache[key].to(device)
                else:
                    feat = torch.zeros(1, 1, self.bert_dim).to(device)
            
            batch_embeddings.append(feat)
            if feat.shape[1] > max_len:
                max_len = feat.shape[1]
                
        # --- B. Batch Padding ---
        padded_batch = []
        for feat in batch_embeddings:
            curr_len = feat.shape[1]
            if curr_len < max_len:
                pad_size = max_len - curr_len
                padding = torch.zeros(1, pad_size, self.bert_dim).to(device)
                feat = torch.cat([feat, padding], dim=1)
            padded_batch.append(feat)
            
        bert_features = torch.cat(padded_batch, dim=0) # [B, max_L, 768]
        
        # --- C. Projection (MLP) and Reshaping ---
        # 1. Projection: [B, L, 768] -> [B, L, target_dim * k]
        # Here, self.projection is a Sequential module containing two Linear layers
        projected = self.projection(bert_features)
        
        # 2. Reshape: "Fold" the expanded dimensions into the sequence length
        # [B, L, target_dim * k] -> [B, L * k, target_dim]
        B, L, _ = projected.shape
        language_tokens = projected.view(
            B, 
            L * self.tokens_per_word, 
            self.target_dim
        )
        
        return language_tokens
