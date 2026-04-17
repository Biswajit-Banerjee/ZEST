import torch
import torch.nn as nn
import torch.nn.functional as F
from zest.model.positional import ScaledPositionalEmbedding

class ExpansionHead(nn.Module):
    """
    Expands token embeddings to residue-level logits over 20 AAs.
    """
    def __init__(
        self,
        embed_dim: int,
        max_residues: int,
        vocab_size: int = 20,
        padding_index: int = 31,
        local_extend_strategy: str = "alibi",
    ):
        super().__init__()

        self.max_residues = max_residues
        self.PADDING_INDEX = padding_index
        self.local_pos_emb = ScaledPositionalEmbedding(
            num_embeddings=self.PADDING_INDEX + 1,
            embedding_dim=embed_dim,
            extend_strategy=local_extend_strategy,
            padding_idx=self.PADDING_INDEX,
        )
        
        self.residue_expansion = self.bottleneck_mlp(embed_dim, embed_dim, 0.1, 2)
        self.aa_logits_proj = nn.Linear(embed_dim, vocab_size)
        
    
    def bottleneck_mlp(self, input_dim, hidden, dropout, num_proj_layers, output_dim=None):
        layers = []
        
        # Linear Norm GELU
        lng_layer = lambda input_dim, hidden, dropout: [
            nn.Linear(input_dim, hidden), 
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        ]
        
        for _ in range(num_proj_layers):
            layers.extend(
                lng_layer(input_dim, hidden, dropout) 
                + lng_layer(hidden, input_dim, dropout)
            )
        
        if output_dim is not None:
            layers.append(nn.Linear(input_dim, output_dim))
        
        return nn.Sequential(*layers)

    def _expand_tokens(
        self,
        z: torch.Tensor,
        token_lengths: torch.Tensor,
        target_len: int
    ) -> torch.Tensor:
        """
        Fully vectorized token expansion - no Python loops over batch.
        
        Args:
            z: [B, T, D] token embeddings
            token_lengths: [B, T] length of each token (how many residues it covers)
            target_len: target sequence length for residues
            
        Returns:
            z_expanded: [B, target_len, D] expanded embeddings
            local_indices: [B, target_len] local position indices (0 to length-1 per token)
        """
        batch, num_tokens, embed_dim = z.shape
        device = z.device
        
        # Ensure token_lengths is a tensor and clamp to valid range
        if not torch.is_tensor(token_lengths):
            token_lengths = torch.tensor(token_lengths, device=device, dtype=torch.long)
                
        # Clamp lengths to prevent issues (lengths should be >= 0)
        token_lengths = token_lengths.clone()
        token_lengths = torch.clamp(token_lengths, min=0)
        
        # Calculate cumulative positions for each token
        # cumsum gives us the END position of each token's residues
        cumsum = torch.cumsum(token_lengths, dim=1)  # [B, T]
        
        # Total residues per batch element
        total_residues = cumsum[:, -1]  # [B]
        
        # Create position indices [0, 1, 2, ..., target_len-1] for each batch
        positions = torch.arange(target_len, device=device).unsqueeze(0).expand(batch, -1)  # [B, target_len]
        
        # For each residue position, find which token it belongs to
        # A residue at position p belongs to token t if cumsum[t-1] <= p < cumsum[t]
        # Using searchsorted: find the first token where cumsum > position
        # This is the token that "owns" this residue position
        token_indices = torch.searchsorted(cumsum, positions + 1)  # [B, target_len]
        token_indices = torch.clamp(token_indices, 0, num_tokens - 1)
        
        # Gather embeddings based on token indices
        # token_indices: [B, target_len] -> expand to [B, target_len, D]
        token_indices_expanded = token_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        z_expanded = torch.gather(z, 1, token_indices_expanded)  # [B, target_len, D]
        
        # Calculate local indices (position within each token)
        # local_idx = position - start_of_token = position - (cumsum[token-1] if token > 0 else 0)
        # Shift cumsum by 1 to get start positions (prepend 0)
        cumsum_shifted = F.pad(cumsum[:, :-1], (1, 0), value=0)  # [B, T]
        start_positions = torch.gather(cumsum_shifted, 1, token_indices)  # [B, target_len]
        local_indices = positions - start_positions  # [B, target_len]
        
        # Mask out positions beyond actual sequence length
        # Positions >= total_residues should be padding
        padding_mask = positions >= total_residues.unsqueeze(1)  # [B, target_len]
        
        # Zero out padded embeddings
        z_expanded = z_expanded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        # Set padded local indices to PADDING_INDEX; keep true values elsewhere (may exceed padding index)
        local_indices = torch.where(
            padding_mask,
            torch.full_like(local_indices, self.PADDING_INDEX),
            local_indices.clamp(min=0),
        )
            
        return z_expanded, local_indices

    def get_residue_embeddings(
        self,
        z: torch.Tensor,
        token_lengths,
        global_indices: torch.Tensor,
        global_pos_emb: nn.Embedding
    ) -> torch.Tensor:
        """
        Returns residue-level embeddings with positional encoding.
        
        Args:
            z: [B, T, D] token embeddings from encoder
            token_lengths: [B, T] length of each token
            global_indices: [B, L] global residue positions
            global_pos_emb: embedding layer for global positions
            
        Returns:
            z_final: [B, L, D] residue embeddings with local+global position info
        """
        target_len = global_indices.shape[1]
        # 1. Expand Z and get Local Context
        z_expanded, local_indices = self._expand_tokens(z, token_lengths, target_len)
        
        # 2. Global Position (allow extrapolation beyond max_residues)
        global_indices = torch.clamp(global_indices, min=0)
        pos_global = global_pos_emb(global_indices).to(z_expanded.dtype)

        # 3. Local Position (already padded within _expand_tokens)
        pos_local = self.local_pos_emb(local_indices).to(z_expanded.dtype)
        
        # 4. Combine
        z_final = z_expanded + pos_global + pos_local
        
        return z_final

    def forward(
        self,
        z: torch.Tensor,
        token_lengths,
        global_indices: torch.Tensor,
        global_pos_emb: nn.Embedding
    ) -> torch.Tensor:
        """
        Forward pass for profile prediction (emission distributions).
        """
        z_final = self.get_residue_embeddings(z, token_lengths, global_indices, global_pos_emb)
        res_logits = self.residue_expansion(z_final)
        aa_logits  = self.aa_logits_proj(res_logits)
        return aa_logits

    def get_residue_logits(self, z: torch.Tensor, token_lengths, global_indices: torch.Tensor, global_pos_emb: nn.Embedding) -> torch.Tensor:
        z_final = self.get_residue_embeddings(z, token_lengths, global_indices, global_pos_emb)
        res_logits = self.residue_expansion(z_final)
        return res_logits
