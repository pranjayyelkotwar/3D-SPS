import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector3D(nn.Module):
    """
    Projector module to map both object features and language features into a similar embedding space.
    Used for contrastive learning between 3D object proposals and text descriptions.
    """
    
    def __init__(self, object_dim, lang_dim, out_dim=512):
        """
        Initialize the projector with separate heads for object and language features.
        
        Args:
            object_dim (int): Input object feature dimension
            lang_dim (int): Input language feature dimension
            out_dim (int): Output embedding dimension (default: 512)
        """
        super().__init__()
        
        # Separate projection heads for object and language features
        self.object_projector = nn.Sequential(
            nn.Linear(object_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, out_dim)
        )
        
        self.lang_projector = nn.Sequential(
            nn.Linear(lang_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, out_dim)
        )
    
    def forward(self, object_feat=None, lang_feat=None):
        """
        Forward pass that projects features and applies L2 normalization.
        
        Args:
            object_feat (torch.Tensor, optional): Object features of shape [..., object_dim]
            lang_feat (torch.Tensor, optional): Language features of shape [..., lang_dim]
            
        Returns:
            dict: Dictionary containing projected features
                - 'object': L2-normalized projected object features if object_feat provided
                - 'lang': L2-normalized projected language features if lang_feat provided
        """
        results = {}
        
        if object_feat is not None:
            projected_obj = self.object_projector(object_feat)
            results['object'] = F.normalize(projected_obj, dim=-1)  # L2 norm
        
        if lang_feat is not None:
            projected_lang = self.lang_projector(lang_feat)
            results['lang'] = F.normalize(projected_lang, dim=-1)  # L2 norm
            
        return results