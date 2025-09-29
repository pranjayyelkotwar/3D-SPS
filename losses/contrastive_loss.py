import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Multi-positive InfoNCE contrastive loss for 3D visual grounding.
    
    This loss maps object features and language features into a similar embedding space
    using contrastive learning. It supports multiple positive selection strategies
    and weighting schemes based on IoU, distance, or uniform weighting.
    """
    
    def __init__(self, 
                 temperature=0.07,
                 positive_selection='iou_threshold',
                 iou_threshold=0.25,
                 top_k1=5,
                 top_k2=32,
                 weighting='uniform',
                 sigma=1.0,
                 symmetric=False):
        """
        Initialize contrastive loss.
        
        Args:
            temperature (float): Temperature parameter τ for scaling logits (default: 0.07)
            positive_selection (str): Method for selecting positives ('iou_threshold', 'top_k', 'hybrid')
            iou_threshold (float): IoU threshold for positive selection (default: 0.25)
            top_k1 (int): Number of top proposals to consider for positives (default: 5)
            top_k2 (int): Number of nearest points to GT center for point-level positives (default: 32)
            weighting (str): Weighting scheme for positives ('uniform', 'iou', 'distance_gaussian')
            sigma (float): Standard deviation for distance-based Gaussian weighting (default: 1.0)
            symmetric (bool): Whether to use symmetric loss (text→3D + 3D→text) (default: False)
        """
        super().__init__()
        self.temperature = temperature
        self.positive_selection = positive_selection
        self.iou_threshold = iou_threshold
        self.top_k1 = top_k1
        self.top_k2 = top_k2
        self.weighting = weighting
        self.sigma = sigma
        self.symmetric = symmetric
        
    def compute_iou_3d(self, proposals, gt_boxes):
        """
        Compute 3D IoU between proposals and ground truth boxes.
        
        Args:
            proposals (torch.Tensor): Proposal boxes [N, 6] (cx, cy, cz, w, h, d)
            gt_boxes (torch.Tensor): GT boxes [B, 6] (cx, cy, cz, w, h, d)
            
        Returns:
            torch.Tensor: IoU matrix [B, N]
        """
        B, N = gt_boxes.shape[0], proposals.shape[0]
        
        # Convert center-size to min-max format
        def center_to_corners(boxes):
            centers = boxes[:, :3]  # [*, 3]
            sizes = boxes[:, 3:]    # [*, 3]
            min_coords = centers - sizes / 2
            max_coords = centers + sizes / 2
            return min_coords, max_coords
        
        gt_min, gt_max = center_to_corners(gt_boxes)  # [B, 3], [B, 3]
        prop_min, prop_max = center_to_corners(proposals)  # [N, 3], [N, 3]
        
        # Compute intersection
        inter_min = torch.max(gt_min.unsqueeze(1), prop_min.unsqueeze(0))  # [B, N, 3]
        inter_max = torch.min(gt_max.unsqueeze(1), prop_max.unsqueeze(0))  # [B, N, 3]
        
        inter_dims = torch.clamp(inter_max - inter_min, min=0)  # [B, N, 3]
        inter_volume = torch.prod(inter_dims, dim=2)  # [B, N]
        
        # Compute union
        gt_volume = torch.prod(gt_boxes[:, 3:], dim=1)  # [B]
        prop_volume = torch.prod(proposals[:, 3:], dim=1)  # [N]
        union_volume = gt_volume.unsqueeze(1) + prop_volume.unsqueeze(0) - inter_volume  # [B, N]
        
        # Compute IoU
        iou = inter_volume / (union_volume + 1e-8)
        return iou
    
    def select_positives(self, ious, proposal_scene_ids, gt_scene_ids, proposal_centers, gt_centers):
        """
        Select positive proposals for each text query based on IoU and scene matching.
        
        Args:
            ious (torch.Tensor): IoU matrix [B, N]
            proposal_scene_ids (torch.Tensor): Scene IDs for proposals [N]
            gt_scene_ids (torch.Tensor): Scene IDs for GT [B]
            proposal_centers (torch.Tensor): Proposal centers [N, 3]
            gt_centers (torch.Tensor): GT centers [B, 3]
            
        Returns:
            list: List of positive indices for each batch item
        """
        B, N = ious.shape
        positives = []
        
        for b in range(B):
            # Only consider proposals from the same scene
            same_scene_mask = (proposal_scene_ids == gt_scene_ids[b])
            same_scene_indices = torch.where(same_scene_mask)[0]
            
            if len(same_scene_indices) == 0:
                # Fallback: use all proposals if no scene match
                same_scene_indices = torch.arange(N, device=ious.device)
            
            scene_ious = ious[b, same_scene_indices]
            
            if self.positive_selection == 'iou_threshold':
                # Select proposals with IoU > threshold
                valid_mask = scene_ious > self.iou_threshold
                if valid_mask.sum() > 0:
                    pos_indices = same_scene_indices[valid_mask]
                else:
                    # Fallback: top-1 if no proposal exceeds threshold
                    top_idx = torch.argmax(scene_ious)
                    pos_indices = same_scene_indices[top_idx:top_idx+1]
                    
            elif self.positive_selection == 'top_k':
                # Select top-k proposals by IoU
                k = min(self.top_k1, len(same_scene_indices))
                _, top_indices = torch.topk(scene_ious, k)
                pos_indices = same_scene_indices[top_indices]
                
            elif self.positive_selection == 'hybrid':
                # Hybrid: top-k AND IoU > threshold
                valid_mask = scene_ious > self.iou_threshold
                if valid_mask.sum() > 0:
                    valid_indices = same_scene_indices[valid_mask]
                    valid_ious = scene_ious[valid_mask]
                    k = min(self.top_k1, len(valid_indices))
                    _, top_indices = torch.topk(valid_ious, k)
                    pos_indices = valid_indices[top_indices]
                else:
                    # Fallback: top-k proposals
                    k = min(self.top_k1, len(same_scene_indices))
                    _, top_indices = torch.topk(scene_ious, k)
                    pos_indices = same_scene_indices[top_indices]
            
            positives.append(pos_indices)
        
        return positives
    
    def compute_positive_weights(self, positive_indices, ious, proposal_centers, gt_centers):
        """
        Compute weights for positive proposals.
        
        Args:
            positive_indices (list): List of positive indices for each batch item
            ious (torch.Tensor): IoU matrix [B, N]
            proposal_centers (torch.Tensor): Proposal centers [N, 3]
            gt_centers (torch.Tensor): GT centers [B, 3]
            
        Returns:
            list: List of weights for each batch item's positives
        """
        B = len(positive_indices)
        weights = []
        
        for b in range(B):
            pos_indices = positive_indices[b]
            
            if self.weighting == 'uniform':
                # Uniform weights
                w = torch.ones(len(pos_indices), device=ious.device) / len(pos_indices)
                
            elif self.weighting == 'iou':
                # IoU-weighted
                pos_ious = ious[b, pos_indices]
                w = pos_ious / (pos_ious.sum() + 1e-8)
                
            elif self.weighting == 'distance_gaussian':
                # Distance-based Gaussian weighting
                gt_center = gt_centers[b:b+1]  # [1, 3]
                pos_centers = proposal_centers[pos_indices]  # [P, 3]
                distances = torch.norm(pos_centers - gt_center, dim=1)  # [P]
                w = torch.exp(-distances ** 2 / (2 * self.sigma ** 2))
                w = w / (w.sum() + 1e-8)
                
            weights.append(w)
        
        return weights
    
    def forward(self, text_embeddings, proposal_embeddings, gt_boxes, proposal_boxes, 
                text_scene_ids, proposal_scene_ids, proposal_centers=None):
        """
        Compute contrastive loss between text and 3D proposals.
        
        Args:
            text_embeddings (torch.Tensor): L2-normalized text embeddings [B, D]
            proposal_embeddings (torch.Tensor): L2-normalized proposal embeddings [N, D]
            gt_boxes (torch.Tensor): Ground truth boxes [B, 6]
            proposal_boxes (torch.Tensor): Proposal boxes [N, 6]
            text_scene_ids (torch.Tensor): Scene IDs for text queries [B]
            proposal_scene_ids (torch.Tensor): Scene IDs for proposals [N]
            proposal_centers (torch.Tensor, optional): Proposal centers [N, 3]. If None, extracted from boxes.
            
        Returns:
            torch.Tensor: Contrastive loss scalar
            dict: Additional information (logits, positive masks, etc.)
        """
        B, D = text_embeddings.shape
        N = proposal_embeddings.shape[0]
        device = text_embeddings.device
        
        # Extract centers if not provided
        if proposal_centers is None:
            proposal_centers = proposal_boxes[:, :3]
        gt_centers = gt_boxes[:, :3]
        
        # Compute IoU between proposals and GT
        ious = self.compute_iou_3d(proposal_boxes, gt_boxes)  # [B, N]
        
        # Select positive proposals
        positive_indices = self.select_positives(
            ious, proposal_scene_ids, text_scene_ids, proposal_centers, gt_centers
        )
        
        # Compute positive weights
        positive_weights = self.compute_positive_weights(
            positive_indices, ious, proposal_centers, gt_centers
        )
        
        # Compute logits: text-to-3D similarities
        logits = torch.matmul(text_embeddings, proposal_embeddings.T) / self.temperature  # [B, N]
        
        # Compute text→3D loss
        text_to_3d_losses = []
        for b in range(B):
            pos_indices = positive_indices[b]
            pos_weights = positive_weights[b]
            
            if len(pos_indices) == 0:
                continue
                
            # Positive logits (numerator)
            pos_logits = logits[b, pos_indices]  # [P]
            weighted_pos_exp = (pos_weights * torch.exp(pos_logits)).sum()
            
            # All logits (denominator)
            all_exp = torch.exp(logits[b]).sum()  # [N] -> scalar
            
            # InfoNCE loss
            loss_b = -torch.log(weighted_pos_exp / (all_exp + 1e-8))
            text_to_3d_losses.append(loss_b)
        
        if len(text_to_3d_losses) == 0:
            text_to_3d_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            text_to_3d_loss = torch.stack(text_to_3d_losses).mean()
        
        total_loss = text_to_3d_loss
        
        # Optional: Symmetric 3D→text loss
        if self.symmetric:
            # For each proposal, find which texts it's positive for
            proposal_to_text_losses = []
            for i in range(N):
                # Find texts where this proposal is positive
                positive_texts = []
                text_weights = []
                
                for b in range(B):
                    if i in positive_indices[b]:
                        pos_idx_in_list = (positive_indices[b] == i).nonzero(as_tuple=True)[0]
                        if len(pos_idx_in_list) > 0:
                            positive_texts.append(b)
                            text_weights.append(positive_weights[b][pos_idx_in_list[0]])
                
                if len(positive_texts) == 0:
                    continue
                
                positive_texts = torch.tensor(positive_texts, device=device)
                text_weights = torch.stack(text_weights)
                text_weights = text_weights / (text_weights.sum() + 1e-8)
                
                # Positive logits (numerator)
                pos_logits = logits[positive_texts, i]  # [P']
                weighted_pos_exp = (text_weights * torch.exp(pos_logits)).sum()
                
                # All logits (denominator) - all texts for this proposal
                all_exp = torch.exp(logits[:, i]).sum()
                
                # InfoNCE loss
                loss_i = -torch.log(weighted_pos_exp / (all_exp + 1e-8))
                proposal_to_text_losses.append(loss_i)
            
            if len(proposal_to_text_losses) > 0:
                proposal_to_text_loss = torch.stack(proposal_to_text_losses).mean()
                total_loss = total_loss + proposal_to_text_loss
        
        # Additional info for monitoring
        info = {
            'logits': logits,
            'positive_indices': positive_indices,
            'positive_weights': positive_weights,
            'ious': ious,
            'text_to_3d_loss': text_to_3d_loss,
            'num_positives': [len(pos) for pos in positive_indices]
        }
        
        return total_loss, info


def create_contrastive_loss(**kwargs):
    """Factory function to create ContrastiveLoss with default parameters."""
    return ContrastiveLoss(**kwargs)