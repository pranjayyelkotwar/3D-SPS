"""
Example usage of contrastive learning in 3D-SPS.

This example demonstrates how to use the Projector3D modules and ContrastiveLoss 
to map object features and language features into a similar embedding space.
"""

import torch
from models.vgnet import VGNet
from losses.contrastive_loss import ContrastiveLoss


def example_contrastive_training_step(model, data_dict, loss_config):
    """
    Example training step showing how to integrate contrastive loss.
    
    Args:
        model: VGNet model with projectors
        data_dict: Input data dictionary
        loss_config: Configuration for contrastive loss
    
    Returns:
        dict: Dictionary containing losses and additional info
    """
    
    # Forward pass through VGNet
    data_dict = model(data_dict)
    
    # Extract projected features (added by the updated VGNet)
    projected_object_feat = data_dict['projected_object_feat']  # [B, n_proposal, 512]
    projected_lang_feat = data_dict['projected_lang_feat']      # [B, n_word, 512]
    
    # For contrastive loss, we need to aggregate language features per query
    # Typically this would be mean pooling over words (excluding padding)
    lang_mask = data_dict['lang_mask']  # [B, n_word]
    
    # Mean pool language features (handle padding with mask)
    lang_lengths = lang_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
    aggregated_lang_feat = (projected_lang_feat * lang_mask.unsqueeze(-1)).sum(dim=1) / (lang_lengths + 1e-8)  # [B, 512]
    
    # Reshape object features: [B, n_proposal, 512] -> [B*n_proposal, 512]
    B, n_proposal, feat_dim = projected_object_feat.shape
    flattened_object_feat = projected_object_feat.view(-1, feat_dim)  # [B*n_proposal, 512]
    
    # Create scene IDs and proposal info
    batch_indices = torch.arange(B, device=projected_object_feat.device)
    text_scene_ids = batch_indices  # [B]
    proposal_scene_ids = batch_indices.unsqueeze(1).repeat(1, n_proposal).view(-1)  # [B*n_proposal]
    
    # Extract ground truth and proposal boxes (assuming they're in data_dict)
    # These should be in format [cx, cy, cz, w, h, d]
    gt_boxes = data_dict['center_label']  # [B, 3] centers
    gt_sizes = data_dict['size_gts']      # [B, 3] sizes  
    gt_boxes_full = torch.cat([gt_boxes, gt_sizes], dim=-1)  # [B, 6]
    
    # Proposal boxes - assuming they come from the proposal head
    proposal_centers = data_dict['last_center']  # [B, n_proposal, 3]
    proposal_sizes = data_dict['last_size_residuals']  # [B, n_proposal, 3] 
    proposal_boxes = torch.cat([proposal_centers, proposal_sizes], dim=-1)  # [B, n_proposal, 6]
    proposal_boxes_flat = proposal_boxes.view(-1, 6)  # [B*n_proposal, 6]
    
    # Initialize contrastive loss
    contrastive_loss = ContrastiveLoss(
        temperature=loss_config.get('temperature', 0.07),
        positive_selection=loss_config.get('positive_selection', 'hybrid'),
        iou_threshold=loss_config.get('iou_threshold', 0.25),
        top_k1=loss_config.get('top_k1', 5),
        weighting=loss_config.get('weighting', 'iou'),
        symmetric=loss_config.get('symmetric', False)
    )
    
    # Compute contrastive loss
    contrastive_loss_value, contrastive_info = contrastive_loss(
        text_embeddings=aggregated_lang_feat,           # [B, 512]
        proposal_embeddings=flattened_object_feat,      # [B*n_proposal, 512]
        gt_boxes=gt_boxes_full,                         # [B, 6]
        proposal_boxes=proposal_boxes_flat,             # [B*n_proposal, 6]
        text_scene_ids=text_scene_ids,                  # [B]
        proposal_scene_ids=proposal_scene_ids           # [B*n_proposal]
    )
    
    return {
        'contrastive_loss': contrastive_loss_value,
        'contrastive_info': contrastive_info,
        'projected_object_feat': projected_object_feat,
        'projected_lang_feat': projected_lang_feat,
        'aggregated_lang_feat': aggregated_lang_feat
    }


def example_usage():
    """
    Example of how to use the contrastive loss in training.
    """
    
    # Example loss configuration
    loss_config = {
        'temperature': 0.07,
        'positive_selection': 'hybrid',  # 'iou_threshold', 'top_k', or 'hybrid'
        'iou_threshold': 0.25,
        'top_k1': 5,
        'weighting': 'iou',  # 'uniform', 'iou', or 'distance_gaussian'
        'symmetric': False
    }
    
    print("Contrastive Loss Configuration:")
    print(f"- Temperature: {loss_config['temperature']}")
    print(f"- Positive selection: {loss_config['positive_selection']}")
    print(f"- IoU threshold: {loss_config['iou_threshold']}")
    print(f"- Top-k proposals: {loss_config['top_k1']}")
    print(f"- Weighting scheme: {loss_config['weighting']}")
    print(f"- Symmetric loss: {loss_config['symmetric']}")
    
    print("\nIn your training loop, you would:")
    print("1. Forward pass through VGNet (now includes projectors)")
    print("2. Extract projected object and language features")
    print("3. Aggregate language features (e.g., mean pooling)")
    print("4. Compute contrastive loss using IoU-based positive selection")
    print("5. Combine with existing losses (detection, reference, etc.)")
    
    print("\nKey benefits:")
    print("- Maps 3D objects and text to similar embedding space")
    print("- Uses IoU-based positive/negative sampling")
    print("- Supports multiple weighting schemes")
    print("- Compatible with existing 3D-SPS architecture")


if __name__ == "__main__":
    example_usage()