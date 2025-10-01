# 3D-SPS Contrastive Loss and Wandb Integration

## Overview
This document describes the integration of contrastive loss and Weights & Biases (wandb) logging into the 3D-SPS (3D Spatial-Proposal-based) visual grounding framework.

## Changes Made

### 1. Contrastive Loss Integration

#### File Moved
- **Source**: `losses/contrastive_loss.py`
- **Destination**: `lib/contrastive_loss.py`
- **Reason**: Better organization within the lib directory alongside other core components

#### Loss Integration (`lib/loss_helper.py`)
- Added import: `from lib.contrastive_loss import ContrastiveLoss`
- Added `compute_contrastive_loss()` function with comprehensive parameter handling
- Integrated contrastive loss into main `get_loss()` function
- Added configurable weight: `args.contrastive_loss_weight` (default: 0.1)

#### Key Features of Contrastive Loss
- **Multi-positive InfoNCE**: Supports multiple positive proposals per text query
- **Flexible Positive Selection**: 
  - `'iou_threshold'`: IoU-based selection
  - `'top_k'`: Top-k proposals by IoU
  - `'hybrid'`: Combined approach
- **Weighting Schemes**:
  - `'uniform'`: Equal weights for all positives
  - `'iou'`: IoU-based weighting
  - `'distance_gaussian'`: Distance-based Gaussian weighting
- **Symmetric Loss**: Optional bidirectional text↔3D contrastive learning

### 2. Wandb Integration (`lib/solver.py`)

#### New Constructor Parameters
```python
def __init__(self, ..., 
    use_wandb=True,           # Enable/disable wandb logging
    wandb_project=None,       # Project name (default: "3d-sps-training")
    wandb_run_name=None,      # Run name (default: f"run_{stamp}")
    wandb_config=None         # Additional config dict
):
```

#### Features Added
- **Automatic Initialization**: Initializes wandb on main process only (distributed training safe)
- **Model Watching**: Tracks gradients and parameters (`wandb.watch`)
- **Comprehensive Logging**: All training/validation metrics logged to wandb
- **Best Model Tracking**: Logs best metrics when new best model is found
- **Graceful Fallback**: Works without wandb installed (prints warning)

#### Logging Structure
- **Training Metrics**: Logged every iteration step
- **Validation Metrics**: Logged every epoch
- **Learning Rate**: Tracked automatically
- **Best Metrics**: Special logging when best model is achieved

### 3. Enhanced Reporting Templates

All reporting templates updated to include contrastive loss:
- `ITER_REPORT_TEMPLATE`: Training iteration reports
- `EPOCH_REPORT_TEMPLATE`: Epoch summary reports  
- `BEST_REPORT_TEMPLATE`: Best model reports

### 4. Updated Training Script (`scripts/train.py`)

Modified Solver instantiation to include wandb parameters:
```python
solver = Solver(
    ...,
    use_wandb=getattr(args, 'use_wandb', True),
    wandb_project=getattr(args, 'wandb_project', None),
    wandb_run_name=getattr(args, 'wandb_run_name', None),
    wandb_config=getattr(args, 'wandb_config', None)
)
```

## Usage

### Basic Usage with Contrastive Loss
```bash
python scripts/train.py \
    --use_contrastive_loss \
    --contrastive_loss_weight 0.1 \
    --contrastive_temperature 0.07 \
    --contrastive_positive_selection iou_threshold \
    --contrastive_iou_threshold 0.25
```

### Wandb Configuration
```bash
python scripts/train.py \
    --use_wandb \
    --wandb_project "my-3d-grounding-project" \
    --wandb_run_name "contrastive_experiment_1"
```

### Disable Wandb
```bash
python scripts/train.py --no_wandb  # or set use_wandb=False
```

### Advanced Contrastive Loss Settings
```bash
python scripts/train.py \
    --use_contrastive_loss \
    --contrastive_temperature 0.05 \
    --contrastive_positive_selection hybrid \
    --contrastive_iou_threshold 0.3 \
    --contrastive_top_k1 3 \
    --contrastive_weighting iou \
    --contrastive_symmetric
```

## Configuration Parameters

### Contrastive Loss Parameters
- `use_contrastive_loss`: Enable contrastive loss (default: False)
- `contrastive_loss_weight`: Loss weight (default: 0.1)
- `contrastive_temperature`: Temperature parameter τ (default: 0.07)
- `contrastive_positive_selection`: Selection strategy (default: 'iou_threshold')
- `contrastive_iou_threshold`: IoU threshold for positives (default: 0.25)
- `contrastive_top_k1`: Top-k proposals (default: 5)
- `contrastive_top_k2`: Top-k points (default: 32)
- `contrastive_weighting`: Weighting scheme (default: 'uniform')
- `contrastive_sigma`: Gaussian weighting σ (default: 1.0)
- `contrastive_symmetric`: Bidirectional loss (default: False)

### Wandb Parameters
- `use_wandb`: Enable wandb logging (default: True)
- `wandb_project`: Project name (default: "3d-sps-training")
- `wandb_run_name`: Run name (default: f"run_{timestamp}")
- `wandb_config`: Additional config dictionary

## Technical Details

### Data Flow
1. **Model Forward**: Generates predictions including embeddings
2. **Loss Computation**: `get_loss()` orchestrates all losses including contrastive
3. **Contrastive Loss**: Computes InfoNCE loss between text and 3D embeddings
4. **Logging**: Both tensorboard and wandb receive metrics
5. **Backward**: Combined loss drives gradient computation

### Expected Data Dictionary Keys
For contrastive loss to work, the model should provide:
- `text_embeddings` or `{prefix}text_embeddings`: Text embeddings [B, D]
- `{prefix}proposal_embeddings` or `{prefix}aggregated_vote_features`: Proposal embeddings
- Standard 3D detection outputs (centers, sizes, etc.)

### Error Handling
- Graceful fallback when required data is missing
- Warning messages for debugging
- Distributed training compatibility
- No-op behavior when wandb unavailable

## Benefits

### Contrastive Loss
- **Better Text-3D Alignment**: Learns joint embedding space
- **Multi-positive Learning**: Handles multiple valid groundings
- **Flexible Training**: Configurable positive selection and weighting
- **Improved Grounding**: Better text-to-3D correspondence

### Wandb Integration
- **Experiment Tracking**: Comprehensive metric logging
- **Visualization**: Rich plots and dashboards
- **Reproducibility**: Automatic config and code logging
- **Collaboration**: Easy sharing and comparison
- **Model Management**: Track best models and hyperparameters

## Installation Requirements

### For Contrastive Loss
No additional requirements - uses existing PyTorch functionality.

### For Wandb Integration
```bash
pip install wandb
wandb login  # First time setup
```

If wandb is not installed, the system will print a warning and continue without wandb logging.

## Backward Compatibility

All changes are backward compatible:
- Contrastive loss is disabled by default
- Wandb is enabled by default but gracefully handles missing installation
- Existing training scripts will work without modification
- All original functionality preserved

## Future Enhancements

Potential improvements:
- Dynamic contrastive loss weighting based on training progress
- Additional contrastive loss variants (supervised, unsupervised)
- Advanced wandb features (artifacts, sweeps)
- Integration with other experiment tracking tools
- Custom evaluation metrics for contrastive learning