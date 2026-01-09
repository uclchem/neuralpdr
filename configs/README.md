# Configuration Files

This directory contains configuration files for training NeuralPDR models across different dataset versions.

## Quick Start

**Training a model:**
```bash
python src/neuralpdr/train.py configs/v3/base.yaml
```

**Running inference:**
```bash
python src/neuralpdr/inference.py --dataset_path PATH --model_dir RESULTS_DIR --weights_file WEIGHTS
```
### Feature & Normalization Files

**Feature files** (e.g., `v3/features/input_features.yaml`) define which chemical species and auxiliary parameters to use as inputs and outputs.

**Normalization files** (e.g., `v3/features/normalisations/default.yaml`) contain pre-computed statistics for data normalization. If not specified, normalization is computed from training data.

## Required Configuration Fields

### Data Configuration
```yaml
dataset_path: "path/to/dataset.h5"          # HDF5 dataset file
input_features_file: "configs/input_features_v3.yaml"
normalisations_file: "configs/normalisations_try.yaml"  # Optional
start_index: 0                              # Dataset slice start
end_index: -1                               # Dataset slice end (-1 = all)
minimal_timeseries_length: 48               # Minimum sequence length
aux_features: True                          # Use auxiliary features
train_split: 0.7                            # Training set fraction
val_split: 0.15                             # Validation set fraction
test_split: 0.15                            # Test set fraction
training_batch_subsampling: 1.0             # Batch subsampling ratio
```

### Model Configuration
```yaml
batch_size: 32                              # Training batch size
enc_dec_depth: 3                            # Encoder/decoder layers
enc_dec_width: 512                          # Encoder/decoder hidden size
latent_depth: 3                             # Latent ODE layers
latent_width: 512                           # Latent ODE hidden size
latent_bottleneck: 128                      # Latent space dimension
weight_scale: 1.0                           # Weight initialization scale
weight_truncation: 10.0                     # Weight initialization truncation
latent_final_activation: "tanh"             # Final activation function
```

### Training Configuration
```yaml
shuffle_every_n_epochs: 1                   # Data shuffling frequency
epochs_per_partial_timeseries: 200          # Epochs per stage
weight_decay: 1.0E-4                        # L2 regularization

learning_schemes:                           # Multi-stage training schedule
  - timeseries_fraction: 32                 # Use first 32 timesteps
    epochs: 5                               # Train for 5 epochs
    lr_scheduler: "sgdr"                    # Learning rate scheduler
    warmup_epochs: 2                        # Warmup period
    learning_rate: 2.0E-4                   # Learning rate
  - timeseries_fraction: 1.0                # Use full sequence
    epochs: 60
    lr_scheduler: "sgdr"
    warmup_epochs: 5
    learning_rate: 2.0E-4
```

### Output Configuration
```yaml
save_file_path: "results/experiment_name"   # Output directory
neptune_project: "workspace/project-name"   # Neptune logging project
```

### Optional Fields
```yaml
checkpoint_file: "results/checkpoints/weights_epoch_20.eqx"
checkpoint_epoch: 20
```

## File Naming Conventions

- `vX/base.yaml` - Base configuration for version X
- `vX/experiments/{description}.yaml` - Variant configuration
- Clear descriptive names (no abbreviations):
  - `bottleneck_8.yaml` (not `bn_8.yaml`)
  - `batch_size_64.yaml` (not `bs_64.yaml`)

## Common Patterns

### Testing with Reduced Data

For quick testing, modify:
```yaml
start_index: 0
end_index: 50                # Use only first 50 models
minimal_timeseries_length: 32  # Shorter sequences
learning_schemes:
  - timeseries_fraction: 32
    epochs: 2                # Fewer epochs
```

### Hyperparameter Experiments

When testing different hyperparameters:
1. Copy base config (e.g., `v3/base.yaml`) to `v3/experiments/`
2. Modify specific parameter(s)
3. Update `save_file_path` to reflect the experiment
4. Save with descriptive name (e.g., `bottleneck_64.yaml`)

### Path Management

**Relative paths** (recommended for config files):
```yaml
input_features_file: "configs/v3/features/input_features.yaml"
```

**Absolute paths or environment variables** (recommended for datasets):
```yaml
dataset_path: "${DATA_ROOT}/v3/3dpdr_dataset_v3.h5"
save_file_path: "~/results/neuralpdr/v3"
```

Set environment variables in your shell:
```bash
export DATA_ROOT="/path/to/data"
python src/neuralpdr/train.py configs/v3/base.yaml
```

## Troubleshooting

### Missing normalisations_file

If not specified, normalization parameters will be computed from the training data. This is slower but works for new datasets.

