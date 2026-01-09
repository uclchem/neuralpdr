# V1 Configuration Files (Legacy)

These configurations are from the original NeuralPDR paper and use the V1 dataset.

## Backward Compatibility ✅

**V1 configs are automatically adapted to work with the current training infrastructure!**

The parser detects V1 format (using `depth` and `width` parameters) and automatically converts
them to the current encoder-decoder architecture. This means you can run V1 paper configs
directly without manual modification:

```bash
python src/neuralpdr/train.py configs/v1/paper_archived/mlps_model_1.yaml
```

**V1 → Current Mapping:**
- `depth` and `width` → split into `enc_dec_depth`, `latent_depth`, `enc_dec_width`, `latent_width`
- Simple learning rate scheme → converted to multi-stage `learning_schemes`
- Missing parameters → sensible defaults added (splits, bottleneck, activation)

## Status

**Legacy/Archived** - These configs use a different model architecture than current V2/V3 configs,
but are automatically adapted at runtime for backward compatibility.

## Architecture Differences

V1 configs use a simpler architecture with `depth`/`width` parameters that are automatically converted to the current encoder-decoder format at runtime.

## Files

### paper_archived/
Contains all original configurations from the published paper:
- `mlps_model_[1-4].yaml` - Numbered model variants
- `mlps_model_[a-f].yaml` - Lettered model variants  
- `mlps_model_linear.yaml` - Linear baseline model

### features/
- `input_features.yaml` - Feature definitions for V1 dataset
