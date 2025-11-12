# Hyperparameter Tuning Summary Report

**Generated:** 2025-11-12 21:38:35

## Overall Statistics

- Total experiments: 4
- Successful: 4
- Success rate: 100.0%

## Validation Loss Statistics

- Mean: 0.552772
- Std: 0.032611
- Min: 0.534865
- Max: 0.601653
- Median: 0.537285

## Top 10 Experiments

|   Rank | Experiment   |   Hidden Size |   Num Layers |   Dropout |   Learning Rate |   Batch Size |   best_val_loss |   best_train_loss |   best_val_correlation |
|-------:|:-------------|--------------:|-------------:|----------:|----------------:|-------------:|----------------:|------------------:|-----------------------:|
|      1 | grid_0002    |            32 |            2 |       0.1 |           0.005 |            8 |        0.534865 |          0.579078 |               0.483923 |
|      2 | grid_0004    |            64 |            2 |       0.1 |           0.005 |            8 |        0.536673 |          0.59761  |               0.485413 |
|      3 | grid_0003    |            64 |            2 |       0.1 |           0.001 |            8 |        0.537897 |          0.597125 |               0.483662 |
|      4 | grid_0001    |            32 |            2 |       0.1 |           0.001 |            8 |        0.601653 |          0.668892 |               0.460075 |

## Best Experiment Details

**Experiment:** grid_0002

**Configuration:**
- Hidden Size: 32
- Num Layers: 2
- Dropout: 0.1
- Learning Rate: 0.005
- Batch Size: 8
- Optimizer: adam

**Performance:**
- final_train_loss: 0.579078
- final_val_loss: 0.537901
- best_train_loss: 0.579078
- best_val_loss: 0.534865
- best_val_correlation: 0.483923
- best_val_r2: 0.234038
- best_val_snr: 1.157927

