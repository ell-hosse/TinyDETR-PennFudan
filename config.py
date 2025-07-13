import torch

# Hyperparameters & paths
DATA_ROOT   = "data/PennFudanPed"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 2
LR          = 1e-5
EPOCHS      = 10
