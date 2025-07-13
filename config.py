from pathlib import Path

# Paths
DATA_ROOT = Path("data/PennFudanPed")
CKPT_DIR  = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# Training
NUM_CLASSES      = 2
EPOCHS           = 30
BATCH_SIZE       = 2
LR               = 1e-4
LR_BACKBONE      = 1e-5
WEIGHT_DECAY     = 1e-4
LR_DROP_EPOCHS   = 20
PRINT_FREQ       = 20
NUM_WORKERS      = 4
CONF_THRESHOLD   = 0.7
