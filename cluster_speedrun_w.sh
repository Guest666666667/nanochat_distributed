#!/bin/bash

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "ERROR: Tokenizer not found. Run train_and_distribute_tokenizer.sh first."
    exit 1
fi

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt" ]; then
    echo "ERROR: Token bytes mapping not found. Run train_and_distribute_tokenizer.sh first."
    exit 1
fi

if [ ! -d "$NANOCHAT_BASE_DIR/base_data" ]; then
    echo "ERROR: Training data not found. Run cluster_prepare.sh first."
    exit 1
fi

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

source .venv/bin/activate
python -m nanochat.report reset

torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=1 --device_batch_size=1 --num_iterations=3 --run=$WANDB_RUN

python -m nanochat.report generate
echo "Training Done!"
