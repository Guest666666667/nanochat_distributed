#!/bin/bash
#SBATCH --job-name=nanochat_deepspeed
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --gpus=4
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/nanochat-%N-%j.out
#SBATCH --mem=0
#SBATCH --nodelist=node4,node5

# 环境变量设置
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

echo "Host="$(hostname)
echo "NODELIST="${SLURM_NODELIST}
echo "SLURM_NNODES="${SLURM_NNODES}
echo "SLURM_NTASKS="${SLURM_NTASKS}

# 检查必要文件
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
python3 -m nanochat.report reset

deepspeed --launcher=slurm \
    --num_nodes=2 \
    --num_gpus=2 \
    scripts/base_train_DP.py \
    --deepspeed \
    --deepspeed_config ds_config.json \
    --depth=1 \
    --device_batch_size=1 \
    --num_iterations=3 \
    --run=$WANDB_RUN

# 生成报告
python3 -m nanochat.report generate
echo "Training Done!"