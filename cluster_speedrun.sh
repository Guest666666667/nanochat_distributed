#!/bin/bash
#SBATCH --job-name=nanochat_speedrun
#SBATCH --time=12:00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/nanochat-%N-%j.out
#SBATCH --mem=0
#SBATCH --nodelist=node5    ###node5,node6

# slurm
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

echo "Host="$(hostname)
echo "NODELIST="${SLURM_NODELIST}
echo "MASTER="${MASTER_ADDR}":"${MASTER_PORT}
echo "SLURM_NNODES="${SLURM_NNODES}
echo "SLURM_NTASKS="${SLURM_NTASKS}

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

# -----------------------------------------------------------------------------
# Base model pretraining (使用srun启动分布式训练)
#srun python -m scripts.base_train -- --depth=1 --device_batch_size=1 --num_iterations=3 --run=$WANDB_RUN
srun torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- --depth=1 --device_batch_size=1 --num_iterations=3 --run=$WANDB_RUN


# 生成报告 (只在主节点执行)
python3 -m nanochat.report generate
echo "Training Done!"
