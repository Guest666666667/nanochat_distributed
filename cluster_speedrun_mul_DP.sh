#!/bin/bash
#SBATCH --job-name=nanochat_deepspeed
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/nanochat-%N-%j.out
#SBATCH --mem=0
#SBATCH --nodelist=node4,node5

# 环境变量设置
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NUM_NODES=2
export NUM_GPUS=4

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

HOSTFILE="/tmp/deepspeed_hostfile_${SLURM_JOB_ID}"
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read node; do
    echo "${node} slots=${SLURM_GPUS_PER_NODE}" >> $HOSTFILE
done

echo "Generated hostfile:"
cat $HOSTFILE

deepspeed  --num_nodes $NUM_NODES \
    --num_gpus $NUM_GPUS \
    --hostfile $HOSTFILE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --launcher SLURM \
    scripts/base_train_DP.py \
    --deepspeed_config=ds_config.json \
    --depth=1 \
    --device_batch_size=1 \
    --num_iterations=3 \
    --run=$WANDB_RUN

#deepspeed --launcher=slurm \
#    --hostfile $HOSTFILE \
#    --no_ssh_check \
#    scripts/base_train_DP.py \
#    --deepspeed_config=ds_config.json \
#    --depth=1 \
#    --device_batch_size=1 \
#    --num_iterations=3 \
#    --run=$WANDB_RUN

# 生成报告
python3 -m nanochat.report generate
echo "Training Done!"