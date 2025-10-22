#!/bin/bash  
  
# 针对2×4080 GPU的快速训练配置  
export OMP_NUM_THREADS=1  
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"  
mkdir -p $NANOCHAT_BASE_DIR  
  
# -----------------------------------------------------------------------------  
# Python环境设置  
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh  
[ -d ".venv" ] || uv venv  
uv sync  
source .venv/bin/activate  
# -----------------------------------------------------------------------------  
# wandb设置（可选）  
if [ -z "$WANDB_RUN" ]; then  
    WANDB_RUN=dummy  
fi  
  
# -----------------------------------------------------------------------------  
# 初始化报告  
python3 -m nanochat.report reset  
  
# -----------------------------------------------------------------------------  
# Tokenizer构建  
if command -v rustc >/dev/null 2>&1; then
    echo "Skip install rustup"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
fi
source "$HOME/.cargo/env"  
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml  

# 下载20个数据分片
python3 -m nanochat.dataset -n 20 
echo "Download nanochat done"
  
# 训练tokenizer（使用较少数据）  
python3 -m scripts.tok_train --max_chars=2000000000  
python3 -m scripts.tok_eval  
  
# -----------------------------------------------------------------------------  
# 下载评估数据  
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip  
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then  
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL  
    unzip -q eval_bundle.zip  
    rm eval_bundle.zip  
    mv eval_bundle $NANOCHAT_BASE_DIR  
fi  
echo "Download EVAL_BUNDLE_URL done"
  
# -----------------------------------------------------------------------------  
# 预训练
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- --depth=16 --device_batch_size=2 --num_iterations=6000 --run=$WANDB_RUN 
torchrun --standalone --nproc_per_node=2 -m scripts.base_loss -- --device_batch_size=2 
torchrun --standalone --nproc_per_node=2 -m scripts.base_eval  
  
# -----------------------------------------------------------------------------  
# 中期训练   
torchrun --standalone --nproc_per_node=2 -m scripts.mid_train -- --device_batch_size=2 --run=$WANDB_RUN   
torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval -- -i mid
  
# -----------------------------------------------------------------------------  
# 监督微调  
torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft -- --device_batch_size=2 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval -- -i sft
  
# -----------------------------------------------------------------------------  
# 生成报告  
python3 -m nanochat.report generate  
cp $NANOCHAT_BASE_DIR/report/report.md .  
  
echo "Training Done!"
