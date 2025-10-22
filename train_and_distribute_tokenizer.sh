#!/bin/bash
# 手动在主节点执行的 tokenizer 训练和分发脚本
# 使用方法: bash train_and_distribute_tokenizer.sh

set -e  # 遇到错误立即退出

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

echo "=========================================="
echo "Tokenizer Training and Distribution"
echo "=========================================="
echo "Current Node: $(hostname)"
echo "Base Directory: $NANOCHAT_BASE_DIR"
echo "=========================================="

# 激活虚拟环境
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found. Please run cluster_prepare.sh first."
    exit 1
fi

source .venv/bin/activate

# 验证数据已下载
if [ ! -d "$NANOCHAT_BASE_DIR/base_data" ]; then
    echo "ERROR: Training data not found. Please run cluster_prepare.sh first."
    exit 1
fi

# 训练 tokenizer
echo ""
echo "=========================================="
echo "Step 1: Training tokenizer..."
echo "=========================================="
python3 -m scripts.tok_train --max_chars=5000000

if [ $? -ne 0 ]; then
    echo "ERROR: Tokenizer training failed"
    exit 1
fi

# 评估 tokenizer
echo ""
echo "=========================================="
echo "Step 2: Evaluating tokenizer..."
echo "=========================================="
python3 -m scripts.tok_eval

echo ""
echo "Tokenizer training completed on $(hostname)"

# 分发到其他节点
echo ""
echo "=========================================="
echo "Step 3: Distributing tokenizer to other nodes..."
echo "=========================================="
TARGET_NODES=""  #node2 node3 node4
echo "Distributing to: $TARGET_NODES"
for node in $TARGET_NODES; do
    echo "Copying to $node..."
    scp -r $NANOCHAT_BASE_DIR/tok_checkpoints/ $node:$NANOCHAT_BASE_DIR/
done

echo ""
echo "=========================================="
echo "Tokenizer Setup Complete! location: $NANOCHAT_BASE_DIR/tok_checkpoints/tok.model"
echo "=========================================="