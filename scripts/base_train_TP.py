"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys

venv_path = '/home/user/Desktop/nanochat_distributed/.venv/lib/python3.10/site-packages'
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)
import time
from contextlib import nullcontext

import json
import wandb
import torch
import deepspeed

from nanochat.gpt_TP import GPT, GPTConfig, Block
from nanochat.dataloader_TP import tokenizing_distributed_data_loader
from nanochat.common_deepspeed import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, \
    autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
# from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
from deepspeed.module_inject.tp_shard import (
    set_num_kv_heads,
    set_n_embd,
    set_num_attention_heads,
    set_tp_grain_size
)

print_banner()

# -----------------------------------------------------------------------------
deepspeed_config = "ds_config_TP.json"
# User settings
run = "dummy"  # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = ""  # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = 20  # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 2048  # max context length
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1  # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0  # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20  # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 32  # per-device batch size (set to not OOM)
total_batch_size = 524288  # total desired batch size, in #tokens
embedding_lr = 0.2  # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004  # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0  # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02  # learning rate for the matrix parameters (Muon)
grad_clip = 1.0  # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.0  # ratio of iterations for LR warmup
warmdown_ratio = 0.2  # ratio of iterations for LR warmdown
final_lr_frac = 0.0  # final LR is this fraction of the initial LR
# Evaluation
eval_every = 250  # every how many steps to evaluate the model for val bpb
eval_tokens = 20 * 524288  # number of tokens to evaluate val loss on
core_metric_every = 2000  # every how many steps to evaluate the core metric (-1 = disable)
core_metric_max_per_task = 500  # examples per task in estimating the core metric
sample_every = 2000  # every how many steps to sample from the model
# Output
model_tag = ""  # optionally override the model tag for the output checkpoint directory name
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())  # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
tp_size = 2
dp_size = ddp_world_size // tp_size
tp_rank = ddp_rank % tp_size
dp_rank = ddp_rank // tp_size
tp_group_ranks = [list(range(i * tp_size, (i + 1) * tp_size)) for i in range(dp_size)]
tp_groups = [torch.distributed.new_group(ranks) for ranks in tp_group_ranks]
tp_group = tp_groups[dp_rank]
dp_group_ranks = [list(range(i, ddp_world_size, tp_size)) for i in range(tp_size)]
dp_groups = [torch.distributed.new_group(ranks) for ranks in dp_group_ranks]
dp_group = dp_groups[tp_rank]
print(f"ddp_rank:{ddp_rank}, ddp_local_rank: {ddp_local_rank}, ddp_world_size: {ddp_world_size}, "
      f"TP rank: {tp_rank}/{tp_size}, DP rank: {dp_rank}/{dp_size}")
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type,
                                  dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
# num_heads = max(1, (model_dim + 127) // 128)  # head dim 128 (the division here is ceil div)
num_heads = tp_size
num_kv_heads = num_heads  # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * dp_size  # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
# -----------------------------------------------------------------------------
# Initialize the Model
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads,
                           n_kv_head=num_kv_heads, n_embd=model_dim)
# For ZeRO-3 only
# with deepspeed.zero.Init(config_dict_or_path=deepspeed_config):
#     model_config = GPTConfig(**model_config_kwargs)
#     model = GPT(model_config)
# For ZeRO-2/1/0
model_config = GPTConfig(**model_config_kwargs)
set_num_kv_heads(model_config.n_kv_head)
set_n_embd(model_config.n_embd)
set_num_attention_heads(model_config.n_head)
set_tp_grain_size(128)
model = GPT(model_config, tp_group)
orig_model = model  # original, uncompiled model, for saving raw model state_dict

# num_params = sum(p.ds_numel for p in model.parameters())
# For both ZeRO-2 and ZeRO-3
num_params = sum(
    p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
    for p in model.parameters()
)
print0(f"Number of parameters: {num_params:,}")
with deepspeed.zero.GatheredParameters(list(model.parameters())):
    # num_params = sum(p.numel() for p in model.parameters())
    # print0(f"Number of parameters: {num_params:,}")
    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}")  # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
# optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
# adamw_optimizer, muon_optimizer = optimizers
param_groups = [
    {
        "params": [p for p in model.parameters()],
        "lr": matrix_lr,
        "initial_lr": matrix_lr,
    }
]

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=param_groups,
    config=deepspeed_config
)
print(f"Initialized deepspeed in rank: {ddp_rank}")

# Initialize the DataLoaders for train/val
base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")
train_loader = tokenizing_distributed_data_loader(dp_rank, dp_size, device_batch_size, max_seq_len, split="train",
                                                  device=device)
build_val_loader = lambda: tokenizing_distributed_data_loader(dp_rank, dp_size, device_batch_size, max_seq_len,
                                                              split="val", device=device)
x, y = next(train_loader)  # kick off load of the very first batch of data


# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


# -----------------------------------------------------------------------------
# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0  # EMA of training loss
ema_beta = 0.9  # EMA decay factor
total_training_time = 0  # total wall-clock time of training
# note that we run +1 steps only so that we can eval and save at the end
for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model_engine.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * dp_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model_engine.module, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        print0("11111111111111111")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        print0("22222222222222222")
        model_engine.train()
        print0("33333333333333333")

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model_engine.eval()
        orig_model = model_engine.module
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model_engine.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model_engine.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        orig_model = model_engine.module
        engine = Engine(orig_model, tokenizer)  # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        print0("Model eval done!")
        model_engine.train()

    # save checkpoint at the end of the run (only on master process)
    if last_step:
        # in deepspeed, every rank should save the checkpoint
        print0("last step!!!")
        output_dirname = model_tag if model_tag else f"d{depth}"  # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
        model_engine.save_checkpoint(checkpoint_dir, tag=f"step_{step}")
        print0(f"last step!!!save_checkpoint done! master_process: f{master_process}")
        if master_process:
            meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
            with open(meta_path, 'w') as f:
                json.dump({
                    "step": step,
                    "val_bpb": val_bpb,
                    "model_config": model_config_kwargs,
                    "user_config": user_config,
                    "device_batch_size": device_batch_size,
                    "max_seq_len": max_seq_len,
                }, f, indent=2)
        print0("last step!!!all done")

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()

    model_engine.optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        loss = model_engine(x, y)
        train_loss = loss.detach()  # for logging
        model_engine.backward(loss)

        # param_before = next(model_engine.parameters()).clone()
        # grad_before = model_engine.get_global_grad_norm()
        model_engine.step()
        # print0(f"After step - DeepSpeed micro_steps: {model_engine.micro_steps}")
        # param_after = next(model_engine.parameters())
        # grad_after = model_engine.get_global_grad_norm()
        # param_diff = (param_after - param_before).abs().max()
        # print0(f"Max parameter change: {param_diff:.6e}")
        # print0(f"Grad norm: {grad_before},{grad_after}")

        if micro_step < grad_accum_steps - 1:
            x, y = next(train_loader)

    # step the optimizers
    lrm = get_lr_multiplier(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["initial_lr"] * lrm
    print0(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))  # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size  # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # in %
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    grad_norm = model_engine.get_global_grad_norm()
    if grad_norm is not None:
        print_grad_norm = f" grad norm: {grad_norm:.4f} |"
    else:
        print_grad_norm = " grad norm: N/A |"
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time / 60:.2f}m")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/grad_norm": grad_norm,
        }
        wandb_run.log(log_data)

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report

get_report().log(section="Base model training", data=[
    user_config,  # CLI args
    {  # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "TP size": tp_size,
        "DP size": dp_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    {  # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time / 60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish()  # wandb run finish
compute_cleanup(model_engine)
