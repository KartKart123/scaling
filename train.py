import math
import os
import json
import torch
from model import Model
from dataset import LMTextDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

#Config
out_dir = 'models'
eval_interval = 500
log_interval = 50
max_checkpoint = 3  # max number of checkpoints to save

# tensorboard logging
tb_log = True
run_count = 0

# data
batch_size = 32
batch_size_per_device = 32
gradient_accumulation_steps = batch_size // batch_size_per_device
max_seq_len = 256 #128

# model
d_model = 64
n_heads = 4
n_layers = 4

# adamw optimizer
learning_rate = 5e-4  # max learning rate
max_iters = 50000   # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 500
lr_decay_iters = 50_000
min_lr = learning_rate / 10

# system
device = 'cuda'
dtype = 'bfloat16' 
compile_model = True 

def get_lr(it):
    """
    Learning rate decay schedule:
    - warm up for warmup_iters steps
    - then cosine decay from learning_rate to min_lr
    """
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    progress = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1.0 + math.cos(math.pi * progress))

def train(d_model, n_heads, n_layers, run_count):
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = torch.load("data/train_dataset.pt")
    val_dataset   = torch.load("data/val_dataset.pt")

    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_device, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size_per_device, shuffle=False)

    vocab_size = tokenizer.vocab_size
    model = Model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len
    )

    if dtype == 'bfloat16':
        model = model.to(dtype=torch.bfloat16)

    model.to(device)

    if compile_model:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )

    if tb_log:
        from torch.utils.tensorboard import SummaryWriter
        run_id = f"run_{run_count}"
        tb_writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tb_logs', run_id))
    else:
        tb_writer = None

    @torch.no_grad()
    def eval_split(loader):
        model.eval()
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x, targets=y) 
            losses.append(loss.item())
        model.train()
        return float(sum(losses) / len(losses))

    iter_count = 0 
    running_loss = 0.0
    datalog = []
    flop_per_token = get_flop_per_token(d_model, n_layers)

    while True:
        for x, y in train_loader:
            iter_count += 1
            lr_now = get_lr(iter_count) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_now

            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, targets=y)

            loss.backward()

            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Gradient accumulation
            if (iter_count % gradient_accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

            if iter_count % log_interval == 0:
                avg_loss = running_loss / log_interval
                print(f"Iter {iter_count}: loss {avg_loss:.4f}, lr {lr_now:e}")
                if tb_writer:
                    tb_writer.add_scalar("train/loss", avg_loss, iter_count)
                    tb_writer.add_scalar("train/lr", lr_now, iter_count)
                running_loss = 0.0

            # Evaluation & checkpoint
            if iter_count % eval_interval == 0:
                val_loss = eval_split(val_loader)
                print(f"Iter {iter_count}: val loss {val_loss:.4f}")
                if tb_writer:
                    tb_writer.add_scalar("val/loss", val_loss, iter_count)
                
                tokens = batch_size_per_device * max_seq_len * iter_count
                flop = flop_per_token * tokens
                datalog.append({'loss': val_loss, 'tokens': tokens, 'compute': flop, 'iter': iter_count, 'params': 12 * n_layers * d_model**2})

                # # Save checkpoint if it's good or just rotate
                # ckpt_path = os.path.join(out_dir, f'ckpt-{iter_count}.pt')
                # torch.save({'model_state_dict': model.state_dict(),
                #             'iter_count': iter_count,
                #             'val_loss': val_loss}, ckpt_path)
                # _cleanup_old_ckpts()

            if iter_count >= max_iters:
                print("Training complete.")
                ckpt_path = os.path.join(out_dir, f'ckpt-{iter_count}.pt')
                torch.save({'model_state_dict': model.state_dict(),
                            'iter_count': iter_count,
                            'val_loss': val_loss}, ckpt_path)
                run_id = f"data_{run_count}"
                with open(os.path.join(out_dir, run_id), 'w') as f:
                    json.dump(datalog, f)
                return datalog


def _cleanup_old_ckpts():
    # utility to remove older checkpoints,
    # so we only keep the last 'max_checkpoint'.
    ckpts = [f for f in os.listdir(out_dir) if f.startswith("ckpt-") and f.endswith(".pt")]
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    if len(ckpts_sorted) > max_checkpoint:
        to_remove = ckpts_sorted[:-max_checkpoint]
        for ckpt in to_remove:
            os.remove(os.path.join(out_dir, ckpt))

def get_flop_per_token(d_model, num_layers):
    total_flop = 6 * 12 * num_layers * d_model**2 # Kaplan
    return total_flop

if __name__ == "__main__":
    train(d_model, n_heads, n_layers, run_count)