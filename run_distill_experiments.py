import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from distill import DistillationTrainer
from dataset import LMTextDataset
from model import Model
from matplotlib.cm import viridis

# Configuration
alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]  # Different mixing ratios to try
max_iters = 2000
temperature = 1.0
out_dir = 'models/distillation'

# Student configuration (smaller architecture)
student_config = {
    'd_model': 32,   # 1/16 of teacher size
    'n_heads': 4,
    'n_layers': 4
}

# Teacher configuration (larger architecture)
teacher_config = {
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 4
}

def run_experiments():
    os.makedirs(out_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = torch.load("data/train_dataset.pt")
    val_dataset = torch.load("data/val_dataset.pt")
    
    # Initialize teacher model and load weights
    teacher_model = Model(**teacher_config)
    teacher_model = torch.compile(teacher_model)
    teacher_ckpt = torch.load('models/ckpt-50000.pt')
    teacher_model.load_state_dict(teacher_ckpt['model_state_dict'])
    
    # Store results for each alpha
    results = {}
    
    for alpha in alpha_values:
        print(f"\nRunning experiment with alpha = {alpha}")
        
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            student_config=student_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            temperature=temperature,
            alpha=alpha,
            max_iters=max_iters,
            out_dir=out_dir
        )
        
        # Train and collect validation losses
        val_losses = []
        tokens_seen = []
        
        def log_callback(iter_count, val_loss):
            tokens = iter_count * trainer.train_loader.batch_size * trainer.student.max_seq_len
            val_losses.append(val_loss)
            tokens_seen.append(tokens)
        
        # Modify trainer.train() to call our callback
        trainer.train(callback=log_callback)
        
        # Store results
        results[alpha] = {
            'val_losses': val_losses,
            'tokens_seen': tokens_seen
        }
        
        # Save results
        with open(os.path.join(out_dir, 'results.json'), 'w') as f:
            json.dump(results, f)

        del trainer.student
        del trainer.teacher
        del trainer
        torch.cuda.empty_cache()

def plot_results():
    # Load results
    with open(os.path.join(out_dir, 'results.json'), 'r') as f:
        results = json.load(f)
    
    plt.figure(figsize=(12, 8))
    
    # Create color map
    alpha_values = sorted(list(map(float, results.keys())))
    colors = viridis(np.linspace(0, 1, len(alpha_values)))
    
    for alpha, color in zip(alpha_values, colors):
        tokens = results[str(alpha)]['tokens_seen']
        losses = results[str(alpha)]['val_losses']
        plt.plot(tokens, losses, label=f'α={alpha}', color=color, linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tokens', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.title('Distillation Performance with Different α Values', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiments()
    plot_results() 