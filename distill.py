import torch
import torch.nn.functional as F
from model import Model
from torch.utils.data import DataLoader
import os
import math
from torch.utils.tensorboard import SummaryWriter

learning_rate=5e-4
warmup_iters = 200
lr_decay_iters = 2_000
min_lr = learning_rate / 10

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

class DistillationTrainer:
    def __init__(
        self,
        teacher_model,
        student_config,
        train_dataset,
        val_dataset,
        temperature=1.0,
        alpha=0.5,
        batch_size=32,
        max_iters=6000,
        device='cuda',
        out_dir='models/distillation'
    ):
        self.teacher = teacher_model.to(dtype=torch.bfloat16).to(device)
        self.teacher.eval()  # Teacher model should always be in eval mode
        
        # Initialize student model with smaller architecture
        self.student = Model(
            vocab_size=teacher_model.head.weight.shape[0],
            d_model=student_config['d_model'],
            n_heads=student_config['n_heads'],
            n_layers=student_config['n_layers'],
            max_seq_len=teacher_model.max_seq_len
        ).to(dtype=torch.bfloat16).to(device)
        self.student = torch.compile(self.student)
        
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        
        # Setup data
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Setup logging
        os.makedirs(out_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tb_logs', f'alpha_{alpha}'))
        self.out_dir = out_dir

    def distillation_loss(self, student_logits, teacher_logits, targets):
        """
        Compute the distillation loss:
        - soft_loss: KL divergence between softened teacher and student predictions
        - hard_loss: regular cross-entropy with true labels
        """
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        
        # Soften the logits by dividing by temperature
        soft_targets = F.log_softmax(teacher_logits / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Compute the soft and hard losses
        soft_loss = F.kl_div(
            soft_predictions,
            soft_targets,
            reduction='batchmean',
            log_target=True
        ) * (self.temperature ** 2)
        
        hard_loss = F.cross_entropy(student_logits, targets.view(-1))
        
        # Combine the losses
        loss = (self.alpha * soft_loss) + ((1 - self.alpha) * hard_loss)
        return loss, soft_loss, hard_loss

    @torch.no_grad()
    def evaluate(self, loader):
        self.student.eval()
        total_loss = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Get student predictions
            _, loss = self.student(x, targets=y)
            total_loss += loss.item()
            
        self.student.train()
        return total_loss / len(loader)

    def train(self, callback=None):
        iter_count = 0
        
        while iter_count < self.max_iters:
            for x, y in self.train_loader:
                iter_count += 1

                lr_now = get_lr(iter_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_now
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits, _ = self.teacher(x)
                
                # Get student predictions
                student_logits, _ = self.student(x)
                
                # Compute distillation loss
                loss, soft_loss, hard_loss = self.distillation_loss(
                    student_logits, teacher_logits, y
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Logging
                if iter_count % 50 == 0:
                    val_loss = self.evaluate(self.val_loader)
                    print(f"Iter {iter_count}: val_loss {val_loss:.4f}")
                    
                    # Tensorboard logging
                    self.writer.add_scalar('train/total_loss', loss.item(), iter_count)
                    self.writer.add_scalar('train/soft_loss', soft_loss.item(), iter_count)
                    self.writer.add_scalar('train/hard_loss', hard_loss.item(), iter_count)
                    self.writer.add_scalar('val/loss', val_loss, iter_count)
                    
                    # Call callback if provided
                    if callback:
                        callback(iter_count, val_loss)
                
                if iter_count >= self.max_iters:
                    break
        
        return self.student 