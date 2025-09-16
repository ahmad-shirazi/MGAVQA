"""
Learning rate schedulers for MGA-VQA training.
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing scheduler with warmup and restarts.
    
    Args:
        optimizer: Wrapped optimizer
        first_cycle_steps: Number of steps for the first cycle
        cycle_mult: Cycle multiplication factor (default: 1.0)
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate  
        warmup_steps: Number of warmup steps
        gamma: Decay factor for max_lr after each restart
    """
    
    def __init__(self, optimizer, first_cycle_steps: int, cycle_mult: float = 1.0,
                 max_lr: float = 0.1, min_lr: float = 0.001, 
                 warmup_steps: int = 0, gamma: float = 1.0, last_epoch: int = -1):
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # Initialize learning rate
        self.init_lr()
    
    def init_lr(self):
        """Initialize learning rates."""
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # Warmup phase
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            return [base_lr + (self.max_lr - base_lr) * 
                   (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                                (self.cur_cycle_steps - self.warmup_steps))) / 2
                   for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class PolynomialDecayLR(_LRScheduler):
    """
    Polynomial learning rate decay scheduler.
    
    Args:
        optimizer: Wrapped optimizer
        max_steps: Maximum number of training steps
        power: Power for polynomial decay (default: 1.0 for linear)
        min_lr: Minimum learning rate
    """
    
    def __init__(self, optimizer, max_steps: int, power: float = 1.0, 
                 min_lr: float = 0.0, last_epoch: int = -1):
        self.max_steps = max_steps
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.last_epoch == 0 or self.last_epoch > self.max_steps:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        decay_factor = (1 - self.last_epoch / self.max_steps) ** self.power
        return [max(self.min_lr, base_lr * decay_factor) for base_lr in self.base_lrs]


class WarmupLR(_LRScheduler):
    """
    Linear warmup scheduler.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        target_lr: Target learning rate after warmup
    """
    
    def __init__(self, optimizer, warmup_steps: int, target_lr: float, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [self.target_lr * self.last_epoch / self.warmup_steps 
                   for _ in self.base_lrs]
        else:
            # After warmup, use target learning rate
            return [self.target_lr for _ in self.base_lrs]


def create_scheduler(optimizer, config, scheduler_type: str = 'cosine'):
    """
    Factory function to create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration
        scheduler_type: Type of scheduler ('cosine', 'polynomial', 'warmup')
        
    Returns:
        scheduler: Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=config.max_epochs,
            max_lr=config.learning_rate,
            min_lr=config.learning_rate * 0.01,
            warmup_steps=config.warmup_steps,
            gamma=0.5
        )
    
    elif scheduler_type == 'polynomial':
        return PolynomialDecayLR(
            optimizer,
            max_steps=config.max_epochs,
            power=1.0,
            min_lr=config.learning_rate * 0.01
        )
    
    elif scheduler_type == 'warmup':
        return WarmupLR(
            optimizer,
            warmup_steps=config.warmup_steps,
            target_lr=config.learning_rate
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
