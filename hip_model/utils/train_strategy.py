# -*- coding: utf-8 -*-
import math
import torch
import numpy as np
from utils.early_stopping import EarlyStopping
from config import Config

class WarmupScheduler:
    """改进的学习率预热调度器"""
    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_epoch = 0
        
        # 添加余弦退火
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.EPOCHS - warmup_epochs,
            eta_min=initial_lr
        )
    
    def step(self):
        """预热学习率调度器"""
        if self.current_epoch < self.warmup_epochs:
            # 线性预热
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * \
                 (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # 余弦退火
            self.cosine_scheduler.step()
        
        self.current_epoch += 1
        
    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr'] 