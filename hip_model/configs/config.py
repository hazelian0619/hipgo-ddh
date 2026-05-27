class Config:
    # 数据集配置
    IMAGE_SIZE = 224
    BATCH_SIZE = 16  # 先用小batch测试
    NUM_WORKERS = 4
    
    # 模型配置
    PRETRAINED = True
    
    # 训练配置
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-6
    EPOCHS = 10  # 先跑10轮测试
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    
    # 损失函数权重
    POINT_WEIGHT = 1.0
    ANGLE_WEIGHT = 0.5
    
    # 验证配置
    VAL_FREQUENCY = 1
    
    # 保存配置
    CHECKPOINT_DIR = 'checkpoints'
    
    # 优化配置
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
