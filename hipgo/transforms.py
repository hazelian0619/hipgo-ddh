import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(train=True, img_size=512):
    """获取数据转换
    
    Args:
        train (bool): 是否为训练集
        img_size (int): 图像大小
    
    Returns:
        A.Compose: 数据转换组合
    """
    if train:
        return A.Compose([
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=0.9,
                translate_percent=0.1,
                rotate=(-10, 10),
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
            ], p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                fill=0,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])

def get_prediction_transforms(img_size=512):
    """获取预测用的数据转换（不需要关键点参数）"""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            value=0,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def get_train_transforms():
    """获取训练数据增强转换"""
    return get_transforms(train=True)

def get_val_transforms():
    """获取验证数据增强转换"""
    return get_transforms(train=False) 