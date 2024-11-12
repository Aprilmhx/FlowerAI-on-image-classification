import albumentations as albu

def get_training_augmentation(height, width):
    train_transform = [
        albu.Resize(height=height, width=width, p = 1.0),
        albu.augmentations.transforms.HorizontalFlip(always_apply=False, p=0.0),
        albu.augmentations.transforms.VerticalFlip(always_apply=False, p=0.0),
        albu.OneOf(
            [
                albu.CLAHE(p=0.3),
                albu.RandomGamma(p=0.4),
                albu.augmentations.transforms.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.08, always_apply=False, p=0.4),
            ],
            p=0.0,
        ),  

        albu.OneOf(
            [
                albu.Blur(blur_limit=3, p=0.3),
                albu.MotionBlur(blur_limit=3, p=0.4),
                albu.augmentations.transforms.GaussianBlur (blur_limit=(3, 7), sigma_limit=0, p=0.4)
            ],
            p=0.0,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1.0),
                albu.HueSaturationValue(p=0.0),
            ],
            p=0.0,
        ),
        
        albu.OneOf(
            [
                albu.augmentations.transforms.Flip (p = 0.3),
                albu.augmentations.transforms.HorizontalFlip(p=0.4),
                albu.augmentations.transforms.VerticalFlip(p=0.4)
            ],
            p=0.0,
        ),
        albu.augmentations.transforms.GridDistortion(num_steps=6, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.0),
        albu.augmentations.transforms.OpticalDistortion(distort_limit=0.15, shift_limit=0.15, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.0),
      
        albu.augmentations.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
                        

    ]
    
    
    return albu.Compose(train_transform)


def get_validation_augmentation(height, width):
    """Add paddings to make image shape divisible by 32"""    
    test_transform = [
        albu.Resize(height=height, width=width, p = 1.0),
        albu.augmentations.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
    ]
    return albu.Compose(test_transform)