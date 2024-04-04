import albumentations as A


cls_dict = {
    0: 'Background',
    1: 'Building',
    2: 'Road',
    3: 'Vehicle',
    4: 'Debris',
    5: 'Fire',
    6: 'Water',
    7: 'Animal',
    8: 'Injured_Person', # Keeping this class for convenience, since it is part of the annotations. With a small dataset it does not make sense to distinguish healthy and injured persons, however.
    9: 'Sky',
    10: 'Smoke',
    11: 'Tree',
    12: 'Person'
}
        

def get_augmentations(mode='train'):
    """
    On-the-Fly augmentations to increase training dataset size artificially. 
    """
    if mode == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.4,
                rotate_limit=(-25, 25), border_mode=0,
                shift_limit_x=0.2,
                shift_limit_y=0.2),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.5), contrast_limit=0.7, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-180, 50),
                val_shift_limit=0,
                p=0.5),
            A.RandomToneCurve(p=0.2),
            A.ISONoise(p=0.1),
            A.OneOf([
                A.MedianBlur(blur_limit=5, p=0.5),
                A.GaussianBlur(blur_limit=5, p=0.5)
            ], p=0.1),
            A.ImageCompression(quality_lower=35, p=0.3),
            A.Perspective(scale=(0.05, 0.13), p=0.1),
            A.Cutout(num_holes=8, max_h_size=40, max_w_size=40, p=0.05),
            A.LongestMaxSize(max_size=512, p=1),
            A.PadIfNeeded(512, 512, border_mode=0, p=1),
        ],
            p=1.0)

    elif mode == 'val':
        return None

