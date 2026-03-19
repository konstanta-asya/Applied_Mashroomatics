from torchvision import transforms
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(mode='train', img_size=224, augmentation_strength='standard'):
    """
    Get transforms for training or validation.

    Args:
        mode: 'train' or 'val'
        img_size: target image size
        augmentation_strength: 'light', 'standard', or 'strong'
    """
    if mode == 'train':
        if augmentation_strength == 'light':
            return transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        elif augmentation_strength == 'strong':
            # Optimized for fungi classification - color preserved
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                # RandomGrayscale REMOVED — color is a key diagnostic feature for fungi
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
            ])
        else:  # standard
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                # RandomGrayscale REMOVED — color is a key diagnostic feature for fungi
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
            ])
    else:
        # Validation transform with CenterCrop for consistency
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def get_tta_transforms(img_size=224, n_augments=5):
    """
    Test-Time Augmentation transforms.
    Returns a list of transform pipelines for TTA.
    """
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    tta_transforms = [base_transform]

    if n_augments >= 2:
        tta_transforms.append(transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]))

    if n_augments >= 3:
        tta_transforms.append(transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]))

    if n_augments >= 4:
        # Deterministic rotation +15
        tta_transforms.append(transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.Lambda(lambda x: TF.rotate(x, 15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]))

    if n_augments >= 5:
        # Deterministic rotation -15
        tta_transforms.append(transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.Lambda(lambda x: TF.rotate(x, -15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ]))

    return tta_transforms