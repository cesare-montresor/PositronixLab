import torch
from torchvision import transforms


def _flatten(x):
    return torch.flatten(x, start_dim=0)


def _to_float(x):
    return x.float()


def _to_long(x):
    return x.long()


class MetaTransform():
    """
    A class to hold options for the MetaTransform.
    This class can be extended to include various configuration parameters.
    """
    def __init__(self, transform_list):
        self.transform_list = transform_list
        self.transforms = None

        ts = []
        for name in self.transform_list:

            if name == 'normalize':
                t = transforms.Normalize(mean=[0.5], std=[0.5])
            elif name == 'flatten':
                t = transforms.Lambda(_flatten)
            elif name == 'to_tensor':
                t = transforms.ToTensor()
            elif name == 'to_float':
                t = transforms.Lambda(_to_float)
            elif name == 'to_long':
                t = transforms.Lambda(_to_long)
            elif name == 'normalize_image':
                t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            elif name == 'augment_image':
                t = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(360),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                raise ValueError(f"Unsupported transform type: {name}")
            ts.append(t)
        self.transforms = transforms.Compose(ts)

    def apply(self, x):
        return self.transforms(x)

    def __call__(self, x):
        return self.apply(x)
