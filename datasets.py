import torch
from torchvision import transforms
from data_constants import (IMAGE_TASKS, IMAGENET_DEFAULT_MEAN,
                             IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN,
                             IMAGENET_INCEPTION_STD)
class DataAugmentation(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        trans = [transforms.RandomResizedCrop(args.input_size)]
        if args.hflip > 0.0:
            trans.append(transforms.RandomHorizontalFlip(args.hflip))
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))])

        self.transform = transforms.Compose(trans)

    def __call__(self, image):
        return self.transform(image)

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr

