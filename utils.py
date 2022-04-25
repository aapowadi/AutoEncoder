import torch

from torchvision.utils import save_image


# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def save_decoded_image(output_dir, img, epoch):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, output_dir+'/re{}.png'.format(epoch))

def save_og_image(output_dir, img, epoch):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, output_dir+'/og{}.png'.format(epoch))

def Average(lst):
    return sum(lst) / len(lst)
