import torch

from torchvision.utils import save_image


# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def save_decoded_image(img, epoch):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, './Training_output/cnv_ae_image{}.png'.format(epoch))

def save_og_image(img, epoch):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, './Training_output/cnv_og_image{}.png'.format(epoch))

def Average(lst):
    return sum(lst) / len(lst)
