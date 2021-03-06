# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import wandb
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import data_constants
import yaml
from ConvAutoEncoder import *
from utils import get_device
from utils import save_decoded_image
from utils import save_og_image
from utils import Average
from save_results import save_csv
import os
import pathlib

def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MultiMAE pre-training script', add_help=False)

    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (default: %(default)s)')
    parser.add_argument('--epochs', default=1600, type=int,
                        help='Number of epochs (default: %(default)s)')
    parser.add_argument('--save_ckpt_freq', default=20, type=int,
                        help='Checkpoint saving frequency in epochs (default: %(default)s)')

    # Task parameters
    parser.add_argument('--in_domains', default='rgb_drone-thermal', type=str,
                        help='Input domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--out_domains', default='rgb_drone', type=str,
                        help='Output domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--standardize_depth', action='store_true')
    parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
    parser.set_defaults(standardize_depth=False)
    parser.add_argument('--extra_norm_pix_loss', action='store_true')
    parser.add_argument('--no_extra_norm_pix_loss', action='store_false', dest='extra_norm_pix_loss')
    parser.set_defaults(extra_norm_pix_loss=True)

    # Model parameters
    parser.add_argument('--model', default='pretrain_multimae_base', type=str, metavar='MODEL',
                        help='Name of model to train (default: %(default)s)')
    parser.add_argument('--num_encoded_tokens', default=98, type=int,
                        help='Number of tokens to randomly choose for encoder (default: %(default)s)')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='Number of global tokens to add to encoder (default: %(default)s)')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Base patch size for image-like modalities (default: %(default)s)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Images input size for backbone (default: %(default)s)')
    parser.add_argument('--alphas', type=float, default=1.0,
                        help='Dirichlet alphas concentration parameter (default: %(default)s)')
    parser.add_argument('--sample_tasks_uniformly', default=False, action='store_true',
                        help='Set to True/False to enable/disable uniform sampling over tasks to sample masks for.')

    parser.add_argument('--decoder_use_task_queries', default=True, action='store_true',
                        help='Set to True/False to enable/disable adding of task-specific tokens to decoder query tokens')
    parser.add_argument('--decoder_use_xattn', default=True, action='store_true',
                        help='Set to True/False to enable/disable decoder cross attention.')
    parser.add_argument('--decoder_dim', default=256, type=int,
                        help='Token dimension inside the decoder layers (default: %(default)s)')
    parser.add_argument('--decoder_depth', default=2, type=int,
                        help='Number of self-attention layers after the initial cross attention (default: %(default)s)')
    parser.add_argument('--decoder_num_heads', default=8, type=int,
                        help='Number of attention heads in decoder (default: %(default)s)')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: %(default)s)')

    parser.add_argument('--loss_on_unmasked', default=False, action='store_true',
                        help='Set to True/False to enable/disable computing the loss on non-masked tokens')
    parser.add_argument('--no_loss_on_unmasked', action='store_false', dest='loss_on_unmasked')
    parser.set_defaults(loss_on_unmasked=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: %(default)s)')
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: %(default)s)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='CLIPNORM',
                        help='Clip gradient norm (default: %(default)s)')
    parser.add_argument('--skip_grad', type=float, default=None, metavar='SKIPNORM',
                        help='Skip update if gradient norm larger than threshold (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: %(default)s)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD.  (Set the same value as args.weight_decay to keep weight decay unchanged)""")
    parser.add_argument('--decoder_decay', type=float, default=None, help='decoder weight decay')

    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: %(default)s)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='Warmup learning rate (default: %(default)s)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0 (default: %(default)s)')
    parser.add_argument('--task_balancer', type=str, default='none',
                        help='Task balancing scheme. One out of [uncertainty, none] (default: %(default)s)')
    parser.add_argument('--balancer_lr_scale', type=float, default=1.0,
                        help='Task loss balancer LR scale (if used) (default: %(default)s)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')

    parser.add_argument('--fp32_output_adapters', type=str, default='',
                        help='Tasks output adapters to compute in fp32 mode, separated by hyphen.')

    # Augmentation parameters
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Probability of horizontal flip (default: %(default)s)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic) (default: %(default)s)')

    # Dataset parameters
    parser.add_argument('--data_path', default='../Processed/rgbdrone', type=str, help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    # Misc.
    parser.add_argument('--output_dir', default='output/sweep_rgbdrone_100e',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')

    parser.add_argument('--seed', default=0, type=int, help='Random seed ')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=True)

    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='Log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='Project name on wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='User or team name on wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='Run name on wandb')
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return args


def main(opts):
    # Create a folder for the output
    pathlib.Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    # Transforms images to a PyTorch Tensor
    # tensor_transform = transforms.ToTensor()
    # image transformations
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(IMAGENET_INCEPTION_MEAN),std=torch.tensor(IMAGENET_INCEPTION_STD)),
        transforms.RandomResizedCrop(opts.input_size)
    ])
    # Download the MNIST Dataset
    dataset = datasets.ImageFolder(root=opts.data_path,
                                   transform=transform)

    # DataLoader is used to load the dataset
    # for training
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=opts.batch_size,
                                         shuffle=True)
    # Model Initialization
    model = ConvAutoEncoder()

    # get the computation device
    device = get_device()
    print(device)
    # load the neural network onto the device
    model.to(device)
    # train the network
    train(loader, model, opts.epochs, device)


def train(loader, model, epochs, device):
    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()
    #opts.lr = opts.blr * opts.batch_size / 256
    opts.lr = opts.blr
    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opts.blr)
    outputs = []
    losses = []
    train_dict = {}
    for epoch in range(config["epochs"]):
        ep_losses = []
        for (image, _) in loader:
            # Reshaping the image to (-1, 784)
            #image = image.reshape(-1, opts.input_size, opts.input_size)
            #image = image.unsqueeze(0)  # if torch tensor
            image = image.to(device)
            # Output of Autoencoder
            reconstructed = model(image, False)

            # Calculating the loss function
            loss = loss_function(reconstructed, image)

            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Storing the losses in a list for plotting
            ep_losses.append(loss)
            #print(f'Epoch %d: Loss is %f', )
            if epoch % 5 == 0:
                testimg = image[0].unsqueeze(0)
                testimg = testimg.to(device)
                test_reconst = model(testimg, False)
                save_decoded_image(opts.output_dir,test_reconst.cpu(), epoch)
                save_og_image(opts.output_dir,testimg,epoch)
        avg_loss = Average(ep_losses)
        avg_loss = avg_loss.cpu().detach().numpy()
        wandb.log({"loss": loss})

        # Optional
        wandb.watch(model)
        losses.append(avg_loss)
        # Save the results
        save_csv(opts.output_dir, epoch, avg_loss)
        print(f"Epoch {epoch}: Loss is {avg_loss}")

    # Defining the Plot Style
    plt.plot(losses)
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(opts.output_dir+'/lossplot.png')

    # for i, item in enumerate(image):
    #     # Reshape the array for plotting
    #     item = item.reshape(-1, opts.input_size, opts.input_size)
    #     im = Image.fromarray(item[0])
    #     im.save(f"og%i.png")
    #
    # for i, item in enumerate(reconstructed):
    #     item = item.reshape(-1, opts.input_size, opts.input_size)
    #     im = Image.fromarray(item[0])
    #     im.save(f"re%i.png")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    opts = get_args()
    wandb.init(config=opts, project="AE", entity="plantscience")
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    main(config)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
