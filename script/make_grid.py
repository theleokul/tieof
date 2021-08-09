import argparse

import imageio
from skimage import io
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


def parse_args():
    parser = argparse.ArgumentParser(description='Make a grid from a reconstucted tensor.')
    parser.add_argument('tensor', type=str, help='Path to numpy representation of a reconstructed tensor')
    parser.add_argument('-O', '--out', type=str, help='Save path for the grid', required=True)
    parser.add_argument('--log', action='store_true', help='Activate logarithmic scale')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    tensor = torch.tensor(np.load(args.tensor))
    tensor.unsqueeze_(0)  # Add color channel
    tensor = tensor.permute(3, 0, 1, 2)  # Move time to the beginning
    # save_image(tensor, args.out)
    img = make_grid(tensor, normalize=True, value_range=(tensor[~torch.isnan(tensor)].min(), tensor[~torch.isnan(tensor)].max()))
    img = img.permute(1, 2, 0)

    if args.log:
        img[img < 1e-8] += 1e-8
        img[~torch.isnan(img)] = torch.log(img[~torch.isnan(img)]).abs()
        img /= img[~torch.isnan(img)].max()
        img = 1 - img

    img[torch.isnan(img)] = 0
    img = img.numpy() * 255
    img = img.astype(np.uint8)

    io.imsave(args.out, img)
    

if __name__ == '__main__':
    main()