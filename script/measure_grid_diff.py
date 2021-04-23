import argparse

import imageio
from skimage import io
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from skimage.exposure import rescale_intensity


def parse_args():
    parser = argparse.ArgumentParser(description='Make a grid from a reconstucted tensor.')
    parser.add_argument('img1', type=str, help='Path to numpy representation of a reconstructed tensor')
    parser.add_argument('img2', type=str, help='Path to numpy representation of a reconstructed tensor')
    parser.add_argument('-O', '--out', type=str, help='Save path for the grid', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    img1 = io.imread(args.img1)
    img2 = io.imread(args.img2)

    img = img1 - img2
    img = np.clip(img, 0, 255)
    io.imsave(args.out, img)
    

if __name__ == '__main__':
    main()